import ROSLIB from 'roslib';

export const DAGGER_INITIAL = { state: 'IDLE', busy: false, pending: null, online: false,
  gateway: {}, sensor_reason: '', error: '', profiles: [] };

export class DaggerClient {
  constructor(ros, clock = () => Date.now()) {
    this.ros = ros;
    this.clock = clock;
    this.receivedAt = 0;
    this.value = { ...DAGGER_INITIAL };
    this.status = new ROSLIB.Topic({ ros, name: '/dagger/status', messageType: 'std_msgs/String',
      queue_length: 1, reconnect_on_close: false });
    this.commandTopic = new ROSLIB.Topic({ ros, name: '/dagger/command', messageType: 'std_msgs/String',
      reconnect_on_close: false });
    // roslib 1.4.1 loses the Ros receiver with reconnect_on_close=false.
    // Bind both status subscription and command advertisement to this connection;
    // neither operation should queue across disconnect/reconnect.
    [this.status, this.commandTopic].forEach((topic) => {
      topic.callForSubscribeAndAdvertise = (message) => {
        if (this.ros.isConnected) this.ros.callOnConnection(message);
      };
    });
    this.status.subscribe((message) => {
      try {
        const next = JSON.parse(message.data);
        if (typeof next.state !== 'string' || typeof next.busy !== 'boolean') return;
        this.value = { ...DAGGER_INITIAL, ...next, profiles: Array.isArray(next.profiles) ? next.profiles : [], gateway: next.gateway || {} };
        this.receivedAt = this.clock();
      } catch (_) { /* malformed status never enables a control */ }
    });
    this.lastHeartbeat = 0;
  }

  snapshot() {
    return { ...this.value, online: this.ros.isConnected && this.receivedAt > 0 &&
      this.clock() - this.receivedAt < 1500 };
  }

  command(op, fields = {}) {
    if (!this.snapshot().online) throw new Error('DAgger 연결을 확인해주세요.');
    const request_id = `${this.clock()}-${Math.random().toString(16).slice(2)}`;
    this.commandTopic.publish(new ROSLIB.Message({ data: JSON.stringify({ ...fields, op, request_id }) }));
    return request_id;
  }

  heartbeat(enabled) {
    if (!enabled || !this.snapshot().online || this.clock() - this.lastHeartbeat < 500) return;
    this.lastHeartbeat = this.clock();
    this.command('heartbeat');
  }

  dispose() {
    this.status.unsubscribe();
    if (this.ros.isConnected) this.commandTopic.unadvertise();
  }
}
