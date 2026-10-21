import ROSLIB from "roslib";
import { HeadTeleopClient, HEAD_COMMAND_TOPIC, HEAD_JOINTS, HEAD_LIMITS, HEAD_TOPIC, headPositions, headTrajectory } from "./headTeleop";

jest.mock("roslib", () => ({
  Topic: jest.fn().mockImplementation((options) => ({
    ...options, subscribe: jest.fn(), unsubscribe: jest.fn(), publish: jest.fn(), unadvertise: jest.fn(),
  })),
  Service: jest.fn().mockImplementation(() => ({ callService: jest.fn() })),
  Message: jest.fn().mockImplementation((data) => data),
  ServiceRequest: jest.fn().mockImplementation((data) => data),
}));
let time, ros, client;
const state = (positions = [0.5, 0]) => client.subscription.subscribe.mock.calls[0][0]({ name: HEAD_JOINTS, position: positions });
const ready = () => {
  state();
  client.commandSubscription.subscribe.mock.calls[0][0]({ header: { frame_id: "leader_joystick_controller" } });
  client.tick();
  client.service.callService.mock.calls.at(-1)[1]({ controller: [{ name: "head_controller", state: "active" }] });
};
beforeEach(() => {
  ROSLIB.Topic.mockImplementation((options) => ({ ...options, subscribe: jest.fn(), unsubscribe: jest.fn(), publish: jest.fn(), unadvertise: jest.fn() }));
  ROSLIB.Service.mockImplementation(() => ({ callService: jest.fn() }));
  ROSLIB.Message.mockImplementation((data) => data);
  ROSLIB.ServiceRequest.mockImplementation((data) => data);
  time = 0; ros = { isConnected: true }; client = new HeadTeleopClient(ros, () => time); });
afterEach(() => client.dispose());

test("requires fresh measured joints and active controller; activation alone sends nothing", () => {
  expect(client.activate()).toBe(false);
  state(); expect(client.activate()).toBe(false);
  ready(); expect(client.activate()).toBe(true);
  expect(client.publisher.publish).not.toHaveBeenCalled();
  expect(client.publisher.name).toBe(HEAD_COMMAND_TOPIC);
  expect(client.commandSubscription.name).toBe(HEAD_TOPIC);
});
test("a stopped leader stream blocks activation and cancels active teleop", () => {
  state(); client.tick();
  client.service.callService.mock.calls.at(-1)[1]({ controller: [{ name: "head_controller", state: "active" }] });
  expect(client.blocker()).toMatch(/Waiting for leader head command stream/);
  ready(); expect(client.activate()).toBe(true);
  time = 1001; state(); client.tick();
  expect(client.active).toBe(false);
  expect(client.publisher.publish).not.toHaveBeenCalled();
});
test("rejects the actual 1022 out-of-range posture and malformed state", () => {
  ready(); state([0.926524, 0.004602]);
  expect(client.blocker()).toMatch(/Pitch outside/);
  expect(client.activate()).toBe(false);
  state([0.5, NaN]); expect(client.positions).toBeNull();
  expect(headPositions({ name: [...HEAD_JOINTS, HEAD_JOINTS[0]], position: [0, 0, 0] })).toBeNull();
  expect(client.publisher.publish).not.toHaveBeenCalled();
});
test("jogs from measured position with a finite horizon; repeated commands never accumulate", () => {
  ready(); client.activate(); client.start("up");
  for (let n = 1; n <= 4; n += 1) { time = n * 150; state(); client.tick(); }
  for (const [message] of client.publisher.publish.mock.calls) {
    expect(message.joint_names).toEqual(HEAD_JOINTS);
    expect(message.points[0].positions).toEqual([0.485, 0]);
    expect(message.points[0].time_from_start).toEqual({ sec: 0, nanosec: 300000000 });
  }
  state([0.49, 0.002]); client.stop();
  expect(client.publisher.publish.mock.calls.at(-1)[0].points[0].positions).toEqual([0.49, 0.002]);
});
test.each(["up", "down", "left", "right"])("%s stays in bounds even at boundaries", (direction) => {
  for (const pitch of [HEAD_LIMITS[0][0], 0, HEAD_LIMITS[0][1]]) {
    for (const yaw of [HEAD_LIMITS[1][0], 0, HEAD_LIMITS[1][1]]) {
      const result = headTrajectory([pitch, yaw], direction).points[0].positions;
      result.forEach((value, i) => {
        expect(value).toBeGreaterThanOrEqual(HEAD_LIMITS[i][0]);
        expect(value).toBeLessThanOrEqual(HEAD_LIMITS[i][1]);
        expect(Math.abs(value - [pitch, yaw][i])).toBeLessThanOrEqual(0.015001);
      });
    }
  }
});
test("does not send a queued stop or resume after disconnect", () => {
  ready(); client.activate(); client.start("left");
  ros.isConnected = false; client.tick();
  expect(client.active).toBe(false);
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
  ros.isConnected = true; client.tick();
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
});
test("stale telemetry cancels activation and sends no unverified hold", () => {
  ready(); client.activate(); client.start("down");
  time = 800; client.tick();
  expect(client.active).toBe(false);
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
  state(); client.tick(); expect(client.direction).toBeNull();
});
test("stale or inactive controller blocks commands", () => {
  ready(); client.activate();
  time = 2100; state(); client.tick();
  expect(client.active).toBe(false);
  client.service.callService.mock.calls.at(-1)[1]({ controller: [{ name: "head_controller", state: "unconfigured" }] });
  client.commandSubscription.subscribe.mock.calls[0][0]({ header: { frame_id: "leader_joystick_controller" } });
  expect(client.blocker()).toBe("Head controller unconfigured");
  expect(client.start("down")).toBe(false);
});
test("competing commands deactivate teleop; own echoes do not", () => {
  ready(); client.activate(); client.start("left");
  const onCommand = client.commandSubscription.subscribe.mock.calls[0][0];
  onCommand(client.publisher.publish.mock.calls[0][0]); expect(client.active).toBe(true);
  onCommand({ header: { frame_id: "leader_joystick_controller" } });
  expect(client.active).toBe(true);
  onCommand({ header: { frame_id: "" } });
  expect(client.active).toBe(false); expect(client.blocker()).toMatch(/Another head/);
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
});
test("dispose holds once, unsubscribes, and ignores late callbacks", () => {
  ready(); client.activate(); client.start("up"); client.dispose();
  expect(client.publisher.publish).toHaveBeenCalledTimes(2);
  expect(client.subscription.unsubscribe).toHaveBeenCalled();
  expect(client.commandSubscription.unsubscribe).toHaveBeenCalled();
  client.service.callService.mock.calls[0][1]({ controller: [{ name: "head_controller", state: "active" }] });
  expect(client.activate()).toBe(false);
});
test("a delayed controller response cannot revive its expired request", () => {
  state();
  client.commandSubscription.subscribe.mock.calls[0][0]({ header: { frame_id: "leader_joystick_controller" } });
  client.tick(); const callback = client.service.callService.mock.calls[0][1];
  time = 1001; state(); callback({ controller: [{ name: "head_controller", state: "active" }] });
  expect(client.activate()).toBe(false);
});
