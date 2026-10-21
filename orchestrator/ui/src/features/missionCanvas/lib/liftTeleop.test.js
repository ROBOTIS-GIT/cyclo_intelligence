import ROSLIB from "roslib";
import { LiftTeleopClient, LIFT_JOINTS, LIFT_LIMITS, LIFT_TOPIC, liftPositions, liftTrajectory } from "./liftTeleop";

jest.mock("roslib", () => ({
  Topic: jest.fn().mockImplementation((options) => ({
    ...options, subscribe: jest.fn(), unsubscribe: jest.fn(), publish: jest.fn(), unadvertise: jest.fn(),
  })),
  Service: jest.fn().mockImplementation(() => ({ callService: jest.fn() })),
  Message: jest.fn().mockImplementation((data) => data),
  ServiceRequest: jest.fn().mockImplementation((data) => data),
}));
let time, ros, client;
const state = (positions = [-0.2]) => client.subscription.subscribe.mock.calls[0][0]({ name: LIFT_JOINTS, position: positions });
const ready = () => {
  state(); client.tick();
  client.service.callService.mock.calls.at(-1)[1]({ controller: [{ name: "lift_controller", state: "active" }] });
};
beforeEach(() => {
  ROSLIB.Topic.mockImplementation((options) => ({ ...options, subscribe: jest.fn(), unsubscribe: jest.fn(), publish: jest.fn(), unadvertise: jest.fn() }));
  ROSLIB.Service.mockImplementation(() => ({ callService: jest.fn() }));
  ROSLIB.Message.mockImplementation((data) => data);
  ROSLIB.ServiceRequest.mockImplementation((data) => data);
  time = 0; ros = { isConnected: true }; client = new LiftTeleopClient(ros, () => time); });
afterEach(() => client.dispose());

test("requires fresh measured joints and active controller; activation alone sends nothing", () => {
  expect(client.activate()).toBe(false);
  state(); expect(client.activate()).toBe(false);
  ready(); expect(client.activate()).toBe(true);
  expect(client.publisher.publish).not.toHaveBeenCalled();
  expect(client.publisher.name).toBe(LIFT_TOPIC);
});
test("rejects out-of-range lift positions and malformed state", () => {
  ready(); state([0.001]);
  expect(client.blocker()).toMatch(/Lift outside/);
  expect(client.activate()).toBe(false);
  state([NaN]); expect(client.positions).toBeNull();
  expect(liftPositions({ name: [...LIFT_JOINTS, LIFT_JOINTS[0]], position: [0, 0, 0] })).toBeNull();
  expect(client.publisher.publish).not.toHaveBeenCalled();
});
test("jogs from measured position with a finite horizon; repeated commands never accumulate", () => {
  ready(); client.activate(); client.start("up");
  for (let n = 1; n <= 4; n += 1) { time = n * 150; state(); client.tick(); }
  for (const [message] of client.publisher.publish.mock.calls) {
    expect(message.joint_names).toEqual(LIFT_JOINTS);
    expect(message.points[0].positions).toEqual([-0.197]);
    expect(message.points[0].time_from_start).toEqual({ sec: 0, nanosec: 300000000 });
  }
  state([-0.198]); client.stop();
  expect(client.publisher.publish.mock.calls.at(-1)[0].points[0].positions).toEqual([-0.198]);
});
test.each(["up", "down"])("%s stays in bounds and never reverses at a limit", (direction) => {
  for (const position of [-0.5, -0.499, -0.25, -0.001, 0]) {
    const result = liftTrajectory([position], direction).points[0].positions[0];
    expect(result).toBeGreaterThanOrEqual(LIFT_LIMITS[0][0]);
    expect(result).toBeLessThanOrEqual(LIFT_LIMITS[0][1]);
    expect(Math.abs(result - position)).toBeLessThanOrEqual(0.003001);
    expect((result - position) * (direction === "up" ? 1 : -1)).toBeGreaterThanOrEqual(0);
  }
});
test("does not send a queued stop or resume after disconnect", () => {
  ready(); client.activate(); client.start("up");
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
  client.service.callService.mock.calls.at(-1)[1]({ controller: [{ name: "lift_controller", state: "unconfigured" }] });
  expect(client.blocker()).toBe("Lift controller unconfigured");
  expect(client.start("down")).toBe(false);
});
test("competing commands deactivate teleop; own echoes do not", () => {
  ready(); client.activate(); client.start("up");
  const onCommand = client.commandSubscription.subscribe.mock.calls[0][0];
  onCommand(client.publisher.publish.mock.calls[0][0]); expect(client.active).toBe(true);
  onCommand({ lifter: { frame_id: "" } });
  expect(client.active).toBe(false); expect(client.blocker()).toMatch(/Another lift/);
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
});
test("dispose holds once, unsubscribes, and ignores late callbacks", () => {
  ready(); client.activate(); client.start("up"); client.dispose();
  expect(client.publisher.publish).toHaveBeenCalledTimes(2);
  expect(client.subscription.unsubscribe).toHaveBeenCalled();
  expect(client.commandSubscription.unsubscribe).toHaveBeenCalled();
  client.service.callService.mock.calls[0][1]({ controller: [{ name: "lift_controller", state: "active" }] });
  expect(client.activate()).toBe(false);
});
test("a delayed controller response cannot revive its expired request", () => {
  state(); client.tick(); const callback = client.service.callService.mock.calls[0][1];
  time = 1001; state(); callback({ controller: [{ name: "lift_controller", state: "active" }] });
  expect(client.activate()).toBe(false);
});
test("yields to leader input and requires explicit reactivation after it stops", () => {
  ready(); client.activate(); client.start("up");
  client.commandSubscription.subscribe.mock.calls[0][0]({ joint_names: ["lift_joint"], header: {} });
  expect(client.active).toBe(false);
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
  time = 1200; ready(); client.tick();
  expect(client.blocker()).toBe("");
  expect(client.active).toBe(false);
  expect(client.publisher.publish).toHaveBeenCalledTimes(1);
});
