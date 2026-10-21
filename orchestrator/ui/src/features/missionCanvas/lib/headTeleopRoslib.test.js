import ROSLIB from "roslib";
import { HeadTeleopClient, HEAD_COMMAND_TOPIC, HEAD_TOPIC } from "./headTeleop";
import { LiftTeleopClient, LIFT_TOPIC } from "./liftTeleop";

test("real roslib 1.4.1 keeps the Ros receiver with automatic reconnect disabled", () => {
  const ros = new ROSLIB.Ros();
  ros.isConnected = true;
  ros.sendEncodedMessage = jest.fn();
  const client = new HeadTeleopClient(ros, () => 0);
  ros.emit("/joint_states", { name: ["head_joint1", "head_joint2"], position: [0.8, 0] });
  ros.emit(HEAD_TOPIC, { header: { frame_id: "leader_joystick_controller" } });
  client.controllerState = "active";
  client.controllerAt = 0;
  expect(client.activate()).toBe(true);
  expect(client.start("up")).toBe(true);
  const sent = ros.sendEncodedMessage.mock.calls.map(([data]) => JSON.parse(data));
  expect(sent).toEqual(expect.arrayContaining([
    expect.objectContaining({ op: "subscribe", topic: "/joint_states" }),
    expect.objectContaining({ op: "subscribe", topic: HEAD_TOPIC }),
    expect.objectContaining({ op: "advertise", topic: HEAD_COMMAND_TOPIC }),
    expect.objectContaining({ op: "publish", topic: HEAD_COMMAND_TOPIC }),
  ]));
  ros.isConnected = false;
  client.stop();
  expect(ros.listeners("connection")).toHaveLength(0);
  // Clean up while connected so the library does not queue unsubscriptions.
  ros.isConnected = true;
  client.dispose();
});

test("expired controller requests and deactivated publishers are released", () => {
  let now = 0;
  const ros = new ROSLIB.Ros(); ros.isConnected = true; ros.sendEncodedMessage = jest.fn();
  const client = new HeadTeleopClient(ros, () => now);
  client.tick();
  const firstId = client.pendingServiceId;
  expect(ros.listeners(firstId)).toHaveLength(1);
  now = 1100; client.tick();
  expect(ros.listeners(firstId)).toHaveLength(0);
  const pendingId = client.pendingServiceId;
  ros.emit("/joint_states", { name: ["head_joint1", "head_joint2"], position: [0.8, 0] });
  ros.emit(HEAD_TOPIC, { header: { frame_id: "leader_joystick_controller" } });
  client.controllerState = "active"; client.controllerAt = now;
  client.activate(); client.start("up"); client.deactivate();
  expect(client.publisher.isAdvertised).toBe(false);
  client.dispose(); expect(ros.listeners(pendingId)).toHaveLength(0);
});

test("real roslib lift transport releases on competing commands without deferred motion", () => {
  const ros = new ROSLIB.Ros(); ros.isConnected = true; ros.sendEncodedMessage = jest.fn();
  const client = new LiftTeleopClient(ros, () => 0);
  ros.emit("/joint_states", { name: ["lift_joint"], position: [-0.2] });
  client.controllerState = "active"; client.controllerAt = 0;
  expect(client.activate()).toBe(true); expect(client.start("up")).toBe(true);
  ros.emit(LIFT_TOPIC, { header: {}, joint_names: ["lift_joint"] });
  expect(client.active).toBe(false); expect(client.publisher.isAdvertised).toBe(false);
  const publishes = ros.sendEncodedMessage.mock.calls.map(([data]) => JSON.parse(data)).filter(m => m.op === "publish");
  expect(publishes).toHaveLength(1);
  expect(publishes[0].msg.points[0].positions).toEqual([-0.197]);
  ros.isConnected = false; client.tick();
  expect(ros.listeners("connection")).toHaveLength(0);
  ros.isConnected = true; client.dispose();
});
