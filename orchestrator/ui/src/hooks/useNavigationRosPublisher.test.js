import { act, renderHook } from "@testing-library/react";
import ROSLIB from "roslib";
import rosConnectionManager from "../utils/rosConnectionManager";
import { useNavigationRosPublisher } from "./useNavigationRosTopic";

jest.mock("react-redux", () => ({
  useSelector: (select) => select({ ros: { rosbridgeUrl: "ws://test" } }),
}));
jest.mock("../utils/rosConnectionManager", () => ({
  __esModule: true,
  default: { url: "ws://test", getCurrentConnection: jest.fn() },
}));

test("mobile commands share one publisher and are never queued after disconnect", async () => {
  jest.useFakeTimers();
  const ros = new ROSLIB.Ros();
  ros.isConnected = true;
  ros.sendEncodedMessage = jest.fn();
  rosConnectionManager.getCurrentConnection.mockReturnValue(ros);
  const { result, unmount } = renderHook(() => useNavigationRosPublisher());
  const topic = "/leader/mission_canvas_mobile/cmd_vel";
  const command = { linear: { x: 0.3, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } };

  await act(async () => {
    await result.current(topic, "geometry_msgs/msg/Twist", command);
    await result.current(topic, "geometry_msgs/msg/Twist", command);
  });
  let sent = ros.sendEncodedMessage.mock.calls.map(([encoded]) => JSON.parse(encoded));
  expect(sent.filter(({ op }) => op === "advertise")).toHaveLength(1);
  expect(sent.filter(({ op }) => op === "publish")).toHaveLength(2);

  ros.isConnected = false;
  await expect(result.current(topic, "geometry_msgs/msg/Twist", command))
    .rejects.toThrow("ROS connection is not available");
  ros.isConnected = true;
  act(() => jest.advanceTimersByTime(1000));
  sent = ros.sendEncodedMessage.mock.calls.map(([encoded]) => JSON.parse(encoded));
  expect(sent.filter(({ op }) => op === "publish")).toHaveLength(2);
  expect(sent.filter(({ op }) => op === "unadvertise")).toHaveLength(1);
  unmount();
  jest.useRealTimers();
});
