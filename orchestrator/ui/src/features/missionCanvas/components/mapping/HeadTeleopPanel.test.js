import { act, fireEvent, render, screen } from "@testing-library/react";
import HeadTeleopPanel from "./HeadTeleopPanel";
import { HeadTeleopClient } from "../../lib/headTeleop";

jest.mock("react-redux", () => ({ useSelector: (fn) => fn({ ros: { rosbridgeUrl: "ws://test" }, tasks: { robotType: "ffw_sg2" } }) }));
jest.mock("../../../../utils/rosConnectionManager", () => ({
  url: "ws://test", getCurrentConnection: () => mockRos,
}));
const mockRos = { isConnected: true };
jest.mock("../../lib/headTeleop", () => ({ HeadTeleopClient: jest.fn() }));
let client;
beforeEach(() => {
  jest.useFakeTimers();
  client = {
    ros: mockRos, active: false, direction: null, reason: "", positions: [0.8, 0],
    tick: jest.fn(), dispose: jest.fn(),
    snapshot() { return { active: this.active, direction: this.direction,
      reason: this.reason, positions: this.positions }; },
    activate: jest.fn(function () { this.active = true; }),
    deactivate: jest.fn(function () { this.active = false; this.direction = null; }),
    start: jest.fn(function (dir) { this.direction = dir; }),
    stop: jest.fn(function () { this.direction = null; }),
  };
  HeadTeleopClient.mockImplementation(() => client);
});
afterEach(() => jest.useRealTimers());

test("explicit activation, focus-scoped keyboard jog, release and blur stop", () => {
  render(<HeadTeleopPanel />);
  expect(screen.getByRole("button", { name: "Head up" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Activate head" }));
  const group = screen.getByRole("group", { name: "Head Teleop" });
  fireEvent.keyDown(group, { key: "i" }); expect(client.start).toHaveBeenLastCalledWith("up");
  fireEvent.keyUp(group, { key: "i" }); expect(client.stop).toHaveBeenCalled();
  fireEvent.keyDown(group, { key: "j" });
  fireEvent(window, new Event("blur")); expect(client.deactivate).toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Head up" })).toBeDisabled();
});
test("shows the blocker and never activates an unsafe head", () => {
  client.reason = "Pitch outside safe range — recover the joint before teleop";
  render(<HeadTeleopPanel />);
  expect(screen.getByRole("status")).toHaveTextContent("Pitch outside safe range");
  fireEvent.click(screen.getByRole("button", { name: "Activate head" }));
  expect(client.activate).not.toHaveBeenCalled();
});
test("pointer release and cancellation stop; unmount disposes the transport", () => {
  const { unmount } = render(<HeadTeleopPanel />);
  fireEvent.click(screen.getByRole("button", { name: "Activate head" }));
  const button = screen.getByRole("button", { name: "Head right" });
  fireEvent.keyDown(button, { key: "Enter" }); expect(client.start).toHaveBeenCalledWith("right");
  fireEvent.pointerUp(button); expect(client.stop).toHaveBeenCalled();
  fireEvent.pointerCancel(button); expect(client.stop).toHaveBeenCalledTimes(2);
  unmount(); expect(client.dispose).toHaveBeenCalled();
});
test("operation lock and hidden tab deactivate", () => {
  const { rerender } = render(<HeadTeleopPanel />);
  fireEvent.click(screen.getByRole("button", { name: "Activate head" }));
  fireEvent(document, new Event("visibilitychange"));
  expect(client.deactivate).toHaveBeenCalled();
  rerender(<HeadTeleopPanel disabled />);
  act(() => jest.advanceTimersByTime(100));
  expect(screen.getByRole("button", { name: "Activate head" })).toBeDisabled();
});
