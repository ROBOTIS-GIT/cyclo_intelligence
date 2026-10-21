import { act, fireEvent, render, screen } from "@testing-library/react";
import LiftTeleopPanel from "./LiftTeleopPanel";
import { LiftTeleopClient } from "../../lib/liftTeleop";

jest.mock("react-redux", () => ({ useSelector: (fn) => fn({ ros: { rosbridgeUrl: "ws://test" }, tasks: { robotType: "ffw_sg2" } }) }));
jest.mock("../../../../utils/rosConnectionManager", () => ({
  url: "ws://test", getCurrentConnection: () => mockRos,
}));
const mockRos = { isConnected: true };
jest.mock("../../lib/liftTeleop", () => ({ LiftTeleopClient: jest.fn() }));
let client;
beforeEach(() => {
  jest.useFakeTimers();
  client = {
    ros: mockRos, active: false, direction: null, reason: "", positions: [-0.2],
    tick: jest.fn(), dispose: jest.fn(),
    snapshot() { return { active: this.active, direction: this.direction, reason: this.reason, positions: this.positions }; },
    activate: jest.fn(function () { this.active = true; }),
    deactivate: jest.fn(function () { this.active = false; this.direction = null; }),
    start: jest.fn(function (dir) { this.direction = dir; }),
    stop: jest.fn(function () { this.direction = null; }),
  };
  LiftTeleopClient.mockImplementation(() => client);
});
afterEach(() => jest.useRealTimers());

test("explicit activation, focus-scoped keyboard jog, release and blur stop", () => {
  render(<LiftTeleopPanel />);
  expect(screen.getByRole("button", { name: "Lift up" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Activate lift" }));
  const group = screen.getByRole("group", { name: "Lift Teleop" });
  fireEvent.keyDown(group, { key: "u" }); expect(client.start).toHaveBeenLastCalledWith("up");
  fireEvent.keyUp(group, { key: "u" }); expect(client.stop).toHaveBeenCalled();
  fireEvent.keyDown(group, { key: "o" });
  fireEvent(window, new Event("blur")); expect(client.deactivate).toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Lift up" })).toBeDisabled();
});
test("shows the blocker and never activates an unsafe lift", () => {
  client.reason = "Lift outside safe range — recover the joint before teleop";
  render(<LiftTeleopPanel />);
  expect(screen.getByRole("status")).toHaveTextContent("Lift outside safe range");
  fireEvent.click(screen.getByRole("button", { name: "Activate lift" }));
  expect(client.activate).not.toHaveBeenCalled();
});
test("pointer release and cancellation stop; unmount disposes the transport", () => {
  const { unmount } = render(<LiftTeleopPanel />);
  fireEvent.click(screen.getByRole("button", { name: "Activate lift" }));
  const button = screen.getByRole("button", { name: "Lift down" });
  fireEvent.keyDown(button, { key: "Enter" }); expect(client.start).toHaveBeenCalledWith("down");
  fireEvent.pointerUp(button); expect(client.stop).toHaveBeenCalled();
  fireEvent.pointerCancel(button); expect(client.stop).toHaveBeenCalledTimes(2);
  unmount(); expect(client.dispose).toHaveBeenCalled();
});
test("operation lock and hidden tab deactivate", () => {
  const { rerender } = render(<LiftTeleopPanel />);
  fireEvent.click(screen.getByRole("button", { name: "Activate lift" }));
  fireEvent(document, new Event("visibilitychange"));
  expect(client.deactivate).toHaveBeenCalled();
  rerender(<LiftTeleopPanel disabled />);
  act(() => jest.advanceTimersByTime(100));
  expect(screen.getByRole("button", { name: "Activate lift" })).toBeDisabled();
});
