import { act, fireEvent, render, screen } from "@testing-library/react";
import MappingTeleopPanel from "../../../mapWorkspace/components/MappingTeleopPanel";

let mockReady = false;
jest.mock("../../../../hooks/useNavigationRosTopic", () => ({
  useNavigationRosTopic: () => ({
    topicData: mockReady ? { available: true, data: { data: true } } : null,
  }),
}));

afterEach(() => { jest.useRealTimers(); mockReady = false; });

test("requires the leader relay, repeats a held command, and stops on release", () => {
  jest.useFakeTimers();
  const publish = jest.fn().mockResolvedValue();
  const message = jest.fn();
  const { rerender } = render(<MappingTeleopPanel onPublish={publish} onMessage={message} />);
  expect(screen.getByRole("button", { name: "Activate" })).toBeDisabled();
  expect(screen.getByText("Waiting for leader controller")).toBeInTheDocument();

  mockReady = true;
  rerender(<MappingTeleopPanel onPublish={publish} onMessage={message} />);
  fireEvent.click(screen.getByRole("button", { name: "Activate" }));
  fireEvent.keyDown(window, { key: "w" });
  act(() => jest.advanceTimersByTime(450));
  expect(publish).toHaveBeenCalledTimes(3);
  expect(publish).toHaveBeenNthCalledWith(1, { linearX: 0.4, angularZ: 0 });

  fireEvent.keyUp(window, { key: "w" });
  expect(publish).toHaveBeenLastCalledWith({ linearX: 0, angularZ: 0 });
  const count = publish.mock.calls.length;
  act(() => jest.advanceTimersByTime(450));
  expect(publish).toHaveBeenCalledTimes(count);

  fireEvent.keyDown(window, { key: "w" });
  fireEvent(window, new Event("blur"));
  expect(publish).toHaveBeenLastCalledWith({ linearX: 0, angularZ: 0 });

  mockReady = false;
  rerender(<MappingTeleopPanel onPublish={publish} onMessage={message} />);
  expect(screen.getByRole("button", { name: "Activate" })).toBeDisabled();
});
