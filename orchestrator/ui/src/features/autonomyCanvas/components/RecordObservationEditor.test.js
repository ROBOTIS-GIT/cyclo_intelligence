// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import RecordObservationEditor from "./RecordObservationEditor";

function manager() {
  return {
    spots: [{ id: "table", label: "Table", pose: { x: 1, y: 2, yaw: 0 } }, { id: "tv", label: "TV" }],
    selectedId: "table", select: jest.fn(),
    route: { routeMode: true, setIncluded: jest.fn() },
    routeEditor: {
      plan: { observations: [{ spot_id: "table", departure_policy: "timed", dwell_seconds: 25, mission_ids: ["clean"] }], excluded_mission_ids: [] },
      missions: [{ mission_id: "clean", mission_name: "Clean table" }, { mission_id: "inspect", mission_name: "Inspect room" }],
      updatePoint: jest.fn(), updateSpot: jest.fn(), removeWaypoint: jest.fn().mockReturnValue(true), save: jest.fn(),
    },
  };
}

test("popover appears only for a selected existing waypoint during route editing", () => {
  const m = manager();
  m.route.routeMode = false;
  const { container, rerender } = render(<RecordObservationEditor manager={m} />);
  expect(container).toBeEmptyDOMElement();
  m.route.routeMode = true;
  m.selectedId = "missing";
  rerender(<RecordObservationEditor manager={m} />);
  expect(container).toBeEmptyDOMElement();
  m.selectedId = "table";
  rerender(<RecordObservationEditor manager={m} />);
  expect(screen.getByRole("region", { name: "Observation waypoint settings" })).toBeInTheDocument();
});

test("name and dwell edits are staged without a separate waypoint save", () => {
  const m = manager();
  render(<RecordObservationEditor manager={m} />);
  expect(screen.getByLabelText("Observation name")).toHaveValue("Table");
  const dwell = screen.getByLabelText("Dwell seconds");
  expect(dwell).toHaveValue(25);
  expect(dwell).toHaveAttribute("min", "2");
  expect(dwell).toHaveAttribute("max", "3600");
  fireEvent.change(screen.getByLabelText("Observation name"), { target: { value: "Dining table" } });
  fireEvent.change(dwell, { target: { value: "40.5" } });
  expect(m.routeEditor.updateSpot).toHaveBeenCalledWith("table", { label: "Dining table" });
  expect(m.routeEditor.updatePoint).toHaveBeenCalledWith("table", { dwell_seconds: 40.5 });
  expect(m.routeEditor.save).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: /save/i })).not.toBeInTheDocument();
});

test("mission checkboxes can add bindings or leave the waypoint unlinked", () => {
  const m = manager();
  render(<RecordObservationEditor manager={m} />);
  fireEvent.click(screen.getByLabelText("Inspect room"));
  expect(m.routeEditor.updatePoint).toHaveBeenLastCalledWith("table", { mission_ids: ["clean", "inspect"] });
  fireEvent.click(screen.getByLabelText("Clean table"));
  expect(m.routeEditor.updatePoint).toHaveBeenLastCalledWith("table", { mission_ids: [] });
});

test("excluded and unavailable bound missions remain visible and removable", () => {
  const m = manager();
  m.routeEditor.plan.excluded_mission_ids = ["clean", "inspect"];
  m.routeEditor.plan.observations[0].mission_ids = ["clean", "deleted"];
  render(<RecordObservationEditor manager={m} />);
  expect(screen.queryByLabelText("Inspect room (excluded)")).not.toBeInTheDocument();
  fireEvent.click(screen.getByLabelText("Clean table (excluded)"));
  expect(m.routeEditor.updatePoint).toHaveBeenLastCalledWith("table", { mission_ids: ["deleted"] });
  fireEvent.click(screen.getByLabelText("Unavailable mission (deleted)"));
  expect(m.routeEditor.updatePoint).toHaveBeenLastCalledWith("table", { mission_ids: ["clean"] });
});

test("route inclusion is explicit and does not delete a waypoint", () => {
  const m = manager();
  render(<RecordObservationEditor manager={m} />);
  fireEvent.click(screen.getByLabelText("Include in route"));
  expect(m.route.setIncluded).toHaveBeenCalledWith("table", false);
  expect(m.routeEditor.removeWaypoint).not.toHaveBeenCalled();
});

test("excluded waypoints retain session settings and require inclusion to edit them", () => {
  const m = manager();
  const retained = m.routeEditor.plan.observations[0];
  m.routeEditor.plan.observations = [];
  m.routeEditor.pointFor = jest.fn().mockReturnValue(retained);
  render(<RecordObservationEditor manager={m} />);
  expect(screen.getByLabelText("Observation name")).toBeEnabled();
  expect(screen.getByLabelText("Dwell seconds")).toHaveValue(25);
  expect(screen.getByLabelText("Dwell seconds")).toBeDisabled();
  expect(screen.getByLabelText("Clean table")).toBeChecked();
  expect(screen.getByLabelText("Clean table")).toBeDisabled();
  expect(screen.getByText("Include in route to edit departure and missions.")).toBeInTheDocument();
  fireEvent.click(screen.getByLabelText("Include in route"));
  expect(m.route.setIncluded).toHaveBeenCalledWith("table", true);
});

test.each(["disabled", "locked", "working", "blocked"])("%s locks changes while allowing the popover to close", lock => {
  const m = manager();
  if (lock === "locked") m.routeEditor.locked = true;
  else if (lock !== "disabled") m[lock] = true;
  const onClose = jest.fn();
  render(<RecordObservationEditor manager={m} disabled={lock === "disabled"} onClose={onClose} />);
  expect(screen.getByLabelText("Observation name")).toBeDisabled();
  expect(screen.getByLabelText("Dwell seconds")).toBeDisabled();
  expect(screen.getByLabelText("Include in route")).toBeDisabled();
  expect(screen.getByLabelText("Clean table")).toBeDisabled();
  expect(screen.getByRole("button", { name: "Delete waypoint" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Close waypoint settings" }));
  expect(m.select).toHaveBeenCalledWith("");
  expect(onClose).toHaveBeenCalledTimes(1);
});

test("Escape closes the selected waypoint popover", () => {
  const m = manager();
  render(<RecordObservationEditor manager={m} />);
  fireEvent.keyDown(screen.getByLabelText("Observation name"), { key: "Escape" });
  expect(m.select).toHaveBeenCalledWith("");
});

test("deleting requires confirmation and stages removal until route save", async () => {
  const m = manager();
  const confirm = jest.spyOn(window, "confirm").mockReturnValue(false);
  try {
    render(<RecordObservationEditor manager={m} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete waypoint" }));
    expect(m.routeEditor.removeWaypoint).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "Delete waypoint" }));
    await waitFor(() => expect(m.routeEditor.removeWaypoint).toHaveBeenCalledWith("table"));
    await waitFor(() => expect(m.select).toHaveBeenCalledWith(""));
    expect(confirm).toHaveBeenCalledWith('Delete waypoint "Table"? It will be removed when you save the route.');
    expect(m.routeEditor.save).not.toHaveBeenCalled();
  } finally { confirm.mockRestore(); }
});

test("new route points default to decision departure and hide the timer",()=>{
 const m=manager();delete m.routeEditor.plan.observations[0].departure_policy;
 render(<RecordObservationEditor manager={m}/>);
 expect(screen.getByLabelText("Observation departure")).toHaveValue("reasoning");
 expect(screen.queryByLabelText("Dwell seconds")).not.toBeInTheDocument();
 fireEvent.change(screen.getByLabelText("Observation departure"),{target:{value:"timed"}});
 expect(m.routeEditor.updatePoint).toHaveBeenCalledWith("table",{departure_policy:"timed"});
});
