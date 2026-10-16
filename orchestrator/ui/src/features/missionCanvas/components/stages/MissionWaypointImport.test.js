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
import MissionWaypointImport from "./MissionWaypointImport";
import { getNavigationMission } from "../../../../utils/navigationMissionsApi";
jest.mock("../../../../utils/navigationMissionsApi", () => ({ getNavigationMission: jest.fn() }));
afterEach(() => jest.clearAllMocks());

test("selects a waypoint from another saved mission and imports its pose", async () => {
  const waypoint = { id: "source", label: "Loading dock", pose: { x: 2, y: 3, yaw: 1 } };
  getNavigationMission.mockResolvedValue({ waypoints: [waypoint] });
  const onImport = jest.fn();
  render(<MissionWaypointImport mapName="warehouse" missionName="Mission1" missionNames={["Mission1", "Mission2"]} onImport={onImport} />);
  fireEvent.click(screen.getByText("From Mission"));
  expect(screen.queryByRole("option", { name: "Mission1" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Source mission"), { target: { value: "Mission2" } });
  await screen.findByRole("option", { name: "Loading dock" });
  fireEvent.change(screen.getByLabelText("Source waypoint"), { target: { value: "0" } });
  fireEvent.click(screen.getByText("Add waypoint"));
  expect(getNavigationMission).toHaveBeenCalledWith("warehouse", "Mission2");
  expect(onImport).toHaveBeenCalledWith(waypoint, "warehouse");
});

test("ignores a previous mission response after selection changes", async () => {
  let finish;
  getNavigationMission.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }))
    .mockResolvedValueOnce({ waypoints: [] });
  render(<MissionWaypointImport mapName="map" missionName="current" missionNames={["A", "B"]} onImport={jest.fn()} />);
  fireEvent.click(screen.getByText("From Mission"));
  fireEvent.change(screen.getByLabelText("Source mission"), { target: { value: "A" } });
  fireEvent.change(screen.getByLabelText("Source mission"), { target: { value: "B" } });
  await screen.findByText("This mission has no saved waypoints.");
  finish({ waypoints: [{ id: "stale", label: "Stale" }] });
  await waitFor(() => expect(screen.queryByRole("option", { name: /Stale/ })).not.toBeInTheDocument());
  expect(screen.getByText("Add waypoint")).toBeDisabled();
});
