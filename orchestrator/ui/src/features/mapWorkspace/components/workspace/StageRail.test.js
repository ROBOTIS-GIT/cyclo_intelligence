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

import { fireEvent, render, screen } from "@testing-library/react";
import StageRail from "./StageRail";


const props = { workspaceKind: "autonomy", workspaceStage: "collect", onSelectStage: jest.fn() };
beforeEach(() => props.onSelectStage.mockClear());

test("Autonomy offers exactly Record, Train, and Operate and navigates to Train", () => {
  render(<StageRail {...props} />);
  expect(screen.getAllByRole("tab").map(tab => tab.textContent)).toEqual(["Record", "Train", "Operate"]);
  fireEvent.click(screen.getByRole("tab", {name: "Train"}));
  expect(props.onSelectStage).toHaveBeenCalledWith("learn");
});

test.each(["learn", "annotate", "train", "reasoning"])("%s selects the single Train tab", stage => {
  render(<StageRail {...props} workspaceStage={stage} />);
  expect(screen.getByRole("tab", {name: "Train"})).toHaveAttribute("aria-selected", "true");
});

test("an active Record session keeps Train disabled until the episode guard clears", () => {
  const view = render(<StageRail {...props} disabledStages={["learn"]} />);
  fireEvent.click(screen.getByRole("tab", {name: "Train"}));
  expect(props.onSelectStage).not.toHaveBeenCalled();
  view.rerender(<StageRail {...props} disabledStages={[]} />);
  fireEvent.click(screen.getByRole("tab", {name: "Train"}));
  expect(props.onSelectStage).toHaveBeenCalledWith("learn");
});

test("unsaved editor work locks stage navigation until the editor releases it", () => {
  const view = render(<StageRail {...props} workspaceStage="learn" interactionLocked />);
  screen.getAllByRole("tab").forEach(tab => expect(tab).toBeDisabled());
  fireEvent.click(screen.getByRole("tab", {name: "Record"}));
  expect(props.onSelectStage).not.toHaveBeenCalled();
  view.rerender(<StageRail {...props} workspaceStage="learn" />);
  fireEvent.click(screen.getByRole("tab", {name: "Record"}));
  expect(props.onSelectStage).toHaveBeenCalledWith("collect");
});


test("the workspace supplies optional footer content without rail-owned training polling", () => {
  const view = render(<StageRail {...props} />);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  view.rerender(<StageRail {...props} footer={<div role="status">Training in background</div>} />);
  expect(screen.getByRole("status")).toHaveTextContent("Training in background");
});
