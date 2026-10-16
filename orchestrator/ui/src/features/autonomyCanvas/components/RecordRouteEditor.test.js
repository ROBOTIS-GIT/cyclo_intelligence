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
import RecordRouteEditor from "./RecordRouteEditor";

function manager() {
  return {
    route: { routeMode: true, orderMode: false, toggleOrderMode: jest.fn() },
    routeEditor: {
      plan: { observations: [{ spot_id: "table" }, { spot_id: "tv" }] },
      dirty: true, save: jest.fn().mockResolvedValue(true),
    },
  };
}

const button = name => screen.getByRole("button", { name });

test("route editor contains only Save and route-order icons", () => {
  render(<RecordRouteEditor manager={manager()} />);
  const buttons = screen.getAllByRole("button");
  expect(buttons).toHaveLength(2);
  expect(buttons.map(item => item.getAttribute("aria-label"))).toEqual([
    "Save observation route", "Set observation route order",
  ]);
  buttons.forEach(item => expect(item.textContent).toBe(""));
  expect(screen.queryByRole("list", { name: "Route order" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Discard|loop/ })).not.toBeInTheDocument();
});

test("route-order button enters and leaves map ordering mode", () => {
  const m = manager();
  const { rerender } = render(<RecordRouteEditor manager={m} />);
  const order = button("Set observation route order");
  expect(order).not.toHaveClass("is-active");
  fireEvent.click(order);
  expect(m.route.toggleOrderMode).toHaveBeenCalledTimes(1);
  m.route.orderMode = true;
  rerender(<RecordRouteEditor manager={m} />);
  expect(order).toHaveAttribute("aria-pressed", "true");
  expect(order).toHaveClass("is-active");
  expect(order).toHaveAttribute("title", expect.stringContaining("first waypoint again"));
  fireEvent.click(order);
  expect(m.route.toggleOrderMode).toHaveBeenCalledTimes(2);
});

test("Save commits staged edits", async () => {
  const m = manager();
  render(<RecordRouteEditor manager={m} />);
  fireEvent.click(button("Save observation route"));
  await waitFor(() => expect(m.routeEditor.save).toHaveBeenCalledTimes(1));
});

test("unchanged routes can finish editing while empty routes cannot be saved", async () => {
  const m = manager();
  m.routeEditor.dirty = false;
  const { rerender } = render(<RecordRouteEditor manager={m} />);
  expect(button("Save observation route")).toBeEnabled();
  expect(button("Set observation route order")).toBeEnabled();
  fireEvent.click(button("Save observation route"));
  await waitFor(() => expect(m.routeEditor.save).toHaveBeenCalledTimes(1));
  m.routeEditor.dirty = true;
  m.routeEditor.plan.observations = [];
  rerender(<RecordRouteEditor manager={m} />);
  expect(button("Save observation route")).toBeDisabled();
  expect(button("Set observation route order")).toBeEnabled();
  m.routeEditor.plan.observations = [{ spot_id: "table" }];
  rerender(<RecordRouteEditor manager={m} />);
  expect(button("Save observation route")).toBeEnabled();
});

test.each(["disabled", "locked", "working", "blocked"])("%s prevents route edits", lock => {
  const m = manager();
  if (lock === "locked") m.routeEditor.locked = true;
  else if (lock !== "disabled") m[lock] = true;
  render(<RecordRouteEditor manager={m} disabled={lock === "disabled"} />);
  screen.getAllByRole("button").forEach(item => expect(item).toBeDisabled());
  fireEvent.click(button("Set observation route order"));
  fireEvent.click(button("Save observation route"));
  expect(m.route.toggleOrderMode).not.toHaveBeenCalled();
  expect(m.routeEditor.save).not.toHaveBeenCalled();
});

test("bar only appears with a plan in route edit mode", () => {
  const m = manager();
  m.route.routeMode = false;
  const { container, rerender } = render(<RecordRouteEditor manager={m} />);
  expect(container).toBeEmptyDOMElement();
  m.route.routeMode = true;
  m.routeEditor.plan = null;
  rerender(<RecordRouteEditor manager={m} />);
  expect(container).toBeEmptyDOMElement();
});
