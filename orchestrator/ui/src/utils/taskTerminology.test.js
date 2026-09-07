import { formatTaskDisplayMessage } from "./taskTerminology";

test("uses Action Canvas terminology for backend messages", () => {
  expect(formatTaskDisplayMessage("BT node rejected the BehaviorTree"))
    .toBe("Task Engine rejected the Task");
  expect(formatTaskDisplayMessage("Behavior trees failed in the BT runtime"))
    .toBe("Tasks failed in the Task Engine");
  expect(formatTaskDisplayMessage("Unsupported BT node class: Wait"))
    .toBe("Unsupported step class: Wait");
  expect(formatTaskDisplayMessage("Unknown node type: Wait"))
    .toBe("Unknown step type: Wait");
  expect(formatTaskDisplayMessage("Failed to save tree: Tree already exists"))
    .toBe("Failed to save task: Task already exists");
  expect(formatTaskDisplayMessage("Trees directory not found; No tree loaded"))
    .toBe("Tasks directory not found; No task loaded");
});

test("uses Waypoint Task terminology for mission messages", () => {
  expect(formatTaskDisplayMessage("Local BT rejected the behavior tree", "Waypoint Task"))
    .toBe("Waypoint Task rejected the Waypoint Task");
  expect(formatTaskDisplayMessage("Failed to load tree", "Waypoint Task"))
    .toBe("Failed to load waypoint task");
});

test("preserves real paths, filenames, process paths, and XML tags", () => {
  const technical = "/bt/status tree.xml /services/bt_node <BehaviorTree> BT.CPP loaded tree.xml";
  expect(formatTaskDisplayMessage(technical, "Waypoint Task")).toBe(technical);
});
