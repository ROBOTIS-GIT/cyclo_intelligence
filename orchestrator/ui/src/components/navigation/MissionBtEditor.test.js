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

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import MissionBtEditor, {
  isValidBtConnection,
} from "./MissionBtEditor";

jest.mock("../../hooks/useBTNodeCatalog", () => ({
  useBTNodeCatalog: () => ({ catalog: [] }),
}));

jest.mock("react-redux", () => ({
  useDispatch: () => jest.fn(),
  useSelector: (selector) => selector({
    ros: { rosbridgeUrl: "ws://robot-host:7090" },
    tasks: { robotType: "ffw_sg2" },
  }),
}));

jest.mock("react-hot-toast", () => ({
  __esModule: true,
  default: { error: jest.fn(), success: jest.fn() },
}));

const treeXml = (waitName) => [
  '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
  `  <BehaviorTree ID="MainTree"><Wait name="${waitName}" duration="1.0"/></BehaviorTree>`,
  "</root>",
].join("\n");

const emptyTreeXml = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
  '  <BehaviorTree ID="MainTree"/>',
  "</root>",
  "",
].join("\n");

afterEach(() => {
  jest.useRealTimers();
});

test("rejects connections that would make the BT cyclic or multi-parent", () => {
  const nodes = [
    { id: "root", type: "btControl" },
    { id: "branch", type: "btControl" },
    { id: "leaf", type: "btAction" },
  ];
  const edges = [
    { source: "root", target: "branch" },
    { source: "branch", target: "leaf" },
  ];

  expect(isValidBtConnection({ source: "root", target: "leaf" }, nodes, edges)).toBe(false);
  expect(isValidBtConnection({ source: "leaf", target: "root" }, nodes, edges)).toBe(false);
  expect(isValidBtConnection({ source: "branch", target: "root" }, nodes, edges)).toBe(false);
  expect(isValidBtConnection({ source: "root", target: "root" }, nodes, edges)).toBe(false);
  expect(isValidBtConnection(
    { source: "root", target: "leaf" },
    nodes,
    [{ source: "root", target: "branch" }],
  )).toBe(true);
});

test("hydrates a loaded tree without emitting an initial empty graph", async () => {
  const onXmlChange = jest.fn();
  render(
    <MissionBtEditor
      title="A"
      filePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={onXmlChange}
    />,
  );
  await screen.findByText("StepA");
  expect(onXmlChange).not.toHaveBeenCalled();
});

test("hides the file banner while a node is selected without a loaded file", async () => {
  render(
    <MissionBtEditor
      title="init Local BT"
      filePath="locals/init.xml"
      defaultFilePath="locals/init.xml"
      xml={treeXml("StepA")}
      onXmlChange={jest.fn()}
    />,
  );

  const step = await screen.findByText("StepA");
  expect(screen.queryByText("init Local BT")).not.toBeInTheDocument();
  expect(screen.queryByText("init.xml")).not.toBeInTheDocument();
  expect(screen.queryByText("Run Task")).not.toBeInTheDocument();

  fireEvent.click(step);
});

test("disables clear for an empty waypoint BT", async () => {
  render(
    <MissionBtEditor
      filePath="locals/a.xml"
      defaultFilePath="locals/a.xml"
      xml={emptyTreeXml}
      onXmlChange={jest.fn()}
    />,
  );

  await screen.findByText("No waypoint task");
  expect(screen.getByRole("button", { name: "Clear current waypoint task" })).toBeDisabled();
});

test.each([
  ["loading", { loading: true }, "Loading waypoint task..."],
  ["parent file actions are busy", { fileActionsDisabled: true }, "StepA"],
])("disables clear while %s", async (_label, busyProps, expectedText) => {
  render(
    <MissionBtEditor
      filePath="locals/a.xml"
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={jest.fn()}
      {...busyProps}
    />,
  );

  await screen.findByText(expectedText);
  expect(screen.getByRole("button", { name: "Clear current waypoint task" })).toBeDisabled();
});

test("disarms and disables clear while a file action is pending", async () => {
  let resolveSave;
  const saveRequest = new Promise((resolve) => {
    resolveSave = resolve;
  });
  const onSaveXml = jest.fn(() => saveRequest);
  render(
    <MissionBtEditor
      filePath="locals/a.xml"
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={jest.fn()}
      onSaveXml={onSaveXml}
    />,
  );
  await screen.findByText("StepA");

  fireEvent.click(screen.getByRole("button", { name: "Clear current waypoint task" }));
  expect(screen.getByRole("button", { name: "Confirm clear current waypoint task" }))
    .toBeEnabled();

  fireEvent.click(screen.getByRole("button", { name: "Save Task" }));

  await waitFor(() => {
    expect(screen.getByRole("button", { name: "Clear current waypoint task" })).toBeDisabled();
  });
  expect(onSaveXml).toHaveBeenCalledTimes(1);

  await act(async () => {
    resolveSave({ path: "locals/a.xml", exists: true });
    await saveRequest;
  });
  await waitFor(() => {
    expect(screen.getByRole("button", { name: "Clear current waypoint task" })).toBeEnabled();
  });
});

test("clears waypoint BT contents after confirmation without changing its file identity", async () => {
  const onXmlChange = jest.fn();
  const onFilePathChange = jest.fn();
  const onSetDefaultXml = jest.fn();
  render(
    <MissionBtEditor
      filePath="locals/a.xml"
      fileOptions={["locals/a.xml", "locals/b.xml"]}
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={onXmlChange}
      onFilePathChange={onFilePathChange}
      onSetDefaultXml={onSetDefaultXml}
      onSaveXml={jest.fn()}
    />,
  );
  fireEvent.click(await screen.findByText("StepA"));
  onXmlChange.mockClear();

  fireEvent.click(screen.getByRole("button", { name: "Clear current waypoint task" }));

  expect(screen.getByRole("button", { name: "Confirm clear current waypoint task" }))
    .toHaveAttribute("title", "Click again to clear the current waypoint task");
  expect(screen.getByText("StepA")).toBeInTheDocument();
  expect(onXmlChange).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Confirm clear current waypoint task" }));

  await screen.findByText("No waypoint task");
  await waitFor(() => {
    expect(onXmlChange).toHaveBeenCalledTimes(1);
  });
  expect(onXmlChange).toHaveBeenCalledWith("locals/a.xml", emptyTreeXml);
  expect(screen.getByRole("button", { name: "Save Task" }))
    .toHaveAttribute("title", "Save Task to locals/a.xml");
  expect(screen.getByRole("button", { name: "Use for Run" }))
    .toHaveAttribute("title", "This task is already used when running the mission");
  expect(onFilePathChange).not.toHaveBeenCalled();
  expect(onSetDefaultXml).not.toHaveBeenCalled();
});

test("restores and reapplies a cleared waypoint BT through undo and redo", async () => {
  const onXmlChange = jest.fn();
  render(
    <MissionBtEditor
      filePath="locals/a.xml"
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={onXmlChange}
    />,
  );
  await screen.findByText("StepA");
  onXmlChange.mockClear();

  fireEvent.click(screen.getByRole("button", { name: "Clear current waypoint task" }));
  fireEvent.click(screen.getByRole("button", { name: "Confirm clear current waypoint task" }));
  await screen.findByText("No waypoint task");
  await waitFor(() => expect(onXmlChange).toHaveBeenCalledTimes(1));
  expect(onXmlChange).toHaveBeenLastCalledWith("locals/a.xml", emptyTreeXml);

  const undo = screen.getByTitle("Undo");
  expect(undo).toBeEnabled();
  fireEvent.click(undo);

  await screen.findByText("StepA");
  await waitFor(() => expect(onXmlChange).toHaveBeenCalledTimes(2));
  expect(onXmlChange.mock.calls[1][0]).toBe("locals/a.xml");
  expect(onXmlChange.mock.calls[1][1]).toContain('<Wait name="StepA" duration="1.0"/>');

  const redo = screen.getByTitle("Redo");
  expect(redo).toBeEnabled();
  fireEvent.click(redo);

  await screen.findByText("No waypoint task");
  await waitFor(() => expect(onXmlChange).toHaveBeenCalledTimes(3));
  expect(onXmlChange).toHaveBeenLastCalledWith("locals/a.xml", emptyTreeXml);
  expect(screen.getByRole("button", { name: "Save Task" }))
    .toHaveAttribute("title", "Save Task to locals/a.xml");
});

test("expires clear confirmation without changing the waypoint BT", async () => {
  const onXmlChange = jest.fn();
  render(
    <MissionBtEditor
      filePath="locals/a.xml"
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={onXmlChange}
    />,
  );
  await screen.findByText("StepA");
  onXmlChange.mockClear();
  jest.useFakeTimers();

  fireEvent.click(screen.getByRole("button", { name: "Clear current waypoint task" }));
  expect(screen.getByRole("button", { name: "Confirm clear current waypoint task" }))
    .toBeEnabled();

  act(() => {
    jest.advanceTimersByTime(4000);
  });

  expect(screen.getByRole("button", { name: "Clear current waypoint task" })).toBeEnabled();
  expect(screen.getByText("StepA")).toBeInTheDocument();
  expect(onXmlChange).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Clear current waypoint task" }));
  expect(screen.getByRole("button", { name: "Confirm clear current waypoint task" }))
    .toBeEnabled();
  expect(screen.getByText("StepA")).toBeInTheDocument();
  expect(onXmlChange).not.toHaveBeenCalled();
});

test("does not emit the previous graph while hydrating a new waypoint path", async () => {
  const onXmlChange = jest.fn();
  const { rerender } = render(
    <MissionBtEditor
      title="A"
      filePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={onXmlChange}
    />,
  );
  await screen.findByText("StepA");

  // Switch to another waypoint; the parent supplies that waypoint's XML.
  onXmlChange.mockClear();
  rerender(
    <MissionBtEditor
      title="B"
      filePath="locals/b.xml"
      xml={treeXml("StepB")}
      onXmlChange={onXmlChange}
    />,
  );
  await screen.findByText("StepB");
  expect(onXmlChange).not.toHaveBeenCalled();
});

test("captures a new parameter edit immediately after undo", async () => {
  render(
    <MissionBtEditor
      title="A"
      filePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={jest.fn()}
    />,
  );
  fireEvent.click(await screen.findByText("StepA"));
  const undo = screen.getByTitle("Undo");
  fireEvent.change(screen.getByDisplayValue("1.0"), { target: { value: "2.0" } });
  await waitFor(() => expect(undo).toBeEnabled());
  fireEvent.click(undo);
  await waitFor(() => expect(undo).toBeDisabled());

  fireEvent.click(screen.getByText("StepA"));
  fireEvent.change(screen.getByDisplayValue("1.0"), { target: { value: "3.0" } });
  await waitFor(() => expect(undo).toBeEnabled());
});

test("loads a selected XML without changing the waypoint default", async () => {
  const onLoadXml = jest.fn().mockResolvedValue({
    path: "locals/b.xml",
    content: treeXml("LoadedStep"),
    exists: true,
  });
  const onFilePathChange = jest.fn();
  const onSetDefaultXml = jest.fn();
  render(
    <MissionBtEditor
      title="A"
      filePath="locals/a.xml"
      fileOptions={["locals/a.xml", "locals/b.xml"]}
      defaultFilePath="locals/a.xml"
      xml={treeXml("OriginalStep")}
      onXmlChange={jest.fn()}
      onLoadXml={onLoadXml}
      onFilePathChange={onFilePathChange}
      onSetDefaultXml={onSetDefaultXml}
    />,
  );
  await screen.findByText("OriginalStep");

  fireEvent.click(screen.getByRole("button", { name: "Open Task" }));
  expect(screen.getByRole("dialog", { name: "Waypoint Task files" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("radio", { name: /b\.xml/ }));
  fireEvent.click(screen.getByRole("button", { name: "Open Selected" }));
  await waitFor(() => expect(onLoadXml).toHaveBeenCalledWith("locals/b.xml"));
  expect(onFilePathChange).toHaveBeenCalledWith("locals/b.xml");
  expect(onSetDefaultXml).not.toHaveBeenCalled();
});

test("saves the latest graph to the current mission-local path", async () => {
  const onSaveXml = jest.fn().mockResolvedValue({
    path: "locals/a.xml",
    exists: true,
  });
  render(
    <MissionBtEditor
      title="A"
      filePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={jest.fn()}
      onSaveXml={onSaveXml}
    />,
  );
  fireEvent.click(await screen.findByText("StepA"));
  fireEvent.change(screen.getByDisplayValue("1.0"), { target: { value: "3.5" } });
  fireEvent.click(screen.getByRole("button", { name: "Save Task" }));

  await waitFor(() => expect(onSaveXml).toHaveBeenCalledWith(
    "locals/a.xml",
    expect.stringContaining('duration="3.5"'),
  ));
});

test("saves the latest graph as another XML in the same waypoint", async () => {
  const onSaveXmlAs = jest.fn().mockResolvedValue({
    path: "locals/spot_a/alternate.xml",
    exists: true,
  });
  const onFilePathChange = jest.fn();
  render(
    <MissionBtEditor
      title="A"
      filePath="locals/a.xml"
      fileOptions={["locals/a.xml"]}
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepA")}
      onXmlChange={jest.fn()}
      onSaveXmlAs={onSaveXmlAs}
      onFilePathChange={onFilePathChange}
    />,
  );
  fireEvent.click(await screen.findByText("StepA"));
  fireEvent.change(screen.getByDisplayValue("1.0"), { target: { value: "4.0" } });
  fireEvent.click(screen.getByRole("button", { name: "Save Task As" }));
  fireEvent.change(screen.getByRole("textbox", { name: "New task file name" }), {
    target: { value: "alternate" },
  });
  fireEvent.click(within(
    screen.getByRole("dialog", { name: "Save Waypoint Task As" }),
  ).getByRole("button", { name: "Save Task As" }));

  await waitFor(() => expect(onSaveXmlAs).toHaveBeenCalledWith(
    "locals/a.xml",
    "alternate",
    expect.stringContaining('duration="4.0"'),
  ));
  expect(onFilePathChange).toHaveBeenCalledWith("locals/spot_a/alternate.xml");
});

test("changes the runtime default only through Use for Run", async () => {
  const onSetDefaultXml = jest.fn().mockResolvedValue(undefined);
  render(
    <MissionBtEditor
      title="A"
      filePath="locals/b.xml"
      fileOptions={["locals/a.xml", "locals/b.xml"]}
      defaultFilePath="locals/a.xml"
      xml={treeXml("StepB")}
      onXmlChange={jest.fn()}
      onSetDefaultXml={onSetDefaultXml}
    />,
  );
  await screen.findByText("StepB");
  fireEvent.click(screen.getByRole("button", { name: "Use for Run" }));
  await waitFor(() => expect(onSetDefaultXml).toHaveBeenCalledWith("locals/b.xml"));
});
