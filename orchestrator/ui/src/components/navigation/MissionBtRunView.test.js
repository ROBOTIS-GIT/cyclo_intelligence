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

import { useLayoutEffect, useRef } from "react";
import { act, render, screen } from "@testing-library/react";
import MissionBtRunView from "./MissionBtRunView";

const reactFlowNodeSnapshots = [];
let reactFlowOnNodesChange;

jest.mock("@xyflow/react", () => ({
  ReactFlow: ({ nodes, onNodesChange, children }) => {
    reactFlowNodeSnapshots.push(nodes);
    reactFlowOnNodesChange = onNodesChange;
    return (
      <div data-testid="react-flow">
        {nodes.map((node) => <span key={node.id}>{node.data.label}</span>)}
        {children}
      </div>
    );
  },
  Controls: () => null,
  Background: () => null,
  useNodesState: (initialNodes) => {
    const React = require("react");
    const [nodes, setNodes] = React.useState(initialNodes);
    const onNodesChange = React.useCallback((changes) => {
      setNodes((current) => current.map((node) => {
        const dimensions = changes.find((change) => (
          change.id === node.id && change.type === "dimensions"
        ));
        return dimensions
          ? { ...node, measured: { ...dimensions.dimensions } }
          : node;
      }));
    }, []);
    return [nodes, setNodes, onNodesChange];
  },
}));

const EMPTY_TREE = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
  '  <BehaviorTree ID="MainTree"/>',
  "</root>",
].join("\n");

const POPULATED_TREE = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
  '  <BehaviorTree ID="MainTree">',
  '    <Sequence name="Sequence A">',
  '      <Wait name="Wait A" msec="1000"/>',
  '      <Wait name="Wait B" msec="1000"/>',
  '    </Sequence>',
  '  </BehaviorTree>',
  '</root>',
].join("\n");

function FirstCommitProbe({ children, onCommit }) {
  const ref = useRef(null);
  useLayoutEffect(() => {
    onCommit(ref.current?.textContent || "");
  }, [onCommit]);
  return <div ref={ref}>{children}</div>;
}

// The ReactFlow-backed graph needs a real layout engine, so these assertions
// cover the plain-DOM states the viewer falls back to.
test("shows a navigate-only message when the waypoint has no waypoint task", () => {
  render(<MissionBtRunView xml={EMPTY_TREE} activeNodeNames={[]} />);
  expect(screen.getByText("Navigate only")).toBeInTheDocument();
  expect(screen.getByText("This waypoint has no waypoint task.")).toBeInTheDocument();
});

test("shows a loading state while the tree is being fetched", () => {
  render(<MissionBtRunView xml="" activeNodeNames={[]} loading />);
  expect(screen.getByText("Loading waypoint task...")).toBeInTheDocument();
});

test("renders waypoint nodes in the first committed frame", () => {
  const commits = [];
  const recordCommit = (text) => commits.push(text);

  render(
    <FirstCommitProbe onCommit={recordCommit}>
      <MissionBtRunView xml={POPULATED_TREE} activeNodeNames={[]} />
    </FirstCommitProbe>,
  );

  expect(commits[0]).toContain("Sequence A");
  expect(commits[0]).toContain("Wait A");
  expect(commits[0]).toContain("Wait B");
  expect(commits[0]).not.toContain("Navigate only");
});

test("preserves measured node dimensions when the active node changes", () => {
  reactFlowNodeSnapshots.length = 0;
  reactFlowOnNodesChange = undefined;
  const { rerender } = render(
    <MissionBtRunView xml={POPULATED_TREE} activeNodeNames={[]} />,
  );
  const unmeasuredNodes = reactFlowNodeSnapshots.at(-1);

  expect(reactFlowOnNodesChange).toEqual(expect.any(Function));
  act(() => {
    reactFlowOnNodesChange(unmeasuredNodes.map((node) => ({
      id: node.id,
      type: "dimensions",
      dimensions: { width: 200, height: 80 },
    })));
  });
  const measuredNodes = reactFlowNodeSnapshots.at(-1);

  rerender(
    <MissionBtRunView xml={POPULATED_TREE} activeNodeNames={["bt_1"]} />,
  );
  const activeNodes = reactFlowNodeSnapshots.at(-1);

  expect(activeNodes).toHaveLength(measuredNodes.length);
  expect(activeNodes.find((node) => node.id === "bt_1")?.data.isActive).toBe(true);
  measuredNodes.forEach((node, index) => {
    expect(activeNodes[index].measured).toEqual({ width: 200, height: 80 });
    if (activeNodes[index].data.isActive === node.data.isActive) {
      expect(activeNodes[index]).toBe(node);
    }
  });
});
