// Structural assertions verify map/bubble ownership and positioning.
/* eslint-disable testing-library/no-container, testing-library/no-node-access */
// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { fireEvent, render, screen } from "@testing-library/react";
import {
  STAGE_OPERATE,
  STAGE_COLLECT,
  STAGE_LEARN,
} from "../../../mapWorkspace/lib/stages";
import AutonomyStageChrome from "./AutonomyStageChrome";

jest.mock("../CollectSidebar", () => {
  const React = require("react");
  return ({mode,mapCanvas,onTestModel}) => {
    const [value,setValue]=React.useState("");
    return <div data-testid="data-sidebar" data-mode={mode}>
      <input aria-label="Shared selection" value={value} onChange={e=>setValue(e.target.value)}/>
      {mode === "live" && mapCanvas}
      <button onClick={() => onTestModel?.({id: "learned-model"})}>Test model</button>
    </div>;
  };
});
jest.mock("../../../mapWorkspace/components/workspace/StageHeader", () => ({ workspaceStage }) => (
  <div data-testid="header">{workspaceStage}</div>
));
const groups = {
  header: {},
  map: { mappingEditorActive: false, waypointBtLayer: null },
  sidebar: { design: {}, mapping: {}, navigation: {}, run: {} },
};

function renderChrome(stage, overrides = {}) {
  return render(
    <AutonomyStageChrome
      {...groups}
      {...overrides}
      stage={stage}
    >
      <section data-testid="scene-surface" />
    </AutonomyStageChrome>,
  );
}

test("Record and Learn reuse the data workspace without losing selection", () => {
  const view=renderChrome(STAGE_COLLECT);
  fireEvent.change(screen.getByLabelText("Shared selection"),{target:{value:"episode-4/frame-7"}});
  for (const stage of [STAGE_LEARN, STAGE_COLLECT, STAGE_LEARN]) {
    view.rerender(<AutonomyStageChrome {...groups} stage={stage}><section data-testid="scene-surface"/></AutonomyStageChrome>);
    expect(screen.getByLabelText("Shared selection")).toHaveValue("episode-4/frame-7");
    expect(screen.getByTestId("data-sidebar")).toHaveAttribute("data-mode",stage === STAGE_COLLECT ? "live" : "dataset");
    expect(screen.queryByTestId("run-sidebar")).not.toBeInTheDocument();
    expect(Boolean(screen.queryByTestId("scene-surface"))).toBe(stage === STAGE_COLLECT);
  }
});


test.each(["annotate", "train", "reasoning"])("legacy %s navigation opens the Learn dataset workspace", stage => {
  renderChrome(stage);
  expect(screen.getByTestId("header")).toHaveTextContent(STAGE_LEARN);
  expect(screen.getByTestId("data-sidebar")).toHaveAttribute("data-mode", "dataset");
  expect(screen.queryByTestId("scene-surface")).not.toBeInTheDocument();
});

test("Learn forwards the model preview callback for returning to Record", () => {
  const onTestModel = jest.fn();
  renderChrome(STAGE_LEARN, { sidebar: {...groups.sidebar, collect: {onTestModel}} });
  fireEvent.click(screen.getByRole("button", {name: "Test model"}));
  expect(onTestModel).toHaveBeenCalledWith({id: "learned-model"});
});


test("Operate uses the full scene with folded tools instead of a reserved sidebar column",()=>{
 const view=renderChrome(STAGE_OPERATE);
 const workspace=view.container.querySelector('.operate-experience');
 expect(workspace).toContainElement(screen.getByTestId('scene-surface'));
 expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Status and settings'}));
 expect(screen.getByRole('complementary',{name:'Status and settings'})).toBeInTheDocument();
 expect(workspace).toContainElement(screen.getByTestId('scene-surface'));
 fireEvent.click(screen.getByRole('button',{name:'Status and settings'}));
 expect(screen.queryByRole('complementary')).not.toBeInTheDocument();
});
