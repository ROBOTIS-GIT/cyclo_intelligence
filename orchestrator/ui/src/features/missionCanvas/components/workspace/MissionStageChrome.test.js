// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { render, screen } from "@testing-library/react";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_NAVIGATE,
  STAGE_RUN,
} from "../../lib/stages";
import MissionStageChrome from "./MissionStageChrome";

jest.mock("./StageHeader", () => ({ workspaceStage }) => (
  <div data-testid="header">{workspaceStage}</div>
));
jest.mock("../stages/DesignStageSidebar", () => () => <div data-testid="design-sidebar" />);
jest.mock("../stages/MappingStageSidebar", () => () => <div data-testid="mapping-sidebar" />);
jest.mock("../stages/NavigationStage", () => ({
  NavigationStageSidebar: () => <div data-testid="navigation-sidebar" />,
}));
jest.mock("../stages/RunStage", () => ({
  RunStageSidebar: () => <div data-testid="run-sidebar" />,
}));

const groups = {
  header: {},
  map: { mappingEditorActive: false, waypointBtLayer: null },
  sidebar: { design: {}, mapping: {}, navigation: {}, run: {} },
};

function renderChrome(stage, overrides = {}) {
  return render(
    <MissionStageChrome
      {...groups}
      {...overrides}
      stage={stage}
    >
      <section data-testid="scene-surface" />
    </MissionStageChrome>,
  );
}

test.each([
  [STAGE_AUTHORING, "design-sidebar"],
  [STAGE_MAPPING, "mapping-sidebar"],
  [STAGE_NAVIGATE, "navigation-sidebar"],
  [STAGE_RUN, "run-sidebar"],
])("renders the %s stage sidebar beside the scene", (stage, sidebarId) => {
  renderChrome(stage);

  expect(screen.getByTestId("header")).toHaveTextContent(stage);
  expect(screen.getByTestId("scene-surface")).toBeInTheDocument();
  expect(screen.getByTestId(sidebarId)).toBeInTheDocument();
});

test("Map Edit keeps the scene as the grid child without an aside", () => {
  renderChrome(STAGE_MAP_EDIT, {
    map: { ...groups.map, mappingEditorActive: true },
  });

  expect(screen.getByTestId("scene-surface")).toBeInTheDocument();
  expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
});

test("an authoring BT layer hides the Design sidebar but keeps the scene", () => {
  const layer = { spot: { id: "wp-1" } };
  renderChrome(STAGE_AUTHORING, {
    map: { ...groups.map, waypointBtLayer: layer },
  });

  expect(screen.getByTestId("scene-surface")).toBeInTheDocument();
  expect(screen.queryByTestId("design-sidebar")).not.toBeInTheDocument();
});

test("Run keeps its sidebar beside the scene, including during its BT view", () => {
  renderChrome(STAGE_RUN, {
    map: { ...groups.map },
  });

  expect(screen.getByTestId("scene-surface")).toBeInTheDocument();
  expect(screen.getByTestId("run-sidebar")).toBeInTheDocument();
});
