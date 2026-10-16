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

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import LabelerWorkflowControls from "./LabelerWorkflowControls";
import { collectionRequest } from "../../../utils/collectionApi";

jest.mock("../../../utils/collectionApi", () => ({ collectionRequest: jest.fn() }));
const idle = { training: { status: "idle" }, batch: { status: "idle" }, models: [] };
const episode = { id: "ep", map_name: "house", ended_at: 1,
  observations: [{ frame_id: "one" }, { frame_id: "two" }, { frame_id: "three" }],
  qwen_sample: { schema_version: "user_count_v1", frame_ids: ["one", "two"] },
  sam_sample: { schema_version: "user_count_v1", frame_ids: ["one", "two"] },
  annotations: ["one", "two"].map(id => ({ frame_ids: [id], review_status: "approved" })),
  regions: ["one", "two"].map(frame_id => ({ frame_id, object_label: "person", review_status: "approved" })),
};
beforeEach(() => collectionRequest.mockReset().mockResolvedValue(idle));

for (const family of ["qwen", "segmentation"]) {
  test(`${family} samples the requested count`, async () => {
    const sampled = { ...episode };
    const onSampled = jest.fn();
    collectionRequest.mockImplementation(async path => path.endsWith("/sample") ? sampled : idle);
    render(<LabelerWorkflowControls family={family} episode={episode} episodes={[episode]} onSampled={onSampled}/>);
    expect(screen.getByRole("button", { name: "Resample frames" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Sample count"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Resample frames" }));
    await waitFor(() => expect(onSampled).toHaveBeenCalledWith(sampled));
    expect(collectionRequest).toHaveBeenCalledWith(`episodes/ep/${family}/sample`, { count: 3 });
    expect(screen.queryByLabelText("Model name")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Training steps")).not.toBeInTheDocument();
  });

  test(`${family} trains saved samples across the map and keeps approved labels during generation`, async () => {
    const old = { id: "old", model_name: "Previous", map_name: "house", strategy: "sam_multiclass_instances_v1" };
    collectionRequest.mockResolvedValue({ ...idle, models: [old] });
    render(<LabelerWorkflowControls family={family} episode={episode} episodes={[episode]}/>);
    fireEvent.click(screen.getByRole("button", { name: "Auto-label" }));
    await screen.findByRole("button", { name: "Previous" });
    fireEvent.click(screen.getByRole("button", { name: "Auto-label" }));
    fireEvent.click(screen.getByRole("button", { name: "Fine-tune" }));
    await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith(`${family}/auto-train`, { map_name: "house" }));
    fireEvent.click(screen.getByRole("button", { name: "Auto-label" }));
    fireEvent.click(screen.getByRole("button", { name: "Auto-label episode" }));
    await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith(`${family}/auto-label`, {
      model_id: "old", episode_ids: ["ep"], replace_unreviewed: false,
    }));
  });

  test(`${family} selects a completed job model and preserves later manual selection`, async () => {
    jest.useFakeTimers();
    const old = { id: "old", model_name: "Previous", map_name: "house", strategy: "sam_multiclass_instances_v1", created_at: 1 };
    const fresh = { ...old, id: "fresh", model_name: "New model", created_at: 2 };
    const queued = { id: family === "qwen" ? "job" : "fresh", model_name: fresh.model_name, map_name: "house", status: "queued" };
    let workflow = { ...idle, models: [old] };
    collectionRequest.mockImplementation(async path => {
      if (path.endsWith("/auto-train")) { workflow = { ...workflow, training: queued }; return queued; }
      return workflow;
    });
    const onJobChange = jest.fn();
    const view = render(<LabelerWorkflowControls family={family} episode={episode} episodes={[episode]} onJobChange={onJobChange}/>);
    try {
      await act(async () => { jest.advanceTimersByTime(0); });
      fireEvent.click(screen.getByRole("button", { name: "Fine-tune" }));
      await act(async () => { jest.advanceTimersByTime(0); });
      workflow = { ...workflow, training: { ...queued, status: "completed" }, models: [old, fresh] };
      await act(async () => { jest.advanceTimersByTime(3000); });
      fireEvent.click(screen.getByRole("button", { name: "Auto-label" }));
      expect(screen.getByRole("button", { name: "New model" })).toHaveAttribute("aria-pressed", "true");
      fireEvent.click(screen.getByRole("button", { name: "Previous" }));
      await act(async () => { jest.advanceTimersByTime(3000); });
      expect(screen.getByRole("button", { name: "Previous" })).toHaveAttribute("aria-pressed", "true");
      expect(onJobChange).toHaveBeenLastCalledWith(workflow.batch);
      const reads = collectionRequest.mock.calls.filter(([path]) => path.endsWith("/workflow")).length;
      view.rerender(<LabelerWorkflowControls family={family} episode={{...episode, id: "other"}} episodes={[episode]}/>);
      await act(async () => { jest.advanceTimersByTime(0); });
      expect(collectionRequest.mock.calls.filter(([path]) => path.endsWith("/workflow"))).toHaveLength(reads);
    } finally { view.unmount(); jest.useRealTimers(); }
  });
}

test("a slow workflow response cannot restore a cancelled training job", async () => {
  jest.useFakeTimers();
  const job = { id: "job", status: "training" };
  let finishPoll;
  let reads = 0;
  collectionRequest.mockImplementation(path => {
    if (path.endsWith("/cancel")) return Promise.resolve({ ...idle, training: { ...job, status: "cancelled" } });
    if (++reads === 1) return Promise.resolve({ ...idle, training: job });
    return new Promise(resolve => { finishPoll = resolve; });
  });
  const view = render(<LabelerWorkflowControls family="qwen" episode={episode} episodes={[episode]}/>);
  try {
    await act(async () => { jest.advanceTimersByTime(0); });
    await act(async () => { jest.advanceTimersByTime(3000); });
    fireEvent.click(screen.getByRole("button", { name: "Cancel fine-tuning" }));
    await act(async () => { jest.advanceTimersByTime(0); });
    await act(async () => { finishPoll({ ...idle, training: job }); });
    expect(screen.getByText("Fine-tuning cancelled")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Cancel fine-tuning" })).not.toBeInTheDocument();
  } finally { view.unmount(); jest.useRealTimers(); }
});


test.each([["qwen", "qwen_sample_approved_count", "frames"], ["segmentation", "sam_sample_trainable_instance_count", "instances"]])(
  "%s includes reviewed samples from every saved episode on the selected map", async (family, field, unit) => {
    const episodes = [
      { id: "a", map_name: "house", ended_at: 1, episode_type: "observation", [field]: 30 },
      { id: "b", map_name: "house", ended_at: 1, episode_type: "mission", [field]: 40 },
      { id: "open", map_name: "house", ended_at: null, [field]: 80 },
      { id: "other", map_name: "zebra", ended_at: 1, [field]: 100 },
    ];
    render(<LabelerWorkflowControls family={family} episode={null} episodes={episodes}/>);
    fireEvent.click(screen.getByRole("button", { name: "Fine-tune" }));
    expect(screen.getByText(`70 approved sampled ${unit} on this map`)).toBeVisible();
    await waitFor(() => expect(screen.getByRole("button", { name: "Start fine-tuning" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Start fine-tuning" }));
    await waitFor(() => expect(collectionRequest).toHaveBeenCalledWith(`${family}/auto-train`, { map_name: "house" }));
  },
);
