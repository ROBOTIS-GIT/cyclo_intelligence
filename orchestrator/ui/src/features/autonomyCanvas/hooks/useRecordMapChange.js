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

export default function useRecordMapChange({
  runCommand, collectDraft, collectExecution, stopActiveNavigationSession, qwenObservation,
  collectObservations, setMessage, running, runRuntimeActive, missionMapLoaded,
  handleOpenRunMapDialog,
}) {
  const shutdownRecordMap = async () => (await runCommand("Change map", async () => {
    collectDraft.stopRecording();
    if (collectDraft.active && !(await collectDraft.finish({ allowEmpty: true }))) return false;
    if ((await collectExecution.stop()) === false) return false;
    const [navigationResult, observationResult] = await Promise.allSettled([
      stopActiveNavigationSession({ runnerAlreadyStopped: true }),
      qwenObservation.needsStop ? qwenObservation.stop() : Promise.resolve(true),
    ]);
    if (navigationResult.status === "rejected") throw navigationResult.reason;
    if (observationResult.status === "rejected") throw observationResult.reason;
    return navigationResult.value !== false && observationResult.value !== false;
  })) === true;

  const openRecordMapPicker = async () => {
    if (collectObservations.draft || collectObservations.planDirty || collectObservations.routeEditor?.dirty) {
      setMessage("Save or discard observation edits before changing maps.");
      return;
    }
    if (running || runRuntimeActive || missionMapLoaded || collectDraft.active) {
      if (!(await shutdownRecordMap())) return;
    }
    handleOpenRunMapDialog();
  };

  return {
    openRecordMapPicker,
  };
}
