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
  ConfirmDialog,
  LoadMapDialog,
  SaveMapDialog,
  SaveMissionDialog,
} from "../dialogs";

export default function MissionCanvasDialogLayer({
  saveMap,
  saveMission,
  renameMission,
  duplicateMission,
  deleteMission,
  unsaved,
  designLoad,
  runLoad,
  editLoad,
}) {
  return (
    <>
      <SaveMapDialog {...saveMap} />
      <SaveMissionDialog {...saveMission} disallowExisting />
      <SaveMissionDialog
        {...renameMission}
        title="Rename Mission"
        fieldLabel="New mission name"
        inputAriaLabel="Rename mission name"
        submitLabel="Rename"
        currentName=""
        disallowExisting
      />
      <SaveMissionDialog
        {...duplicateMission}
        title="Duplicate Mission"
        fieldLabel="New mission name"
        inputAriaLabel="Duplicate mission name"
        submitLabel="Duplicate"
        currentName=""
        disallowExisting
        hint="Duplicates the last saved state."
      />
      <ConfirmDialog {...deleteMission} title="Delete Mission" confirmLabel="Delete" />
      <ConfirmDialog {...unsaved} title="Unsaved changes" confirmLabel="Discard" />
      <LoadMapDialog
        {...designLoad}
        title="Load Map"
        fieldLabel="Map"
        selectAriaLabel="Design mission map file"
        missionSelectAriaLabel="Design mission file"
      />
      <LoadMapDialog
        {...runLoad}
        title="Load Map"
        fieldLabel="Map"
        selectAriaLabel={runLoad.navigationOnly
          ? "Navigation map file"
          : "Run mission map file"}
        missionSelectAriaLabel="Run mission file"
      />
      <LoadMapDialog
        {...editLoad}
        title="Load Map"
        fieldLabel="Map"
        selectAriaLabel="PGM map"
      />
    </>
  );
}
