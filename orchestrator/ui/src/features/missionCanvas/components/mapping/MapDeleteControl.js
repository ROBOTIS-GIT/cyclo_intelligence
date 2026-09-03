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

import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { MdDelete } from "react-icons/md";
import { getNavigationMissions } from "../../../../utils/navigationMissionsApi";
import { mapNameFromPgmPath } from "../../lib/missionNames";
import { MapEditToolButton } from "../primitives";
import { DeleteMapDialog } from "../dialogs";

// Mapping HUD control: a trash icon opening a saved-map list popover (the
// HUD popover idiom); picking a map raises the warning confirm popup, and
// the delete cascades to sidecars and missions on the backend.
export default function MapDeleteControl({ files, disabled = false, protectedPaths = [], onDelete, dialogHost }) {
  const [open, setOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleteMissionCount, setDeleteMissionCount] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const deleteTargetPathRef = useRef("");
  const requestDelete = (file) => {
    deleteTargetPathRef.current = file.path;
    setDeleteTarget(file);
    setDeleteMissionCount(null);
    setOpen(false);
    getNavigationMissions(mapNameFromPgmPath(file.path))
      .then((response) => {
        if (deleteTargetPathRef.current !== file.path) return;
        const missions = Array.isArray(response?.missions) ? response.missions : [];
        setDeleteMissionCount(missions.length);
      })
      .catch(() => {
        // Unknown count: the popup falls back to generic wording.
      });
  };
  return (
    <div className="relative">
      <MapEditToolButton
        label="Delete Map"
        active={open}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        <MdDelete size={16} aria-hidden="true" />
      </MapEditToolButton>
      {open && (
        <div
          className="absolute left-0 top-[calc(100%+6px)] grid w-64 max-h-64 gap-1.5 overflow-y-auto p-2"
          role="menu"
          aria-label="Saved maps"
          style={{ borderRadius: 12, backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)", boxShadow: "var(--mc-shadow)" }}
        >
          {files.map((file) => {
            const inUse = protectedPaths.includes(file.path);
            return (
              <button
                key={file.path}
                type="button"
                disabled={inUse}
                aria-label={`Delete map ${file.path}`}
                title={inUse ? "Stop navigation before deleting this map" : `Delete ${file.path}`}
                onClick={() => requestDelete(file)}
                className="h-8 w-full min-w-0 inline-flex items-center gap-2 px-2.5 text-left text-[12px] font-mono disabled:opacity-45"
                style={{ borderRadius: 8, border: "1px solid var(--mc-border)", backgroundColor: "var(--mc-surface-2)", color: "var(--mc-text)" }}
              >
                <span className="flex-1 truncate">{file.path}</span>
                <MdDelete size={13} aria-hidden="true" style={{ color: "var(--mc-danger)" }} />
              </button>
            );
          })}
          {files.length === 0 && (
            <div className="text-[12px]" style={{ color: "var(--mc-text-muted)" }}>
              No saved maps yet.
            </div>
          )}
        </div>
      )}
      {/* Portal to the page root: escapes the HUD's backdrop-filter (which
          hijacks position:fixed) while staying inside the --mc-* token
          scope, so the popup centers on screen with a proper surface. */}
      {deleteTarget && dialogHost?.current && createPortal(
        <DeleteMapDialog
          key={deleteTarget.path}
          file={deleteTarget}
          missionCount={deleteMissionCount}
          busy={deleting}
          onCancel={() => setDeleteTarget(null)}
          onConfirm={async () => {
            setDeleting(true);
            try {
              await onDelete(deleteTarget.path);
              setDeleteTarget(null);
            } finally {
              setDeleting(false);
            }
          }}
        />,
        dialogHost.current,
      )}
    </div>
  );
}
