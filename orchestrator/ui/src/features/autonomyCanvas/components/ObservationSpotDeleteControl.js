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

import { useState } from "react";
import { FiTrash2 } from "react-icons/fi";

// Match the saved-map/episode deletion pattern, including unused saved spots.
export default function ObservationSpotDeleteControl({ spots = [], selectedSpot = null, disabled, onDelete }) {
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const locked = disabled || deleting;
  async function remove(spot) {
    if (!window.confirm(`Delete saved spot "${spot.label}"? It will also be removed from the observation plan. This cannot be undone.`)) return;
    setDeleting(true);
    try {
      if (await onDelete(spot.id)) setOpen(false);
    } finally {
      setDeleting(false);
    }
  }
  return <div className="observation-spot-delete">
    <button type="button" title="Delete saved spot" aria-label="Delete saved spot" aria-haspopup={selectedSpot ? undefined : "menu"}
      aria-expanded={selectedSpot ? undefined : open} disabled={locked || !spots.length} onClick={() => selectedSpot ? remove(selectedSpot) : setOpen(value => !value)}>
      <FiTrash2 />
    </button>
    {open && !selectedSpot && <div className="observation-spot-delete-list" role="menu" aria-label="Saved spots"
      onKeyDown={event => { if (event.key === "Escape") setOpen(false); }}>
      {spots.map(spot => <button key={spot.id} type="button" role="menuitem" disabled={locked}
        title={spot.label} aria-label={`Delete ${spot.label}`} onClick={() => remove(spot)}>
        <span>{spot.label}</span><FiTrash2 />
      </button>)}
    </div>}
  </div>;
}
