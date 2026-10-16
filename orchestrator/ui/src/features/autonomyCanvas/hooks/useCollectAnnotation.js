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

import { useEffect, useRef, useState } from "react";
import { decisionNeedsMission, episodeReasonPresets, frameReasoningContract, labelMatchesFrame } from "../utils/reasoningLabelContract";

import { frameMatchesEpisode } from "../utils/reasoningDataset";

const EMPTY_ITEMS = [];

export default function useCollectAnnotation({ draft, mapName }) {
  const [unlabeledOnly, setUnlabeledOnly] = useState(false);
  const [decision, setDecision] = useState("wait");
  const [missionId, setMissionId] = useState("");
  const [reason, setReason] = useState("");
  const maskEditor = useRef(null);
  const savingRef = useRef(false);
  const [saving, setSaving] = useState(false);
  const [maskBusy, setMaskBusy] = useState(false);
  const [receipt, setReceipt] = useState("");
  const [loadedFrame, setLoadedFrame] = useState(null);
  const { episode, active, selectedFrame, busy: draftBusy } = draft;
  const busy = draftBusy || saving;
  const autoSelectedEpisode = useRef(null);
  const annotations = episode?.annotations || EMPTY_ITEMS;
  const observations = episode?.observations || EMPTY_ITEMS;
  const capturedMissions = episode?.observation_binding
    ? selectedFrame?.catalog?.missions || episode?.catalog?.missions : null;
  const capturedDisplay = capturedMissions?.map(m => ({ ...m, mission_name:
    episode?.current_catalog?.missions?.find(current => current.mission_id === m.mission_id)?.mission_name || m.mission_name }));
  const catalogMissions = capturedDisplay || episode?.current_catalog?.missions
    || episode?.display_missions
    || episode?.catalog?.missions
    || EMPTY_ITEMS;
  const contract = frameReasoningContract(selectedFrame);
  const frameMission = contract.activeMission;
  const missions = contract.supervision || contract.candidateIds === null
    ? catalogMissions : catalogMissions.filter(m => contract.candidateIds.includes(m.mission_id));
  const selectedLabel = annotations.find((a) => a.frame_ids[0] === selectedFrame?.frame_id);
  const reasonPresets = draft.reasonPresets || episodeReasonPresets(annotations, observations.filter(frame => frameMatchesEpisode(episode, frame)));
  const savedDecision = selectedLabel?.target?.decision || (contract.patrol || contract.supervision ? "continue" : "wait");
  const savedMissionId = contract.supervision
    ? frameMission?.mission_id || ""
    : decisionNeedsMission(savedDecision, contract.task) ? selectedLabel?.target?.mission_id || "" : "";
  const episodeFrameValid = frameMatchesEpisode(episode, selectedFrame);
  const eligibleFrames = observations.filter(frame => frameMatchesEpisode(episode, frame));
  const unsupportedSavedLabel = Boolean(selectedLabel && !labelMatchesFrame(selectedFrame, selectedLabel.target));
  useEffect(() => {
    if (selectedFrame) autoSelectedEpisode.current = episode?.id;
    if (
      episode?.id &&
      eligibleFrames.length &&
      !selectedFrame &&
      autoSelectedEpisode.current !== episode.id
    ) {
      autoSelectedEpisode.current = episode.id;
      draft.selectFrame(
        eligibleFrames.find(
          (f) =>
            !annotations.some(
              (a) => a.frame_ids[0] === f.frame_id && a.review_status === "approved" && labelMatchesFrame(f, a.target),
            ),
        ) || eligibleFrames[0],
      );
    }
  }, [episode?.id, observations, annotations, selectedFrame, draft, eligibleFrames]);
  function resetLabel() {
    setDecision(savedDecision);
    setMissionId(savedMissionId);
    setReason(selectedLabel?.target?.reason || "");
    setReceipt("");
    setMaskBusy(false);
  }
  useEffect(() => { resetLabel();
    // User edits are preserved until the selected frame or its saved revision changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFrame?.frame_id, episode?.id, selectedLabel?.revision, selectedFrame?.metadata, selectedFrame?.context]);
  const labelDirty = !!selectedFrame && episodeFrameValid && !active && (decision !== savedDecision ||
    missionId !== savedMissionId || reason !== (selectedLabel?.target?.reason || "") ||
    unsupportedSavedLabel);
  const labeled = (id) =>
    annotations.some((a) => a.frame_ids[0] === id && a.review_status === "approved" &&
      frameMatchesEpisode(episode, observations.find(frame => frame.frame_id === id)) &&
      labelMatchesFrame(observations.find(frame => frame.frame_id === id), a.target));
  const reviewFrames = eligibleFrames.filter((f) => !unlabeledOnly || !labeled(f.frame_id));
  const frameIndex = reviewFrames.findIndex((f) => f.frame_id === selectedFrame?.frame_id);
  const reviewBusy = active || busy || maskBusy;
  const reviewEpisodes = draft.episodes.filter(
    (ep) => ep.ended_at && (!mapName || ep.map_name === mapName),
  );
  const invalidDecisionContext = !episodeFrameValid || !labelMatchesFrame(selectedFrame, { decision, mission_id: missionId });
  const missionRequired = decisionNeedsMission(decision, contract.task);
  const deletedSelectedMission = missionId && !missions.some((mission) => (
    mission.mission_id === missionId
  ))
    ? [
      ...(selectedLabel?.catalog?.missions || EMPTY_ITEMS),
      ...(episode?.catalog?.missions || EMPTY_ITEMS),
    ].find((mission) => (
      mission.mission_id === missionId
    )) || { mission_id: missionId, mission_name: missionId }
    : null;
  const selectedMissionUnavailable = missionRequired && !contract.supervision && Boolean(deletedSelectedMission);
  async function save(status) {
    if (
      invalidDecisionContext ||
      selectedMissionUnavailable ||
      active ||
      savingRef.current ||
      busy ||
      maskBusy ||
      !reason.trim() ||
      (missionRequired && !missionId)
    )
      return;
    savingRef.current = true;
    setSaving(true);
    setReceipt("");
    try {
      // Reasoning labels are independent of object masks.
      if (loadedFrame !== selectedFrame?.frame_id) return false;
      if (
        await draft.annotate({
          decision,
          mission_id: missionRequired ? missionId : null,
          reason,
          annotator: "collect-ui",
          review_status: status,
        })
      ) {
        setReceipt("Annotation saved for training.");
        return true;
      } else {
        setReceipt("Decision was not saved. Retry Save.");
      }
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }
  return {
    decision,
    setDecision,
    missionId,
    setMissionId,
    reason,
    setReason,
    maskEditor,
    saving,
    maskBusy,
    setMaskBusy,
    receipt,
    loadedFrame,
    setLoadedFrame,
    busy,
    annotations,
    observations,
    episodeFrameValid,
    eligibleFrames,
    missions,
    reasonPresets,
    selectedLabel,
    unlabeledOnly,
    setUnlabeledOnly,
    labeled,
    reviewFrames,
    frameIndex,
    reviewBusy,
    reviewEpisodes,
    frameMission,
    reasoningTask: contract.task,
    supervision: contract.supervision,
    decisionChoices: contract.decisions,
    unsupportedSavedLabel,
    invalidDecisionContext,
    missionRequired,
    deletedSelectedMission,
    selectedMissionUnavailable,
    save,
    labelDirty,
    resetLabel,
  };
}
