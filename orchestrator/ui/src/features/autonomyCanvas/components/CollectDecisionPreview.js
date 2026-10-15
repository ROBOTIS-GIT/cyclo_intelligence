import { resultMissionName, resultTask } from "./autonomyPresentation";
import React from "react";

// Presentation only: this component never loads, starts, or cancels a mission.
export default function CollectDecisionPreview({ result, phase, error }) {
  const decision = result?.decision;
  return <div className="collect-observation-description" aria-label="Mission decision preview">
    {error ? <p role="alert" className="operate-qwen-error">{error}</p> : decision ? <>
      <dl className="collect-decision-fields">
        <div><dt>{resultTask(result) !== "mission_selection" ? "Supervision" : "Decision"}</dt><dd>{decision.decision}</dd></div>
        <div><dt>Mission</dt><dd>{resultMissionName(result)}</dd></div>
      </dl>
      <p className="operate-qwen-text">{decision.reason}</p>
    </> : <p className="operate-sidebar-note">{result ? "No valid mission decision for this frame." : phase}</p>}
  </div>;
}
