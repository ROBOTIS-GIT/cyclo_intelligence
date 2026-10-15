import { MdClose } from "react-icons/md";
import { jobRunning } from "./trainingJobStatus";

export default function LabelerFeedback({ family, workflow }) {
  const { state, error, connectionError, working, running, dismissError } = workflow;
  return <>
    {state?.training?.status === "cancelled" && <small role="status">Fine-tuning cancelled</small>}
    {connectionError && <p role="alert">{connectionError}</p>}
    {error && <p role="alert">{error}</p>}
    {["training", "batch", "reasoning_training"].filter(kind => state?.[kind]?.error && !jobRunning(state[kind])).map(kind =>
      <div key={state[kind].id || kind} className="collect-actions">
        <p role="alert">{family === "qwen" ? state[kind].error.split("\n")[0].slice(0, 200) : state[kind].error}</p>
        {family === "qwen" && <button type="button" title="Clear error" aria-label={`Clear Qwen ${kind.replaceAll("_", " ")} error`}
          disabled={working || running} onClick={() => dismissError(kind)}><MdClose size={18}/></button>}
      </div>)}
  </>;
}
