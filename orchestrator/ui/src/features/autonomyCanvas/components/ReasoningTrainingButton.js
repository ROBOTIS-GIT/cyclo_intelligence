import TrainingCancelButton from "./TrainingCancelButton";
import {MdSchool} from "react-icons/md";
import {jobRunning} from "./LabelerActions";

export default function ReasoningTrainingButton({training,open,onClick}) {
  const job=training.job, running=jobRunning(job);
  const failed=["failed","completed_with_errors"].includes(job?.status)||Boolean(training.error);
  const percent=Math.min(100,Math.round((job?.step||0)/Math.max(1,job?.steps||0)*100));
  return <div className="learn-training-launcher">
    <span className="learn-training-map" title={training.mapName}>{training.mapName}</span>
    <button type="button" className={`learn-training-button${failed?" is-failed":""}`}
      aria-label="Train model" aria-expanded={open} aria-haspopup="dialog"
      title={running?`${job.model_name||"Reasoning model"} · ${job.step||0}/${job.steps||"…"} steps`:failed?"View training error":"Train reasoning model"}
      onClick={onClick}>
      {running && <span className="learn-training-fill" style={{height:`${percent}%`}} aria-hidden="true"/>}
      <MdSchool size={20}/>
      {running && <span className="learn-training-percent" role="progressbar" aria-label="Reasoning training progress"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={job.steps ? percent : undefined}>{job.steps?`${percent}%`:"…"}</span>}
    </button>
    <TrainingCancelButton label="Cancel model training" job={job} onCancel={training.cancel} pending={training.working}/>
  </div>;
}
