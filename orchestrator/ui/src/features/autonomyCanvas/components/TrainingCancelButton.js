import {MdStop} from "react-icons/md";
import {jobRunning} from "./trainingJobStatus";

export default function TrainingCancelButton({job, onCancel, pending=false, label="Cancel fine-tuning",showLabel=false}) {
  if(!onCancel || !jobRunning(job))return null;
  const cancelling=pending || (job.status==="cancelling" && !job.error);
  return <button type="button" className="training-cancel-button" aria-label={label}
    title={cancelling?"Cancelling…":label} aria-busy={cancelling} disabled={cancelling} onClick={onCancel}>
    <MdStop size={19}/>{showLabel&&<span>{cancelling?"Cancelling…":label}</span>}
  </button>;
}
