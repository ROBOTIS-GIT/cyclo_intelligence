import {useState} from "react";
import {MdDeleteOutline} from "react-icons/md";
import {collectionRequest} from "../../../utils/collectionApi";

export default function ReasoningModelDelete({model, disabled, onDeleted}) {
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  const removable=model && !model.id.startsWith("base:") && !model.id.startsWith("legacy:");
  async function remove() {
    if(!removable || disabled || busy)return;
    if(!window.confirm(`Delete “${model.model_name}”?\nThis permanently removes the saved model files. Dataset labels are kept.`))return;
    setBusy(true);setError("");
    try {
      await collectionRequest(`qwen/models/${encodeURIComponent(model.id)}`,undefined,"DELETE");
      await onDeleted?.();
    } catch(err) {setError(err.message);}
    finally {setBusy(false);}
  }
  return <>
    <button type="button" aria-label="Delete selected reasoning model"
      title={removable?"Delete selected reasoning model":"Select a trained model to delete"}
      disabled={!removable||disabled||busy} onClick={remove}><MdDeleteOutline/></button>
    {error&&<span role="alert" style={{fontSize:11,color:"var(--mc-danger, #bd4545)"}}>{error}</span>}
  </>;
}
