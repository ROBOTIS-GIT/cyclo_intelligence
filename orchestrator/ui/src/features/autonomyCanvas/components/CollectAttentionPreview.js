export default function CollectAttentionPreview({result}) {
  const attention=result?.attention_preview;
  const image=attention?.image_base64 || result?.image_base64;
  return <figure className="collect-attention-preview" aria-label="Reasoning attention">
    <figcaption title={attention ? `LLM layer ${attention.layer} · Last prompt query to image · Fixed scale` : undefined}>Attention</figcaption>
    {image ? <img src={`data:image/jpeg;base64,${image}`}
      alt={attention?.image_base64 ? "Attention over the analyzed camera frame" : "Analyzed camera frame"}/>
      : <div className="collect-attention-placeholder">Waiting for reasoning</div>}
    {result?.attention_error && <span role="status" title={result.attention_error}>Attention unavailable for this result.</span>}
  </figure>;
}
