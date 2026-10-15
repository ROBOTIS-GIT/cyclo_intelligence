import FrameRangeDeleteButton from "./FrameRangeDeleteButton";

export default function AnnotationFrameNavigation({ draft, frames, index, disabled, deletionBlocked, progress }) {
  return <div className="collect-actions collect-frame-navigation">
    <button type="button" aria-label="Previous frame" disabled={disabled || index <= 0}
      onClick={() => draft.selectFrame(frames[index - 1])}>←</button>
    <span className="labeling-frame-position" aria-label="Current frame">{index >= 0 ? index + 1 : "–"} / {frames.length}</span>
    <button type="button" aria-label="Next frame" disabled={disabled || !frames.length || index >= frames.length - 1}
      onClick={() => draft.selectFrame(frames[index + 1])}>→</button>
    {progress && <span className="labeling-frame-progress" role="status" aria-label="Labeling progress">{progress}</span>}
    <FrameRangeDeleteButton episode={draft.episode} disabled={disabled || deletionBlocked || !draft.deleteFrames} onDelete={draft.deleteFrames}/>
  </div>;
}
