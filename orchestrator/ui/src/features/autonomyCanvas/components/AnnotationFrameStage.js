// Shared frame stage for reasoning labels and object masks.
export default function AnnotationFrameStage({ episode, controls, tools, footer, children, compact = false }) {
  const name = episode?.storage_path?.split("/").filter(Boolean).pop() || episode?.scenario || "Select an episode";
  return <section className={`annotate-frame-stage${compact ? " annotate-frame-stage-compact" : ""}`} aria-label="Frame preview">
    <header className="annotate-frame-heading">
      <div><strong>{name}</strong><span>{episode?.map_name || "Saved dataset"}</span></div>
      {compact && <div className="annotate-frame-navigation">{controls}</div>}
    </header>
    {!compact && <div className="annotate-frame-navigation">{controls}</div>}
    {tools && <div className="annotate-frame-tools">{tools}</div>}
    <div className="annotate-frame-viewport">{children}</div>
    {footer && <div className="annotate-frame-footer">{footer}</div>}
  </section>;
}
