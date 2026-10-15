export function RecordToolButton({ label, active = false, children, ...props }) {
  return <button type="button" aria-label={label} title={label} aria-pressed={active || undefined}
    className={`record-authoring-button${active ? " is-active" : ""}`} {...props}>{children}</button>;
}
