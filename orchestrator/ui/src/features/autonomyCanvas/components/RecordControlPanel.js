import "./RecordControlPanel.css";

// Capture remains mounted independently of map editing and movement controls.
export default function RecordControlPanel({ capture, footer, error }) {
  return <section className="record-control-panel" aria-label="Record control panel">
    <header className="record-control-header">
      <strong className="record-control-title">Capture</strong>
    </header>
    <div className="record-control-content">{capture}</div>
    {error && <p role="alert" className="record-control-error">{error}</p>}
    <footer className="record-control-footer">{footer}</footer>
  </section>;
}
