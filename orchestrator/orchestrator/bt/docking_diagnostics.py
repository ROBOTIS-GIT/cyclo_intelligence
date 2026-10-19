"""Bounded diagnostic history; failure artifacts are written off the control thread."""
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import threading
import time
import uuid
import cv2


class DockDiagnostics:
    def __init__(self, marker_id, config, image_mode, root=None):
        self.marker_id = marker_id
        self.context = {'marker_id': marker_id, 'config': vars(config), 'image_mode': image_mode}
        self.events = deque(maxlen=160)
        self.frames = deque(maxlen=32)
        self.last_frame = 0.
        self.root = Path(root or os.environ.get('CYCLO_DOCK_DIAGNOSTICS_DIR',
            '/root/ros2_ws/src/cyclo_intelligence/docker/workspace/logs/aruco_docking'))

    def record(self, details, raw=None, rectified=None):
        now = time.monotonic()
        event = deepcopy(details)
        event['monotonic_s'] = now
        self.events.append(event)
        if raw is not None and now-self.last_frame >= .2:
            self.last_frame = now
            self.frames.append((event, raw.copy(), None if rectified is None else rectified.copy()))

    def save_async(self, failure, logger):
        folder = self.root / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
                              + f'_marker_{self.marker_id}_' + uuid.uuid4().hex[:6])
        events, frames = list(self.events), list(self.frames)
        def write():
            try:
                self.write(folder, failure, events, frames)
                logger.info('ArucoDock diagnostics saved: '+str(folder))
            except Exception as exc:
                logger.error('ArucoDock diagnostic write failed: '+str(exc))
        threading.Thread(target=write, daemon=True, name='dock-diagnostics').start()
        return folder

    def write(self, folder, failure, events, frames):
        folder.mkdir(parents=True, exist_ok=False)
        def image(path, pixels):
            if not cv2.imwrite(str(path), pixels): raise OSError('Cannot write '+str(path))
        records = []
        for i, (event, raw, rectified) in enumerate(frames):
            stem = f'{i:03d}'
            image(folder/(stem+'_raw.png'), raw)
            row = dict(event, raw_image=stem+'_raw.png')
            if rectified is not None:
                image(folder/(stem+'_rectified.png'), rectified)
                overlay = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
                import numpy as np
                for corners in event.get('detection', {}).get('detected_corners', []):
                    cv2.polylines(overlay, [np.asarray(corners, dtype=np.int32)], True, (0,255,255), 1)
                reason = event.get('detection', {}).get('reason', 'not_detected')
                cv2.putText(overlay, reason, (5,18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,255), 1)
                image(folder/(stem+'_overlay.png'), overlay)
                row.update(rectified_image=stem+'_rectified.png', overlay_image=stem+'_overlay.png')
            records.append(row)
        report = dict(self.context, failure=failure, frames=records, events=events)
        (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
