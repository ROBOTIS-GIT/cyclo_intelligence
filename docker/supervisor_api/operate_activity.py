"""Small, durable Operate activity journal; no camera images or prompts are stored."""
import json
import os
from pathlib import Path
import re
import threading
import time

ROOT = Path(os.getenv('CYCLO_OPERATE_ACTIVITY_DIR', '/workspace/logs/operate'))
_MAP = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')


class OperateActivity:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.Lock()

    def path(self, map_name):
        if not isinstance(map_name, str) or not _MAP.fullmatch(map_name):
            raise ValueError('Invalid map name')
        return self.root / f'{map_name}.jsonl'

    def append(self, map_name, event):
        path = self.path(map_name)
        row = {'time_ms': int(time.time() * 1000), **event}
        line = (json.dumps(row, ensure_ascii=False) + '\n').encode()
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('ab') as stream:
                stream.write(line)
        return row

    def recent(self, map_name, limit=100):
        path = self.path(map_name)
        with self.lock:
            if not path.exists():
                return []
            # Read only the latest rows, even after a long-running session.
            with path.open('rb') as stream:
                stream.seek(0, 2)
                position = stream.tell()
                data = b''
                while position > 0 and data.count(b'\n') <= limit:
                    size = min(position, 65536)
                    position -= size
                    stream.seek(position)
                    data = stream.read(size) + data
            return [json.loads(line) for line in data.splitlines()[-limit:] if line]
