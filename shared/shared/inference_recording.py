"""Filesystem contract for reusable inference recordings."""

import json
import os
from pathlib import Path
import re
import time

RECORDING_ROOT = Path('/workspace/rosbag2')


def folder_name(session_id: str) -> str:
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', session_id)
            or '..' in session_id):
        raise ValueError('Invalid inference recording session ID')
    return f'Task_{session_id}_inference_MCAP'


def validate_folder(root: Path, session_id: str, robot_type: str) -> Path:
    folder = Path(root) / folder_name(session_id)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f'Recording folder is missing or is a symlink: {folder}')
    if folder.resolve().parent != Path(root).resolve():
        raise ValueError('Recording folder must be directly under the recording root')
    metadata = []

    def scan_error(error):
        raise error

    for parent, directories, files in os.walk(folder, onerror=scan_error):
        for name in directories + files:
            entry = Path(parent) / name
            if entry.is_symlink():
                raise ValueError('Recording folder must not contain symlinks')
            if name == 'episode_info.json':
                metadata.append(entry)
    for entry in metadata:
        try:
            info = json.loads(entry.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read recording metadata: {entry}') from exc
        if not isinstance(info, dict) or info.get('robot_type') != robot_type:
            raise ValueError(f'Recording robot_type does not match {robot_type!r}: {entry}')
        if info.get('format_version') not in ('robotis_v1', 'robotis_v2'):
            raise ValueError(f'Unsupported recording format: {entry}')
    for episode in folder.iterdir():
        if episode.is_dir() and episode.name.isdigit() and any(episode.iterdir()):
            if not any(episode in entry.parents for entry in metadata):
                raise ValueError(f'Episode has no readable metadata: {episode}')
    return folder


def allocate_folder(root: Path) -> str:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
    suffix = 0
    while True:
        session_id = timestamp if suffix == 0 else f'{timestamp}_{suffix:02d}'
        try:
            (root / folder_name(session_id)).mkdir()
            return session_id
        except FileExistsError:
            suffix += 1
