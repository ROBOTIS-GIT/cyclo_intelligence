#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Persistent registry of HuggingFace endpoints and per-endpoint tokens.

Schema on disk::

    {
      "active": "https://huggingface.co",
      "endpoints": {
        "https://huggingface.co": {
          "label": "Hugging Face",
          "token": "hf_xxx",
          "user_id": "kiwoong"
        },
        ...
      }
    }

The file lives under the HuggingFace cache directory (``$HF_HOME`` if set,
otherwise ``~/.cache/huggingface``) as ``hf_endpoints.json``. Inside the
container this resolves to ``/root/.cache/huggingface/hf_endpoints.json``,
which is bind-mounted from ``docker/huggingface/`` on the host.
``.gitignore`` already excludes ``docker/huggingface/`` so the file is never
committed. Set ``CYCLO_HF_ENDPOINT_STORE`` to override the path entirely.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional


_ENV_PATH_OVERRIDE = 'CYCLO_HF_ENDPOINT_STORE'


def _default_path() -> Path:
    hf_home = os.environ.get('HF_HOME')
    base = Path(hf_home) if hf_home else Path.home() / '.cache' / 'huggingface'
    return base / 'hf_endpoints.json'


@dataclass
class HFEndpointEntry:
    endpoint: str
    label: str = ''
    token: str = ''
    user_id: str = ''

    def public_dict(self) -> Dict[str, str]:
        """Dict suitable for sending back to the UI (token redacted)."""
        return {
            'endpoint': self.endpoint,
            'label': self.label,
            'user_id': self.user_id,
        }


def _resolve_path(path: Optional[Path]) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get(_ENV_PATH_OVERRIDE)
    if override:
        return Path(override)
    return _default_path()


class HFEndpointStore:
    """File-backed store of HuggingFace endpoint credentials.

    All public methods load + save under an exclusive flock so concurrent
    callers (e.g. the parent ROS process and the multiprocessing HF worker)
    can update the file safely.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.path = _resolve_path(path)
        self.logger = logger or logging.getLogger('HFEndpointStore')

    # ---------------- locked load / save ----------------

    @contextmanager
    def _locked(self) -> Iterator[Dict]:
        """Yield a mutable in-memory dict; persist on context exit."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open or create the file in r+ mode so we can hold the same fd while
        # we read, mutate, and rewrite.
        if not self.path.exists():
            self.path.write_text(
                json.dumps({'active': '', 'endpoints': {}}, indent=2),
                encoding='utf-8',
            )
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

        with open(self.path, 'r+', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    text = f.read()
                    data = json.loads(text) if text.strip() else {}
                except json.JSONDecodeError as e:
                    self.logger.error(
                        f'Corrupt endpoint store at {self.path}: {e}; resetting'
                    )
                    data = {}
                data.setdefault('active', '')
                data.setdefault('endpoints', {})

                yield data

                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
                f.flush()
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # ---------------- public API ----------------

    def list(self) -> List[HFEndpointEntry]:
        """Return all registered endpoints (token still included; callers
        decide whether to redact)."""
        with self._locked() as data:
            return [
                HFEndpointEntry(
                    endpoint=ep,
                    label=info.get('label', ''),
                    token=info.get('token', ''),
                    user_id=info.get('user_id', ''),
                )
                for ep, info in data['endpoints'].items()
            ]

    def get(self, endpoint: str) -> Optional[HFEndpointEntry]:
        if not endpoint:
            return None
        with self._locked() as data:
            info = data['endpoints'].get(endpoint)
            if info is None:
                return None
            return HFEndpointEntry(
                endpoint=endpoint,
                label=info.get('label', ''),
                token=info.get('token', ''),
                user_id=info.get('user_id', ''),
            )

    def set(
        self,
        endpoint: str,
        label: str,
        token: str,
        user_id: str,
    ) -> HFEndpointEntry:
        if not endpoint:
            raise ValueError('endpoint must be non-empty')
        if not token:
            raise ValueError('token must be non-empty')
        with self._locked() as data:
            data['endpoints'][endpoint] = {
                'label': label or '',
                'token': token,
                'user_id': user_id or '',
            }
            # First registered endpoint becomes the active one by default.
            if not data['active']:
                data['active'] = endpoint
        return HFEndpointEntry(
            endpoint=endpoint, label=label, token=token, user_id=user_id
        )

    def remove(self, endpoint: str) -> bool:
        with self._locked() as data:
            if endpoint not in data['endpoints']:
                return False
            del data['endpoints'][endpoint]
            if data['active'] == endpoint:
                # Fall back to any remaining endpoint, or empty.
                data['active'] = next(iter(data['endpoints']), '')
            return True

    def get_active(self) -> Optional[HFEndpointEntry]:
        with self._locked() as data:
            active = data.get('active', '')
            if not active:
                return None
            info = data['endpoints'].get(active)
            if info is None:
                return None
            return HFEndpointEntry(
                endpoint=active,
                label=info.get('label', ''),
                token=info.get('token', ''),
                user_id=info.get('user_id', ''),
            )

    def set_active(self, endpoint: str) -> bool:
        with self._locked() as data:
            if endpoint and endpoint not in data['endpoints']:
                return False
            data['active'] = endpoint or ''
            return True

    def resolve(self, endpoint: str = '') -> Optional[HFEndpointEntry]:
        """Return ``endpoint`` if registered; otherwise the active endpoint.

        Used by request handlers that accept an optional per-call override.
        """
        if endpoint:
            return self.get(endpoint)
        return self.get_active()


__all__ = ['HFEndpointStore', 'HFEndpointEntry']
