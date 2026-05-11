#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PipelineMixin — chunk RX + trigger TX.

Extracted from ``control_publisher.py``. Holds the two callbacks that
move ActionChunks through the buffer:

- ``_on_chunk`` — Zenoh subscriber callback for the raw chunk topic.
- ``_send_trigger_locked`` — fires a ``run_inference`` trigger when the
  buffer falls below the refill threshold.

See the S5 design note in ``docs/plans/2026-05-10-cyclo_brain-refactor.md``.
Both methods rely on ``self._config_lock`` / ``self._processor`` /
``self._trigger_pub`` set in ``ControlPublisher.__init__`` and
``configure()``. The ``*_locked`` helper must be called with
``self._config_lock`` held.
"""

from __future__ import annotations

import time

import numpy as np

from zenoh_ros2_sdk import get_logger


logger = get_logger("control_publisher")


class PipelineMixin:
    """Chunk RX + trigger TX for ControlPublisher."""

    def _on_chunk(self, msg) -> None:
        with self._config_lock:
            if not self._configured or self._processor is None:
                return
            try:
                seq_id = int(getattr(msg, "seq_id", 0))
                chunk_size = int(msg.chunk_size)
                action_dim = int(msg.action_dim)
                data = np.asarray(msg.data, dtype=np.float64)
                if data.size != chunk_size * action_dim:
                    logger.warning(
                        f"chunk seq={seq_id} size mismatch: data.size={data.size} "
                        f"!= {chunk_size} * {action_dim}"
                    )
                    return
                chunk = data.reshape(chunk_size, action_dim)
                n_pushed = self._processor.push_chunk(chunk)
                logger.info(
                    f"chunk rx seq={seq_id} T={chunk_size} D={action_dim} → "
                    f"pushed={n_pushed} buffer={self._processor.buffer_size}"
                )
                self._publish_trajectory_preview_locked(chunk)
            except Exception as e:
                logger.error(f"chunk decode failed: {e}", exc_info=True)
            finally:
                self._requesting = False

    def _send_trigger_locked(self) -> None:
        if self._trigger_pub is None:
            return
        self._seq_id += 1
        try:
            self._trigger_pub.publish(data=self._seq_id)
            self._requesting = True
            self._request_sent_at = time.time()
            logger.debug(f"trigger pub seq={self._seq_id}")
        except Exception as e:
            logger.error(f"trigger publish failed: {e}", exc_info=True)
