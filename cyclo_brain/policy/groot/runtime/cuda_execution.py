"""Run a complete inference request on a persistent non-default CUDA stream."""

from contextlib import contextmanager

import torch


class InferenceStream:
    def __init__(self, device=0):
        self.stream = torch.cuda.Stream(device=device)

    @contextmanager
    def activate(self):
        # Model weights may have been loaded on the caller's stream. Order
        # those writes before inference without blocking the CPU/device.
        caller = torch.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            # Preserve caller ordering on success and failure, including
            # subsequent tensor use or cleanup. No device-wide synchronize.
            caller.wait_stream(self.stream)
