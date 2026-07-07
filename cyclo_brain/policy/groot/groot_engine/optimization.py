"""GR00T optional TensorRT/GPU optimization module."""

from runtime.inference_engine import build_trt_engine


class TensorRTOptimizer:
    """Compatibility wrapper around the runtime DiT TensorRT builder."""

    @staticmethod
    def build_engine(policy, observation, engine_path):
        return build_trt_engine(policy, observation, engine_path)


__all__ = ["TensorRTOptimizer", "build_trt_engine"]
