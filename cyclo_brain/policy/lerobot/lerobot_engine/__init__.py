"""lerobot_engine package - concrete InferenceEngine for LeRobot.

Re-exports ``LeRobotEngine`` + ``create_engine`` so the Engine process
``importlib.import_module("lerobot_engine")`` +
``getattr(mod, "create_engine")()`` keep working after the split.
"""

__all__ = ["LeRobotEngine", "create_engine"]


def __getattr__(name):
    # Catalog/adapter inspection must not initialize Torch or model frameworks.
    if name in __all__:
        from .engine import LeRobotEngine, create_engine
        globals().update(LeRobotEngine=LeRobotEngine, create_engine=create_engine)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
