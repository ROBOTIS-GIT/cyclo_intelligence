"""Use a cached Cosmos processor without Transformers' remote metadata probe."""
from contextlib import contextmanager
from functools import wraps
import importlib
import logging
from pathlib import Path

COSMOS_MODEL = "nvidia/Cosmos-Reason2-2B"


def _local_processor_source(model_name, kwargs):
    if str(model_name) != COSMOS_MODEL:
        return model_name, kwargs

    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        # Resolve the requested revision locally; do not change a pinned revision
        # or turn a cache lookup into a download.
        directory = snapshot_download(
            repo_id=COSMOS_MODEL,
            revision=kwargs.get("_commit_hash") or kwargs.get("revision") or "main",
            cache_dir=kwargs.get("cache_dir"),
            local_files_only=True,
        )
        required = ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json")
        missing = [name for name in required if not (Path(directory) / name).is_file()]
        if missing:
            raise LocalEntryNotFoundError("Missing Cosmos processor files: " + ", ".join(missing))
    except LocalEntryNotFoundError as exc:
        if constants.HF_HUB_OFFLINE or kwargs.get("local_files_only"):
            raise RuntimeError(
                "Cosmos processor is not fully cached for the requested revision. "
                "Cache nvidia/Cosmos-Reason2-2B processor files before offline GR00T loading."
            ) from exc
        # Preserve the usual authenticated download path on first online load.
        return model_name, kwargs

    logging.getLogger("groot_inference").info("Using cached Cosmos processor: %s", directory)
    return directory, {**kwargs, "local_files_only": True}


@contextmanager
def cached_cosmos_processor():
    """Scope the GR00T N1.7 processor adapter to a serialized policy load.

    Both its collator and action processor use this factory. Other models and
    N1.6 are unaffected; the upstream factory is restored even if loading fails.
    """
    module_name = "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in ("gr00t.model.gr00t_n1d7", module_name):
            raise
        yield
        return
    original = module.build_processor

    @wraps(original)
    def build_processor(model_name, transformers_loading_kwargs):
        source, kwargs = _local_processor_source(model_name, transformers_loading_kwargs)
        return original(source, kwargs)

    module.build_processor = build_processor
    try:
        yield
    finally:
        module.build_processor = original
