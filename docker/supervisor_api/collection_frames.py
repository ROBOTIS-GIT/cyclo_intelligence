"""Bulk frame removal with stable IDs and workflow-aware dataset locking."""
from contextlib import ExitStack
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints, field_validator

from .collection import atomic, read_json, reasoning_counts, rows, write_json

ACTIVE = {"queued", "training", "running", "cancelling"}
TRAINING_FILES = ("training_manifest.json", "train.jsonl", "val.jsonl",
                  "test.jsonl", "training_regions.jsonl")
FrameId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


class DeleteFramesRequest(BaseModel):
    frame_ids: list[FrameId] = Field(min_length=1, max_length=100000)

    @field_validator("frame_ids")
    @classmethod
    def unique_ids(cls, ids):
        if len(set(ids)) != len(ids):
            raise ValueError("Select each frame only once")
        return ids


def relevant(job, episode):
    if job.get("status") not in ACTIVE:
        return False
    ids = job.get("episode_ids")
    if ids is not None:
        return episode["id"] in ids
    if job.get("episode_id"):
        return episode["id"] == job["episode_id"]
    # Older jobs only recorded the map; unknown ownership must block editing.
    return not job.get("map_name") or job["map_name"] == episode.get("map_name")


class FrameDeletionService:
    def __init__(self, store, sam=None, qwen=None):
        self.store, self.sam, self.qwen = store, sam, qwen

    def delete(self, eid, request):
        # Match workflow -> cache -> store ordering used by training and saves.
        # Holding submission locks through the write prevents a new job racing
        # the busy check and opening files while they are being removed.
        with ExitStack() as stack:
            if self.qwen is not None:
                stack.enter_context(self.qwen.lock)
            if self.sam is not None:
                stack.enter_context(self.sam.lock)
                service = self.sam.service
                if not service.inference_lock.acquire(blocking=False):
                    raise HTTPException(409, "Wait for mask generation before deleting frames")
                stack.callback(service.inference_lock.release)
                stack.enter_context(service.cache_lock)
            stack.enter_context(self.store.lock)
            try:
                episode = self.store.episode(eid)
                if not episode.get("ended_at"):
                    raise HTTPException(409, "Save the episode before deleting frames")
                self._check_jobs(episode)
                result = self._delete(episode, request.frame_ids)
                if self.sam is not None:
                    previews = self.sam.service.previews
                    deleted = set(result["deleted_frame_ids"])
                    for key, value in list(previews.items()):
                        if value.get("episode_id") == eid and value.get("frame_id") in deleted:
                            previews.pop(key)
                return result
            except OSError as exc:
                raise HTTPException(507, "Frame deletion could not finish. Check dataset storage permissions and disk space.") from exc

    def _check_jobs(self, episode):
        if self.qwen is not None and any(relevant(job, episode) for job in self.qwen.jobs.values()):
            raise HTTPException(409, "Wait for this episode's Qwen job before deleting frames")
        if self.sam is None:
            return
        if relevant(self.sam.job, episode):
            raise HTTPException(409, "Wait for this episode's mask generation before deleting frames")
        try:
            state = self.sam.service.request("/adaptation")
            self.sam.training_job = dict(state.get("training", {}))
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            # A stopped SAM service owns no live files. A training request
            # started here is remembered, so an unresponsive engine fails closed.
            state = {"training": getattr(self.sam, "training_job", {})}
        if relevant(state.get("training", {}), episode):
            raise HTTPException(409, "Wait for SAM training before deleting frames")

    def _path(self, eid, relative, category):
        if not isinstance(relative, str) or "\\" in relative:
            raise HTTPException(422, "Invalid frame asset path")
        parts = relative.split("/")
        if len(parts) < 2 or parts[0] != category or any(part in ("", ".", "..") for part in parts):
            raise HTTPException(422, "Invalid frame asset path")
        if PurePosixPath(relative).is_absolute():
            raise HTTPException(422, "Invalid frame asset path")
        path = self.store.path("raw", eid, *parts)
        if path.exists() and not path.is_file():
            raise HTTPException(422, "Frame asset must be a regular file")
        return path

    def _assets(self, eid, samples):
        files, objects = set(), set()
        for sample in samples:
            frame = sample["observation"]
            files.add(self._path(eid, frame["image"]["path"], "images"))
            # Camera storage uses the frame ID; preserve it even for legacy rows
            # whose image metadata points at a shared asset.
            files.add(self._path(eid, f"images/{frame['frame_id']}.jpg", "images"))
            for region in sample.get("regions", []):
                if region.get("mask", {}).get("path"):
                    files.add(self._path(eid, region["mask"]["path"], "masks"))
                oid = region.get("object_id")
                if oid:
                    objects.add(self.store.path("raw", eid, "masks", oid))
        return files, objects

    def _delete(self, episode, requested):
        eid = episode["id"]
        directory = self.store.path("raw", eid)
        samples_path = self.store.path("raw", eid, "samples.jsonl")
        metadata_path = self.store.path("raw", eid, "episode.json")
        samples = rows(samples_path)
        known = {sample["observation"]["frame_id"] for sample in samples}
        selected = set(requested)
        if not selected <= known:
            raise HTTPException(422, "Some selected frames no longer exist. Reload the episode")
        removed = [sample for sample in samples if sample["observation"]["frame_id"] in selected]
        kept = [sample for sample in samples if sample["observation"]["frame_id"] not in selected]
        removed_files, removed_objects = self._assets(eid, removed)
        kept_files, kept_objects = self._assets(eid, kept)
        removable_dirs = set()
        for object_dir in removed_objects - kept_objects:
            if object_dir.exists():
                if not object_dir.is_dir():
                    raise HTTPException(422, "Invalid mask revision directory")
                removable_dirs.add(object_dir)
                for child in object_dir.rglob("*"):
                    # Validate *every* revision and reject symlink subdirectories
                    # before touching metadata or any asset.
                    checked = self.store.path("raw", eid, *child.relative_to(directory).parts)
                    if checked.is_dir():
                        removable_dirs.add(checked)
                    elif checked.is_file():
                        removed_files.add(checked)
                    else:
                        raise HTTPException(422, "Invalid mask revision asset")
        derived = [self.store.path("raw", eid, name) for name in TRAINING_FILES]
        if any(path.exists() and not path.is_file() for path in derived):
            raise HTTPException(422, "Invalid training export path")
        metadata = read_json(metadata_path)
        metadata.pop("training", None)
        for key in ("sam_sample", "qwen_sample"):
            selection = metadata.get(key)
            if not selection:
                continue
            ids = [fid for fid in selection.get("frame_ids", []) if fid in known - selected]
            if ids:
                metadata[key] = {**selection, "frame_ids": ids, "requested_count": len(ids)}
            else:
                metadata.pop(key, None)
        if "frame_count" in metadata:
            metadata["frame_count"] = len(kept)
        # Compute the response before publication: even a later filesystem
        # cleanup error must not leave the client showing deleted frames.
        refreshed = self.store.get(eid)
        for key in ("training", "sam_sample", "qwen_sample"):
            refreshed.pop(key, None)
        refreshed.update(metadata)
        refreshed["observations"] = [row for row in refreshed["observations"] if row["frame_id"] not in selected]
        refreshed["predictions"] = [row for row in refreshed["predictions"] if row["frame_id"] not in selected]
        refreshed["regions"] = [row for row in refreshed["regions"] if row["frame_id"] not in selected]
        refreshed["annotations"] = [row for row in refreshed["annotations"] if not selected.intersection(row["frame_ids"])]
        refreshed.update(reasoning_counts(metadata, kept), frame_count=len(kept))
        original_metadata = metadata_path.read_bytes()
        # Derived files are disposable; fail before publishing raw-data changes
        # if an old training export cannot be invalidated.
        for path in derived:
            path.unlink(missing_ok=True)
        write_json(metadata_path, metadata)
        try:
            self.store.save_samples(eid, kept)  # atomic source-of-truth commit
        except Exception:
            atomic(metadata_path, original_metadata)
            raise
        cleanup_pending = False
        for path in removed_files - kept_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup_pending = True
        for path in sorted(removable_dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                if path.exists() and not any(path.iterdir()):
                    path.rmdir()
            except OSError:
                cleanup_pending = True
        result = {"episode": refreshed,
                  "deleted_frame_ids": [sample["observation"]["frame_id"] for sample in removed]}
        if cleanup_pending:
            result.update(cleanup_pending=True,
                warning="Frames deleted, but some unused image or mask files could not be removed. Check dataset storage permissions.")
        return result



def create_router(service):
    router = APIRouter(prefix="/collection", tags=["collection"])

    @router.post("/episodes/{eid}/frames/delete")
    def delete_frames(eid: str, request: DeleteFramesRequest):
        return service.delete(eid, request)

    return router
