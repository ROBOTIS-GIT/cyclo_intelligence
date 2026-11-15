# Copyright 2025 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0

"""Asynchronous analyze/review/apply service for dataset curation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Optional
import uuid

from cyclo_data.curation import CurationCancelled, CurationEngine
from cyclo_data.converter.pipeline_worker import (
    Mp4ConversionWorker,
    _validate_lerobot_outputs,
    resolve_lerobot_output_root,
)
from interfaces.msg import DataOperationStatus
from interfaces.srv import GetDatasetCurationStatus, StartDatasetCuration
from shared.io.dataset_lock import DatasetOperationLock
from shared.robot_configs import schema as robot_schema


_TERMINAL = {
    DataOperationStatus.COMPLETED,
    DataOperationStatus.FAILED,
    DataOperationStatus.CANCELLED,
}


@dataclass
class _CurationJob:
    job_id: str
    status: int = DataOperationStatus.IDLE
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    analysis_request: dict[str, Any] = field(default_factory=dict)
    analysis_result: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class CurationService:
    START_SERVICE_NAME = "/data/curation/start"
    STATUS_SERVICE_NAME = "/data/curation/status"

    def __init__(self, node, status_publisher):
        self._node = node
        self._status_pub = status_publisher
        self._engine = CurationEngine()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="curation")
        self._lock = threading.Lock()
        self._jobs: dict[str, _CurationJob] = {}
        self._active_job_id = ""
        self._jobs_root = Path("/workspace/.cyclo_intelligence/curation_jobs")
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()

        self._start_server = node.create_service(
            StartDatasetCuration,
            self.START_SERVICE_NAME,
            self._start_callback,
            callback_group=node.io_callback_group,
        )
        self._status_server = node.create_service(
            GetDatasetCurationStatus,
            self.STATUS_SERVICE_NAME,
            self._status_callback,
            callback_group=node.io_callback_group,
        )
        node.get_logger().info(f"Service advertised: {self.START_SERVICE_NAME}")
        node.get_logger().info(f"Service advertised: {self.STATUS_SERVICE_NAME}")

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status == DataOperationStatus.RUNNING:
                job.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _start_callback(self, request, response):
        try:
            payload = json.loads(request.request_json or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request_json must contain an object")
        except (json.JSONDecodeError, ValueError) as exc:
            response.success = False
            response.job_id = request.job_id
            response.message = f"Invalid curation request: {exc}"
            return response

        if request.command == StartDatasetCuration.Request.CANCEL:
            return self._cancel(request.job_id, response)

        with self._lock:
            active = self._jobs.get(self._active_job_id)
            if active is not None and active.status == DataOperationStatus.RUNNING:
                response.success = False
                response.job_id = active.job_id
                response.message = f"Curation job is already running: {active.job_id}"
                return response

            if request.command == StartDatasetCuration.Request.ANALYZE:
                job_id = uuid.uuid4().hex
                job = _CurationJob(
                    job_id=job_id,
                    status=DataOperationStatus.RUNNING,
                    stage="analyzing",
                    message="Dataset analysis started",
                    analysis_request=payload,
                )
                self._jobs[job_id] = job
                self._active_job_id = job_id
                self._persist(job)
                self._executor.submit(self._run_analysis, job)
            elif request.command == StartDatasetCuration.Request.APPLY:
                job = self._jobs.get(request.job_id)
                if job is None or not job.analysis_result:
                    response.success = False
                    response.job_id = request.job_id
                    response.message = "Completed analysis is required before APPLY"
                    return response
                job.cancel_event = threading.Event()
                job.status = DataOperationStatus.RUNNING
                job.progress = 0.0
                job.stage = "materializing"
                job.message = "Curated copy creation started"
                self._active_job_id = job.job_id
                self._persist(job)
                self._executor.submit(self._run_apply, job, payload)
            else:
                response.success = False
                response.job_id = request.job_id
                response.message = f"Unknown curation command: {request.command}"
                return response

        self._publish(job)
        response.success = True
        response.job_id = job.job_id
        response.message = job.message
        return response

    def _cancel(self, job_id: str, response):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                response.success = False
                response.job_id = job_id
                response.message = "Curation job not found"
                return response
            job.cancel_event.set()
            if job.status != DataOperationStatus.RUNNING:
                job.status = DataOperationStatus.CANCELLED
                job.stage = "cancelled"
                job.message = "Dataset curation was cancelled"
                self._persist(job)
        self._publish(job)
        response.success = True
        response.job_id = job_id
        response.message = "Cancellation requested"
        return response

    def _status_callback(self, request, response):
        with self._lock:
            job_id = request.job_id or self._active_job_id
            job = self._jobs.get(job_id)
            if job is None:
                response.success = False
                response.status = DataOperationStatus.IDLE
                response.progress_percentage = 0.0
                response.stage = ""
                response.message = "Curation job not found"
                response.result_json = ""
                return response
            result = job.result or job.analysis_result
            response.success = True
            response.status = int(job.status)
            response.progress_percentage = float(job.progress)
            response.stage = job.stage
            response.message = job.message
            response.result_json = (
                json.dumps(result, ensure_ascii=False) if result else ""
            )
        return response

    def _run_analysis(self, job: _CurationJob) -> None:
        try:
            with DatasetOperationLock(exclusive=False):
                result = self._engine.analyze(
                    job.analysis_request,
                    job_id=job.job_id,
                    progress=lambda p, s, m: self._update(job, p, s, m),
                    cancel_event=job.cancel_event,
                )
            with self._lock:
                job.analysis_result = result
                job.result = {}
                job.status = DataOperationStatus.COMPLETED
                job.progress = 100.0
                job.stage = "awaiting_confirmation"
                job.message = (
                    f"Analysis completed: {result['summary']['ready']} ready, "
                    f"{result['summary']['needs_review']} need review"
                )
                self._active_job_id = ""
                self._persist(job)
            self._publish(job)
        except CurationCancelled as exc:
            self._finish_error(job, DataOperationStatus.CANCELLED, "cancelled", str(exc))
        except Exception as exc:  # noqa: BLE001
            self._finish_error(job, DataOperationStatus.FAILED, "analysis_failed", str(exc))

    def _run_apply(self, job: _CurationJob, request: dict[str, Any]) -> None:
        try:
            with DatasetOperationLock(exclusive=True):
                materialized = self._engine.apply(
                    job.analysis_result,
                    request,
                    job_id=job.job_id,
                    progress=lambda p, s, m: self._update(job, p, s, m),
                    cancel_event=job.cancel_event,
                )
                with self._lock:
                    job.result = dict(materialized)
                    self._persist(job)

                conversion_request = request.get("conversion") or {}
                if bool(conversion_request.get("enabled", False)):
                    conversion_result = self._run_conversion(
                        job, materialized, conversion_request
                    )
                    materialized["conversion"] = conversion_result

            with self._lock:
                job.result = materialized
                job.status = DataOperationStatus.COMPLETED
                job.progress = 100.0
                job.stage = "completed"
                job.message = "Dataset curation completed successfully"
                self._active_job_id = ""
                self._persist(job)
            self._publish(job)
        except CurationCancelled as exc:
            self._finish_error(job, DataOperationStatus.CANCELLED, "cancelled", str(exc))
        except Exception as exc:  # noqa: BLE001
            self._finish_error(job, DataOperationStatus.FAILED, "apply_failed", str(exc))

    def _run_conversion(
        self,
        job: _CurationJob,
        materialized: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        dataset_name = str(config.get("dataset_name") or "curated_merged").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_name):
            raise ValueError("conversion.dataset_name contains unsupported characters")
        convert_v21 = bool(config.get("convert_v21", True))
        convert_v30 = bool(config.get("convert_v30", False))
        if not convert_v21 and not convert_v30:
            raise ValueError("at least one LeRobot format must be selected")
        output_root = resolve_lerobot_output_root(config.get("output_root", ""))
        for suffix, enabled in (("v21", convert_v21), ("v30", convert_v30)):
            target = output_root / f"{dataset_name}_lerobot_{suffix}"
            if enabled and target.exists():
                raise FileExistsError(f"LeRobot output already exists: {target}")

        source_folders = [item["output_task_dir"] for item in materialized["sources"]]
        included_by_source = {
            str(item["source_task_dir"]): {
                int(index) for index in item.get("episode_indices", [])
            }
            for item in materialized["sources"]
        }
        robot_types = {
            str(row["robot_type"])
            for source in job.analysis_result["sources"]
            for row in source["episodes"]
            if row.get("robot_type")
            and int(row["source_episode_index"]) in included_by_source.get(
                str(source["task_dir"]), set()
            )
        }
        if not robot_types:
            raise RuntimeError("could not determine robot type for LeRobot conversion")
        if len(robot_types) != 1:
            raise ValueError(
                "LeRobot merge requires one robot type; found: "
                + ", ".join(sorted(robot_types))
            )
        robot_type = next(iter(robot_types))
        robot_config_path = str(robot_schema.find_robot_config_path(robot_type))
        merge_parent = self._jobs_root / job.job_id / "conversion_source"
        merge_path = merge_parent / dataset_name
        if merge_parent.exists():
            shutil.rmtree(merge_parent)
        merge_parent.mkdir(parents=True)

        worker = Mp4ConversionWorker()
        try:
            if not worker.start():
                raise RuntimeError("failed to start LeRobot conversion worker")
            request_data = {
                "dataset_path": str(merge_path),
                "robot_type": robot_type,
                "robot_config_path": robot_config_path,
                "source_folders": source_folders,
                "fps": int(config.get("fps") or job.analysis_result["settings"]["fps"]),
                "convert_v21": convert_v21,
                "convert_v30": convert_v30,
                "lerobot_output_root": str(output_root),
                "delete_source_after_success": False,
                "selected_cameras": list(config.get("selected_cameras") or []),
                "camera_rotations": dict(config.get("camera_rotations") or {}),
                "image_resize": config.get("image_resize"),
                "selected_state_topics": list(config.get("selected_state_topics") or []),
                "selected_action_topics": list(config.get("selected_action_topics") or []),
                "selected_joints": list(config.get("selected_joints") or []),
            }
            if not worker.send_request(request_data):
                raise RuntimeError("failed to queue LeRobot conversion")
            while worker.is_busy():
                if job.cancel_event.is_set():
                    raise CurationCancelled("Dataset conversion was cancelled")
                status = worker.check_task_status()
                worker_pct = float((status.get("progress") or {}).get("percentage", 0.0))
                mapped = 75.0 + max(0.0, min(100.0, worker_pct)) * 0.25
                self._update(job, mapped, "converting_lerobot", status.get("message", ""))
                if status.get("status") == "Failed":
                    raise RuntimeError(status.get("message") or "LeRobot conversion failed")
                if status.get("status") == "Success":
                    break
                time.sleep(0.5)

            _validate_lerobot_outputs(
                merge_path,
                convert_v21=convert_v21,
                convert_v30=convert_v30,
                output_root=output_root,
            )
            return {
                "status": "passed",
                "dataset_name": dataset_name,
                "fps": request_data["fps"],
                "v21": str(output_root / f"{dataset_name}_lerobot_v21") if convert_v21 else "",
                "v30": str(output_root / f"{dataset_name}_lerobot_v30") if convert_v30 else "",
                "source_order": source_folders,
            }
        finally:
            worker.stop()
            if merge_parent.exists():
                shutil.rmtree(merge_parent, ignore_errors=True)

    def _update(self, job: _CurationJob, progress: float, stage: str, message: str) -> None:
        with self._lock:
            job.progress = float(progress)
            job.stage = stage
            if message:
                job.message = message
        self._publish(job)

    def _finish_error(self, job: _CurationJob, status: int, stage: str, message: str) -> None:
        with self._lock:
            job.status = status
            job.stage = stage
            job.message = message
            self._active_job_id = ""
            self._persist(job)
        self._node.get_logger().error(f"Curation job {job.job_id}: {message}")
        self._publish(job)

    def _publish(self, job: _CurationJob) -> None:
        msg = DataOperationStatus()
        msg.operation_type = DataOperationStatus.OP_CURATION
        msg.status = int(job.status)
        msg.job_id = job.job_id
        msg.progress_percentage = float(job.progress)
        msg.stage = job.stage
        msg.message = job.message
        self._status_pub.publish(msg)

    def _job_path(self, job_id: str) -> Path:
        return self._jobs_root / job_id / "job.json"

    def _persist(self, job: _CurationJob) -> None:
        path = self._job_path(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "job_id": job.job_id,
            "status": int(job.status),
            "progress": job.progress,
            "stage": job.stage,
            "message": job.message,
            "analysis_request": job.analysis_request,
            "analysis_result": job.analysis_result,
            "result": job.result,
        }, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _load_persisted_jobs(self) -> None:
        if not self._jobs_root.is_dir():
            return
        for path in self._jobs_root.glob("*/job.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                status = int(data.get("status", DataOperationStatus.FAILED))
                if status == DataOperationStatus.RUNNING:
                    status = DataOperationStatus.FAILED
                    data["stage"] = "interrupted"
                    data["message"] = "Curation service restarted during the job"
                job = _CurationJob(
                    job_id=str(data["job_id"]),
                    status=status,
                    progress=float(data.get("progress", 0.0)),
                    stage=str(data.get("stage", "")),
                    message=str(data.get("message", "")),
                    analysis_request=dict(data.get("analysis_request") or {}),
                    analysis_result=dict(data.get("analysis_result") or {}),
                    result=dict(data.get("result") or {}),
                )
                self._jobs[job.job_id] = job
                if int(data.get("status", DataOperationStatus.FAILED)) == (
                    DataOperationStatus.RUNNING
                ):
                    self._persist(job)
            except Exception:  # noqa: BLE001 - ignore stale/corrupt job cache
                continue
