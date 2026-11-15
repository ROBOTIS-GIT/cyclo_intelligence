import json
import threading

from cyclo_data.services.curation_service import CurationService, _CurationJob
from interfaces.msg import DataOperationStatus
from interfaces.srv import GetDatasetCurationStatus, StartDatasetCuration


class _Executor:
    def __init__(self):
        self.submissions = []

    def submit(self, callback, *args):
        self.submissions.append((callback, args))


def _service_without_ros():
    service = CurationService.__new__(CurationService)
    service._lock = threading.Lock()
    service._jobs = {}
    service._active_job_id = ""
    service._executor = _Executor()
    service._publish = lambda _job: None
    service._persist = lambda _job: None
    return service


def test_analyze_command_creates_async_job():
    service = _service_without_ros()
    request = StartDatasetCuration.Request()
    request.command = StartDatasetCuration.Request.ANALYZE
    request.request_json = json.dumps({
        "schema_version": 1,
        "sources": [{"task_dir": "/workspace/rosbag2/Task_fixture_MCAP"}],
    })
    response = StartDatasetCuration.Response()

    result = service._start_callback(request, response)

    assert result.success is True
    assert result.job_id in service._jobs
    job = service._jobs[result.job_id]
    assert job.status == DataOperationStatus.RUNNING
    assert job.analysis_request["schema_version"] == 1
    assert len(service._executor.submissions) == 1


def test_apply_requires_matching_completed_analysis():
    service = _service_without_ros()
    request = StartDatasetCuration.Request()
    request.command = StartDatasetCuration.Request.APPLY
    request.job_id = "missing"
    request.request_json = "{}"

    result = service._start_callback(request, StartDatasetCuration.Response())

    assert result.success is False
    assert "analysis" in result.message.lower()
    assert service._executor.submissions == []


def test_status_returns_persisted_analysis_payload():
    service = _service_without_ros()
    analysis = {"kind": "analysis", "summary": {"ready": 2, "needs_review": 1}}
    service._jobs["job-1"] = _CurationJob(
        job_id="job-1",
        status=DataOperationStatus.COMPLETED,
        progress=100.0,
        stage="awaiting_confirmation",
        analysis_result=analysis,
    )
    request = GetDatasetCurationStatus.Request()
    request.job_id = "job-1"

    result = service._status_callback(
        request, GetDatasetCurationStatus.Response()
    )

    assert result.success is True
    assert result.status == DataOperationStatus.COMPLETED
    assert json.loads(result.result_json) == analysis
