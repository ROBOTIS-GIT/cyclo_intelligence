import asyncio
import importlib.util
import json
import sys
import threading
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_PATH = Path(__file__).resolve().with_name("app.py")
REPO_ROOT = APP_PATH.parents[2]

docker_stub = types.ModuleType("docker")
docker_errors_stub = types.ModuleType("docker.errors")


class DockerException(Exception):
    pass


class ImageNotFound(DockerException):
    pass


class NotFound(DockerException):
    pass


docker_stub.from_env = lambda: None
docker_errors_stub.DockerException = DockerException
docker_errors_stub.ImageNotFound = ImageNotFound
docker_errors_stub.NotFound = NotFound
sys.modules["docker"] = docker_stub
sys.modules["docker.errors"] = docker_errors_stub

# The supervisor reads the BT robot capability from the shared robot-config
# schema; point it at the checkout when the runtime mounts are absent.
import os as _os  # noqa: E402
_os.environ.setdefault(
    "CYCLO_ROBOT_CONFIGS_DIR",
    str(REPO_ROOT / "shared" / "shared" / "robot_configs"),
)

original_path = list(sys.path)
sys.path = [
    path for path in sys.path
    if Path(path or ".").resolve() != REPO_ROOT
]
try:
    spec = importlib.util.spec_from_file_location("supervisor_api_app", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = app
    spec.loader.exec_module(app)
finally:
    sys.path = original_path

_missing_required_mounts = app._missing_required_mounts
_mount_source_for_destination = app._mount_source_for_destination
_backend_container_image_mismatch = app._backend_container_image_mismatch
_backend_container_stale_reason = app._backend_container_stale_reason
_compose_env = app._compose_env
_load_compose_configuration = app._load_compose_configuration
_host_workspace_dir = app._host_workspace_dir
_require_known_service = app._require_known_service
_validate_bt_robot_type = app._validate_bt_robot_type
_validate_robot_type = app._validate_robot_type
_write_bt_robot_type = app._write_bt_robot_type
_resolve_groot_trt_paths = app._resolve_groot_trt_paths
_trt_status = app._trt_status
_BACKENDS = app._BACKENDS
_USER_SERVICES = app._USER_SERVICES
navigation = sys.modules["supervisor_api.navigation"]
navigation_grid_cache = sys.modules["supervisor_api.navigation_grid_cache"]
navigation_spots = sys.modules["supervisor_api.navigation_spots"]
navigation_missions = sys.modules["supervisor_api.navigation_missions"]
bt_support = sys.modules["supervisor_api.bt_support"]
bt_trees = sys.modules["supervisor_api.bt_trees"]
_GROOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["groot"]
_LEROBOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["lerobot"]


@pytest.fixture(autouse=True)
def isolate_navigation_ros(monkeypatch):
    # API unit tests must not start a native ROS executor in the pytest process.
    monkeypatch.setattr(app._navigation_module, "ensure_ros_grid_subscriber_started", lambda: None)


@pytest.fixture
def trial_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(app, "_TRIAL_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(app, "_TRIAL_MODEL_ROOT", tmp_path / "model")
    with TestClient(app.app) as client:
        yield client


def trial_model(relative):
    path = app._TRIAL_MODEL_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def test_worker_build_uses_mounted_source_not_host_runtime_project(tmp_path, monkeypatch):
    source = tmp_path / "mounted-source"
    (source / "docker").mkdir(parents=True)
    (source / "docker/container.sh").write_text("#!/bin/bash\n")
    (source / ".git").mkdir()
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(source))
    monkeypatch.setattr(app, "_host_project_dir", lambda: "/host/only/cyclo/docker")
    assert app._worker_build_command("lerobot") == [
        "bash", str(source / "docker/container.sh"), "build-policy", "lerobot",
    ]
    with pytest.raises(app.HTTPException) as exc:
        app._worker_build_command("cyclo_intelligence")
    assert exc.value.status_code == 404


def test_worker_build_requires_source_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(tmp_path))
    with pytest.raises(app.HTTPException) as exc:
        app._worker_build_command("lerobot")
    assert exc.value.status_code == 409


def test_worker_build_endpoint_holds_lease_until_completion(monkeypatch):
    from supervisor_api.worker_build import WorkerBuilds
    import time
    builds = WorkerBuilds()
    events = []
    monkeypatch.setattr(app, "_WORKER_BUILDS", builds)
    monkeypatch.setattr(app, "_worker_build_command", lambda name: [sys.executable, "-c", "print('built')"])
    monkeypatch.setattr(app, "_begin_worker_mutation", lambda name: events.append("begin") or "token")
    monkeypatch.setattr(app, "_end_worker_mutation", lambda name, token: events.append((name, token)))
    monkeypatch.setattr(app, "_docker_client", lambda: SimpleNamespace(images=SimpleNamespace(get=lambda tag: events.append(tag))))
    try:
        result = asyncio.run(app.backend_build("lerobot"))
        assert result["state"] == "building"
        deadline = time.monotonic() + 5
        while builds.status("lerobot")["state"] == "building" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert builds.status("lerobot")["state"] == "succeeded"
        assert events == ["begin", app._BACKENDS["lerobot"]["image"], ("lerobot", "token")]
    finally:
        builds.close()


def test_worker_build_launch_rejection_releases_lease(monkeypatch):
    events = []
    def busy(*args):
        raise RuntimeError("already building")
    monkeypatch.setattr(app, "_WORKER_BUILDS", SimpleNamespace(start=busy))
    monkeypatch.setattr(app, "_worker_build_command", lambda name: ["bash", "builder"])
    monkeypatch.setattr(app, "_begin_worker_mutation", lambda name: "token")
    monkeypatch.setattr(app, "_end_worker_mutation", lambda name, token: events.append((name, token)))
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.backend_build("groot"))
    assert exc.value.status_code == 409
    assert events == [("groot", "token")]


def test_worker_build_rejected_by_runtime_never_starts(monkeypatch):
    def denied(name):
        raise app.HTTPException(409, "Clear the loaded model first")
    def unexpected(*args):
        pytest.fail("a build must not start without the maintenance reservation")
    monkeypatch.setattr(app, "_worker_build_command", lambda name: ["bash", "builder"])
    monkeypatch.setattr(app, "_begin_worker_mutation", denied)
    monkeypatch.setattr(app, "_WORKER_BUILDS", SimpleNamespace(start=unexpected))
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.backend_build("lerobot"))
    assert exc.value.status_code == 409


def test_worker_pull_is_retired_without_contacting_docker(monkeypatch):
    def unexpected():
        pytest.fail("Worker pull must not contact Docker")
    monkeypatch.setattr(app, "_docker_client", unexpected)
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.backend_pull("lerobot"))
    assert exc.value.status_code == 410


def test_cancelled_build_request_transfers_acquired_lease(monkeypatch):
    import threading
    entered, proceed, released = threading.Event(), threading.Event(), threading.Event()
    builds = app.WorkerBuilds()
    def acquire(name):
        entered.set()
        assert proceed.wait(5)
        return "token"
    monkeypatch.setattr(app, "_WORKER_BUILDS", builds)
    monkeypatch.setattr(app, "_worker_build_command", lambda name: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(app, "_begin_worker_mutation", acquire)
    monkeypatch.setattr(app, "_end_worker_mutation", lambda name, token: released.set())
    monkeypatch.setattr(app, "_docker_client", lambda: SimpleNamespace(images=SimpleNamespace(get=lambda tag: None)))

    async def cancel():
        task = asyncio.create_task(app.backend_build("lerobot"))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        proceed.set()
        assert await asyncio.to_thread(released.wait, 5)
    try:
        asyncio.run(cancel())
        assert builds.status("lerobot")["state"] == "succeeded"
    finally:
        proceed.set()
        builds.close()


def test_trial_results_crud_and_persistent_model_isolation(trial_client, tmp_path):
    client = trial_client
    model = trial_model("run/checkpoints/100000")
    empty = client.get("/try-results", params={"model": model})
    assert empty.status_code == 200 and empty.json()["rows"] == []
    assert empty.headers["cache-control"] == "no-store"
    assert not app._TRIAL_RESULTS_DIR.exists()
    for result in ["success", "fail", "success"]:
        response = client.post("/try-results", json={"model": model, "result": result})
        assert response.status_code == 200
    original = response.json()
    assert Path(original["file"]) == Path(model) / "evaluation_manifest.json"
    document = json.loads(Path(original["file"]).read_text())
    assert document["schema_version"] == 1
    assert set(document) == {"schema_version", "model", "context", "trials"}
    assert document["model"] == {}
    assert document["context"] == {"mode": "real_robot"}
    assert [trial["trial_id"] for trial in document["trials"]] == [1, 2, 3]
    assert all(
        datetime.fromisoformat(trial["recorded_at"]).utcoffset() == timedelta(hours=9)
        for trial in document["trials"]
    )
    edited = client.patch("/try-results", json={
        "model": model, "try": 2, "result": "success", "revision": original["revision"],
    })
    assert edited.status_code == 200
    assert edited.json()["rows"][1]["result"] == "success"
    assert edited.json()["rows"][1]["created_at"] == original["rows"][1]["created_at"]
    assert datetime.fromisoformat(edited.json()["rows"][1]["updated_at"]).utcoffset() == timedelta(hours=9)
    assert [trial["trial_id"] for trial in json.loads(Path(original["file"]).read_text())["trials"]] == [1, 2, 3]
    other_model = trial_model("other")
    other = client.post("/try-results", json={"model": other_model, "result": "fail"}).json()
    assert other["file"] != original["file"]
    deleted = client.request("DELETE", "/try-results", json={
        "model": model, "tries": [1, 3], "revision": edited.json()["revision"],
    })
    assert deleted.status_code == 200
    assert [row["try"] for row in deleted.json()["rows"]] == ["1"]
    assert deleted.json()["rows"][0]["created_at"] == original["rows"][1]["created_at"]
    assert [trial["trial_id"] for trial in json.loads(Path(original["file"]).read_text())["trials"]] == [1]
    added = client.post("/try-results", json={"model": model + "/", "result": "fail"}).json()
    assert added["file"] == original["file"]
    assert [row["try"] for row in added["rows"]] == ["1", "2"]
    assert client.get("/try-results", params={"model": model}).json() == added
    assert client.get("/try-results", params={"model": other_model}).json() == other
    exported = client.get("/try-results", params={"model": model, "download": True})
    assert exported.status_code == 200 and "text/csv" in exported.headers["content-type"]
    assert "1,success" in exported.text and "2,fail" in exported.text
    assert len(list(app._TRIAL_MODEL_ROOT.rglob("evaluation_manifest.json"))) == 2
    cleared = client.request("DELETE", "/try-results", json={
        "model": model, "tries": [1, 2], "revision": added["revision"],
    }).json()
    assert cleared["rows"] == [] and cleared["file"] == original["file"]
    fresh = client.post("/try-results", json={"model": model, "result": "success"}).json()
    assert fresh["rows"][0]["try"] == "1" and fresh["file"] == original["file"]
    assert not app._TRIAL_RESULTS_DIR.exists()


def test_trial_edits_and_deletes_reject_stale_or_missing_revision(trial_client):
    client = trial_client
    model = trial_model("a")
    stale = client.post("/try-results", json={"model": model, "result": "success"}).json()
    current = client.post("/try-results", json={"model": model, "result": "fail"}).json()
    for revision in [None, stale["revision"]]:
        assert client.patch("/try-results", json={
            "model": model, "try": 1, "result": "fail", "revision": revision,
        }).status_code == 409
        assert client.request("DELETE", "/try-results", json={
            "model": model, "tries": [1], "revision": revision,
        }).status_code == 409
    deleted = client.request("DELETE", "/try-results", json={
        "model": model, "tries": [1], "revision": current["revision"],
    }).json()
    # Old try 1 now refers to a different row; never apply an old edit to it.
    assert client.patch("/try-results", json={
        "model": model, "try": 1, "result": "success", "revision": current["revision"],
    }).status_code == 409
    assert client.get("/try-results", params={"model": model}).json() == deleted


def test_trial_results_invalid_requests_do_not_write(trial_client, tmp_path):
    client = trial_client
    model = trial_model("a")
    assert client.get("/try-results", params={"model": " / "}).status_code == 400
    assert client.post("/try-results", json={"model": model, "result": "invalid"}).status_code == 422
    assert client.post("/try-results", json={"model": model}).status_code == 400
    assert client.patch("/try-results", json={"model": model, "try": 1, "result": "fail"}).status_code == 404
    assert client.request("DELETE", "/try-results", json={"model": model}).status_code == 400
    assert client.request("DELETE", "/try-results", json={"model": model, "tries": [0]}).status_code == 400
    assert not list(Path(model).iterdir())


def test_trial_concurrent_appends_do_not_lose_results(trial_client):
    from concurrent.futures import ThreadPoolExecutor
    model = trial_model("a")
    def append(_):
        return app.add_trial_result(app.TrialResultRequest(model=model, result="success"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(append, range(24)))
    history = trial_client.get("/try-results", params={"model": model}).json()
    assert [row["try"] for row in history["rows"]] == [str(n) for n in range(1, 25)]
    assert len({result["file"] for result in results}) == 1


def test_trial_write_failure_preserves_existing_json(trial_client, monkeypatch):
    model = trial_model("a")
    original = trial_client.post("/try-results", json={"model": model, "result": "success"}).json()
    def fail_write(_):
        raise OSError("disk full")
    with monkeypatch.context() as context:
        context.setattr(app.os, "fsync", fail_write)
        with pytest.raises(OSError, match="disk full"):
            app.add_trial_result(app.TrialResultRequest(model=model, result="fail"))
    assert trial_client.get("/try-results", params={"model": model}).json() == original


def test_trial_files_and_legacy_records_remain_portable(trial_client, tmp_path):
    import hashlib
    import re
    model = trial_model("legacy-act")
    app._TRIAL_RESULTS_DIR.mkdir()
    legacy = app._TRIAL_RESULTS_DIR / (re.sub(r"[^a-zA-Z0-9_.-]", "_", model)[-90:] + "_" + hashlib.sha256(model.encode()).hexdigest()[:16] + ".csv")
    row = {"try": "1", "result": "success", "created_at": "2026-09-18T01:02:03+00:00",
           "updated_at": "2026-09-18T01:02:03+00:00", "model_path": model}
    app._write_trials(legacy, [row])
    migrated = trial_client.get("/try-results", params={"model": model}).json()
    assert Path(migrated["file"]) == Path(model) / "evaluation_manifest.json"
    assert migrated["rows"] == [row] and not legacy.exists()
    preserved = app._TRIAL_RESULTS_DIR / json.loads((app._TRIAL_RESULTS_DIR / "files.json").read_text())[model]
    assert preserved.exists()
    assert trial_client.get("/try-results", params={"model": model}).json() == migrated
    converted = trial_client.post("/try-results", json={"model": model, "result": "fail"}).json()
    assert Path(converted["file"]) == Path(model) / "evaluation_manifest.json"
    assert len(converted["rows"]) == 2
    assert [trial["trial_id"] for trial in json.loads(Path(converted["file"]).read_text())["trials"]] == [1, 2]
    assert preserved.exists()
    assert trial_client.get("/try-results", params={"model": model}).json() == converted
    # A request cannot write into an arbitrary directory outside the model root.
    assert trial_client.post("/try-results", json={
        "model": "../../outside", "result": "fail",
    }).status_code == 400


def test_trial_json_uses_manifest_when_available_and_omits_missing_fields(trial_client):
    model = Path(trial_model("lerobot/act_1k_000024"))
    (model / "training_manifest.json").write_text(json.dumps({
        "model_id": "lerobot-act", "run_id": "1a3a129b-9687-452e-a941-b63ef82c04a4",
        "source_artifact": "checkpoints/001000/pretrained_model",
        "dataset": {"dataset_id": "Dongkkka/Task_000004_Peanut_Pick_Place_lerobot_Intern"},
    }))
    body = {"model": str(model), "robot_type": "ffw_sg2_rev1", "result": "success"}
    response = trial_client.post("/try-results", json=body)
    assert response.status_code == 200
    saved = json.loads(Path(response.json()["file"]).read_text())
    assert saved["model"] == {
        "model_id": "lerobot-act",
        "run_id": "1a3a129b-9687-452e-a941-b63ef82c04a4",
        "source_artifact": "checkpoints/001000/pretrained_model",
    }
    assert saved["context"] == {
        "mode": "real_robot", "robot_type": "ffw_sg2_rev1",
        "dataset_repo_id": "Dongkkka/Task_000004_Peanut_Pick_Place_lerobot_Intern",
    }
    assert len(saved["trials"]) == 1
    assert saved["trials"][0]["result"] == "success"
    assert saved["trials"][0]["trial_id"] == 1
    assert saved["trials"][0]["recorded_at"]
    assert set(model.iterdir()) == {model / "training_manifest.json", model / "evaluation_manifest.json"}
    (model / "training_manifest.json").unlink()
    other = Path(trial_model("lerobot/no_manifest"))
    missing = trial_client.post("/try-results", json={
        "model": str(other), "robot_type": "ffw_sg2_rev1", "result": "fail",
    })
    assert missing.status_code == 200
    saved = json.loads(Path(missing.json()["file"]).read_text())
    assert saved["model"] == {}
    assert saved["context"] == {"mode": "real_robot", "robot_type": "ffw_sg2_rev1"}
    assert list(other.iterdir()) == [other / "evaluation_manifest.json"]


def test_trial_json_rejects_a_different_robot_without_changing_history(trial_client):
    model = trial_model("lerobot/act")
    first = trial_client.post("/try-results", json={
        "model": model, "robot_type": "ffw_sg2_rev1", "result": "success",
    }).json()
    original = Path(first["file"]).read_text()
    second = trial_client.post("/try-results", json={
        "model": model, "robot_type": "f2", "result": "fail",
    })
    assert second.status_code == 409
    assert Path(first["file"]).read_text() == original
    assert trial_client.get("/try-results", params={
        "model": model, "robot_type": "ffw_sg2_rev1",
    }).json()["rows"] == first["rows"]
    assert trial_client.get("/try-results", params={
        "model": model, "robot_type": "f2",
    }).status_code == 409


def test_old_external_json_is_imported_without_losing_trials(trial_client):
    model = trial_model("lerobot/act")
    app._TRIAL_RESULTS_DIR.mkdir()
    old_path = app._old_evaluation_file(model, "f2")
    old = app._new_evaluation(model, "f2")
    old["trials"].append({
        "trial_id": "original-trial", "result": "success", "recorded_at": "2026-10-02T05:00:00Z",
    })
    app._write_evaluation(old_path, old)
    imported = trial_client.get("/try-results", params={"model": model, "robot_type": "f2"}).json()
    assert imported["rows"][0]["result"] == "success"
    trial_client.post("/try-results", json={"model": model, "robot_type": "f2", "result": "fail"})
    saved = json.loads((Path(model) / "evaluation_manifest.json").read_text())
    assert "evaluation_id" not in saved
    assert [trial["trial_id"] for trial in saved["trials"]] == [1, 2]
    assert "local_path" not in saved["model"]
    assert json.loads(old_path.read_text()) == old


def test_existing_single_evaluation_wrapper_is_migrated_without_losing_trials(trial_client):
    model = trial_model("lerobot/act")
    old = app._new_evaluation(model, "ffw_sg2_rev1")
    old["model"]["local_path"] = model
    old["trials"] = [{"trial_id": "old-uuid", "result": "success", "recorded_at": "2026-10-02T05:00:00Z"}]
    path = Path(model) / "evaluation_manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "evaluations": [old]}))
    assert trial_client.get("/try-results", params={"model": model, "robot_type": "ffw_sg2_rev1"}).status_code == 200
    trial_client.post("/try-results", json={"model": model, "robot_type": "ffw_sg2_rev1", "result": "fail"})
    saved = json.loads(path.read_text())
    assert "evaluation_id" not in saved
    assert [trial["trial_id"] for trial in saved["trials"]] == [1, 2]
    assert saved["trials"][0]["recorded_at"] == "2026-10-02T05:00:00Z"


def test_multiple_evaluation_wrapper_requires_manual_separation(trial_client):
    model = trial_model("lerobot/act")
    path = Path(model) / "evaluation_manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "evaluations": [
        app._new_evaluation(model, "ffw_sg2_rev1"), app._new_evaluation(model, "f2"),
    ]}))
    before = path.read_text()
    assert trial_client.post("/try-results", json={
        "model": model, "robot_type": "ffw_sg2_rev1", "result": "fail",
    }).status_code == 409
    assert path.read_text() == before


def test_trial_write_requires_a_model_directory(trial_client, tmp_path):
    missing = app._TRIAL_MODEL_ROOT / "does-not-exist"
    assert trial_client.post("/try-results", json={
        "model": str(missing), "result": "success",
    }).status_code == 404
    outside = tmp_path / "outside"
    outside.mkdir()
    assert trial_client.post("/try-results", json={
        "model": str(outside), "result": "success",
    }).status_code == 400
    assert not list(outside.iterdir())
    app._TRIAL_MODEL_ROOT.mkdir()
    link = app._TRIAL_MODEL_ROOT / "escaped-link"
    link.symlink_to(outside, target_is_directory=True)
    assert trial_client.post("/try-results", json={
        "model": str(link), "result": "success",
    }).status_code == 400
    assert not list(outside.iterdir())


def test_legacy_rows_are_not_assigned_to_a_new_robot(trial_client, tmp_path):
    model = trial_model("old")
    legacy = app._trial_file(model, create=True)
    app._write_trials(legacy, [{
        "try": "1", "result": "success", "created_at": "2026-10-02T05:00:00Z",
        "updated_at": "2026-10-02T05:00:00Z", "model_path": model,
    }])
    response = trial_client.post("/try-results", json={
        "model": model, "robot_type": "ffw_sg2_rev1", "result": "fail",
    }).json()
    saved = json.loads(Path(response["file"]).read_text())
    assert len(saved["trials"]) == 1
    assert saved["context"]["robot_type"] == "ffw_sg2_rev1"
    assert trial_client.get("/try-results", params={"model": model}).status_code == 409
    assert legacy.exists()


def test_trial_json_rejects_symlink_without_touching_target(trial_client, tmp_path):
    model = trial_model("a")
    target = tmp_path / "outside.json"
    target.write_text('{"private": true}')
    app._evaluation_file(model).symlink_to(target)
    response = trial_client.get("/try-results", params={"model": model})
    assert response.status_code == 500
    response = trial_client.post("/try-results", json={"model": model, "result": "success"})
    assert response.status_code == 500
    assert target.read_text() == '{"private": true}'


def test_trial_json_does_not_overwrite_invalid_history(trial_client):
    model = trial_model("a")
    path = app._evaluation_file(model)
    path.write_text("{not json")
    assert trial_client.get("/try-results", params={"model": model}).status_code == 500
    assert trial_client.post("/try-results", json={"model": model, "result": "success"}).status_code == 500
    assert path.read_text() == "{not json"


def test_ten_trials_store_five_successes_and_five_failures(trial_client):
    model = trial_model("lerobot/act_1k_000024")
    files = set()
    for result in ["success", "fail"] * 5:
        response = trial_client.post("/try-results", json={
            "model": model, "robot_type": "ffw_sg2_rev1", "result": result,
        }).json()
        files.add(response["file"])
    assert len(files) == 1
    document = json.loads(Path(response["file"]).read_text())
    assert "evaluation_id" not in document and "uploaded" not in document
    trials = document["trials"]
    assert len(trials) == 10
    assert sum(trial["result"] == "success" for trial in trials) == 5
    assert sum(trial["result"] == "fail" for trial in trials) == 5
    assert [trial["trial_id"] for trial in trials] == list(range(1, 11))
    assert "success_rate" not in document
    assert "task_id" not in document["context"]


def test_existing_evaluation_drops_retired_fields_on_write(trial_client):
    model = trial_model("legacy")
    path = Path(model) / "evaluation_manifest.json"
    document = app._new_evaluation(model, "")
    document["evaluation_id"] = "old-id"
    document["uploaded"] = True
    path.write_text(json.dumps(document))
    assert trial_client.get("/try-results", params={"model": model}).status_code == 200
    trial_client.post("/try-results", json={"model": model, "result": "success"})
    saved = json.loads(path.read_text())
    assert set(saved) == {"schema_version", "model", "context", "trials"}
    assert saved["trials"][0]["result"] == "success"

def test_navigation_grid_cache_starts_with_supervisor_lifespan(monkeypatch):
    started = []
    monkeypatch.setattr(
        app._navigation_module,
        "ensure_ros_grid_subscriber_started",
        lambda: started.append(True),
    )

    async def exercise_lifespan():
        async with app.app.router.lifespan_context(app.app):
            assert started == [True]

    asyncio.run(exercise_lifespan())


def test_supervisor_lifespan_closes_builder_when_navigation_start_fails(monkeypatch):
    closed = []
    monkeypatch.setattr(app, "_WORKER_BUILDS", app._WORKER_BUILDS)
    monkeypatch.setattr(app, "WorkerBuilds", lambda: SimpleNamespace(close=lambda: closed.append(True)))

    def fail():
        raise RuntimeError("navigation startup failed")

    monkeypatch.setattr(app._navigation_module, "ensure_ros_grid_subscriber_started", fail)

    async def exercise():
        with pytest.raises(RuntimeError, match="navigation startup failed"):
            async with app.app.router.lifespan_context(app.app):
                pytest.fail("startup must not succeed")

    asyncio.run(exercise())
    assert closed == [True]


@pytest.mark.parametrize("name", sorted(app._BACKENDS))
def test_worker_start_missing_image_never_provisions_and_releases_lease(monkeypatch, name):
    events = []

    def missing_container(_name):
        raise app.NotFound("not created")

    async def forbidden(*args, **kwargs):
        pytest.fail("missing images must not cause compose build, pull or start")

    monkeypatch.setattr(app, "_docker_client", lambda: SimpleNamespace(
        containers=SimpleNamespace(get=missing_container)))
    monkeypatch.setattr(app, "_local_backend_image", lambda *_args: None)
    monkeypatch.setattr(app, "_begin_worker_mutation", lambda runtime: events.append(("begin", runtime)) or "lease")
    monkeypatch.setattr(app, "_end_worker_mutation", lambda runtime, token: events.append(("end", runtime, token)))
    monkeypatch.setattr(app, "_run", forbidden)
    with pytest.raises(app.HTTPException) as failure:
        asyncio.run(app.backend_start(name))
    assert failure.value.status_code == 409
    assert f"/backends/{name}/build" in failure.value.detail
    assert events == [("begin", name), ("end", name, "lease")]


def test_backend_lifecycle_rejects_target_overrides():
    with pytest.raises(app.HTTPException) as failure:
        asyncio.run(app._ensure_backend_running(
            "lerobot", {**app._BACKENDS["lerobot"], "service": "cyclo_intelligence"},
            restart_running=False,
        ))
    assert failure.value.status_code == 400


def test_navigation_parses_binary_pgm():
    data = b"P5\n# map\n2 2\n255\n" + bytes([0, 127, 254, 255])

    assert navigation._parse_pgm(data) == (
        2,
        2,
        255,
        [0, 127, 254, 255],
    )


def test_navigation_parses_map_yaml_metadata():
    metadata = navigation._parse_map_yaml_metadata(
        b"image: factory.pgm\nresolution: 0.05\norigin: [-1.2, 2.4, 1.570796]\n"
    )

    assert metadata["resolution"] == 0.05
    assert metadata["origin"]["position"] == {"x": -1.2, "y": 2.4, "z": 0.0}
    assert metadata["origin"]["orientation"]["z"] != 0


def test_navigation_get_pgm_includes_yaml_metadata(monkeypatch):
    files = {
        navigation.MAPS_DIR / "factory.pgm": b"P5\n2 2\n255\n" + bytes([0, 127, 254, 255]),
        navigation.MAPS_DIR / "factory.yaml": b"resolution: 0.05\norigin: [-1.0, -2.0, 0.0]\n",
    }

    monkeypatch.setattr(navigation, "_read_container_file", lambda path: files[path])

    result = navigation.get_pgm("factory.pgm")

    assert result["resolution"] == 0.05
    assert result["origin"]["position"] == {"x": -1.0, "y": -2.0, "z": 0.0}
    assert result["width"] == 2
    assert result["height"] == 2


def test_navigation_map_annotations_sidecar(monkeypatch):
    files = {}

    def fake_read(path):
        if path not in files:
            raise FileNotFoundError(str(path))
        return files[path]

    def fake_write(path, content):
        files[path] = content

    monkeypatch.setattr(navigation, "_read_container_file", fake_read)
    monkeypatch.setattr(navigation, "_write_container_file", fake_write)

    empty = navigation.get_map_annotations("factory.pgm")

    assert empty["annotations"] == []

    saved = navigation.save_map_annotations(
        navigation.MapAnnotationsSaveRequest(
            path="factory.pgm",
            annotations=[
                navigation.MapAnnotation(
                    id="dock",
                    label="Dock",
                    color="#5B8266",
                    pose=navigation.MapAnnotationPose(x=1.2, y=-0.4, yaw=0.0),
                    region=navigation.MapAnnotationRegion(
                        seed_cell=navigation.MapAnnotationSeedCell(x=3, y=4),
                        bounds=navigation.MapAnnotationBounds(
                            x_min=2,
                            y_min=3,
                            x_max=6,
                            y_max=8,
                        ),
                        cells=[
                            navigation.MapAnnotationSeedCell(x=3, y=4),
                            navigation.MapAnnotationSeedCell(x=4, y=4),
                        ],
                        cell_count=12,
                        width=20,
                        height=10,
                    ),
                )
            ],
        )
    )

    assert saved["annotations_path"] == "factory.annotations.json"
    assert navigation.MAPS_DIR / "factory.annotations.json" in files

    loaded = navigation.get_map_annotations("factory.pgm")

    assert loaded["annotations"][0]["label"] == "Dock"
    assert loaded["annotations"][0]["pose"]["x"] == 1.2
    assert loaded["annotations"][0]["region"]["seed_cell"] == {"x": 3, "y": 4}
    assert loaded["annotations"][0]["region"]["bounds"] == {
        "x_min": 2,
        "y_min": 3,
        "x_max": 6,
        "y_max": 8,
    }
    assert loaded["annotations"][0]["region"]["cells"] == [
        {"x": 3, "y": 4},
        {"x": 4, "y": 4},
    ]


def test_navigation_rejects_map_path_escape():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        navigation._resolve_pgm_path("../../outside.pgm")


def test_navigation_validates_map_name():
    import pytest
    from fastapi import HTTPException

    assert navigation._validate_map_name("factory-1") == "factory-1"
    with pytest.raises(HTTPException):
        navigation._validate_map_name("factory; reboot")


def test_navigation_delete_pgm_removes_sidecars_and_missions(monkeypatch):
    executed = []

    def fake_exec(command, **kwargs):
        executed.append(command)
        return 0, ""

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setattr(navigation, "remove_map_missions", lambda map_name: 2)

    result = navigation.delete_pgm("factory.pgm")

    assert result == {"path": "factory.pgm", "removed_missions": 2, "deleted": True}
    rm_command = executed[-1]
    assert rm_command[:2] == ["rm", "-f"]
    assert str(navigation.MAPS_DIR / "factory.pgm") in rm_command
    assert str(navigation.MAPS_DIR / "factory.yaml") in rm_command
    assert str(navigation.MAPS_DIR / "factory.annotations.json") in rm_command


def test_navigation_delete_pgm_missing_file(monkeypatch):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation, "_exec", lambda command, **kwargs: (1, ""))
    monkeypatch.setattr(
        navigation,
        "remove_map_missions",
        lambda map_name: pytest.fail("missions must survive a missing map"),
    )

    with pytest.raises(HTTPException) as excinfo:
        navigation.delete_pgm("ghost.pgm")
    assert excinfo.value.status_code == 404


def test_missions_remove_map_missions(tmp_path, monkeypatch):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    mission_root = tmp_path / "missions" / "factory"
    (mission_root / "default").mkdir(parents=True)
    (mission_root / "patrol").mkdir()
    (mission_root / ".staging").mkdir()

    assert navigation_missions.remove_map_missions("factory") == 2
    assert not mission_root.exists()
    # A second call (or a map without missions) is a quiet no-op.
    assert navigation_missions.remove_map_missions("factory") == 0


def test_navigation_save_map_waits_for_artifacts(monkeypatch):
    calls = []
    signatures = [
        {"yaml": None, "pgm": None},
        {"yaml": "yaml:1", "pgm": "pgm:1"},
    ]

    monkeypatch.setattr(
        navigation,
        "_map_artifact_signatures",
        lambda map_name: signatures.pop(0),
    )
    monkeypatch.setattr(
        navigation,
        "_save_map_with_cli",
        lambda map_name: calls.append(map_name) or "map saver complete",
    )

    result = navigation.save_map(
        navigation.MapSaveRequest(map_name="factory")
    )

    assert calls == ["factory"]
    assert result.ok
    assert result.message == "Saved map 'factory' as factory.yaml and factory.pgm"
    assert "map_saver" not in result.message


def test_navigation_save_map_errors_when_artifacts_missing(monkeypatch):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation, "SAVE_MAP_WAIT_SECONDS", 0)
    monkeypatch.setattr(
        navigation,
        "_map_artifact_signatures",
        lambda map_name: {"yaml": None, "pgm": None},
    )
    monkeypatch.setattr(navigation, "_save_map_with_cli", lambda map_name: "")

    with pytest.raises(HTTPException) as exc:
        navigation.save_map(navigation.MapSaveRequest(map_name="factory"))

    assert exc.value.status_code == 503
    assert "factory.yaml" in exc.value.detail


def test_navigation_save_map_cli_does_not_forward_launch_args(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "saved"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation._save_map_with_cli("test_2")

    assert result == "saved"
    assert captured["command"][:4] == [
        "bash", "--noprofile", "--norc", "-c"
    ]
    command_text = captured["command"][-1]
    assert "map_saver_cli" in command_text
    assert "/root/ros2_ws/src/ai_worker/ffw_navigation/maps/test_2" in command_text
    assert "map_name:=" not in command_text
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_start_clears_stale_runtime_before_up(monkeypatch):
    events = []

    monkeypatch.setattr(
        navigation,
        "_request_s6_service_down",
        lambda service: events.append(("request_down", service))
        or f"{service} down requested",
    )
    monkeypatch.setattr(
        navigation,
        "_clear_navigation_runtime_files",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        navigation,
        "_force_stop_navigation_processes",
        lambda: events.append("force") or "",
    )
    monkeypatch.setattr(
        navigation,
        "_write_runtime_file",
        lambda path, content: events.append(("write", path, content)),
    )
    monkeypatch.setitem(
        navigation.GRID_CACHES,
        "/map",
        SimpleNamespace(clear=lambda: events.append("clear_map_cache")),
    )
    monkeypatch.setattr(
        navigation,
        "_s6_command",
        lambda service, action, **kwargs: events.append(("s6", service, action))
        or f"{service} {action}",
    )

    result = navigation.navigation_start(
        navigation.NavigationStartRequest(mode="map", map_name="factory")
    )

    assert result.ok
    assert result.message == "ai_worker_navigation up"
    assert events == [
        ("request_down", "ai_worker_navigation"),
        "force",
        ("s6", "ai_worker_navigation", "down"),
        "clear",
        "clear_map_cache",
        ("write", "/run/navigation_type", "map"),
        ("write", "/run/launch_args/ai_worker_navigation", "map_name:=factory"),
        ("s6", "ai_worker_navigation", "up"),
    ]


def test_navigation_start_keeps_map_cache_for_nav_mode(monkeypatch):
    events = []

    monkeypatch.setattr(
        navigation,
        "_request_s6_service_down",
        lambda service: events.append(("request_down", service))
        or f"{service} down requested",
    )
    monkeypatch.setattr(
        navigation,
        "_clear_navigation_runtime_files",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        navigation,
        "_force_stop_navigation_processes",
        lambda: events.append("force") or "",
    )
    monkeypatch.setattr(
        navigation,
        "_write_runtime_file",
        lambda path, content: events.append(("write", path, content)),
    )
    monkeypatch.setitem(
        navigation.GRID_CACHES,
        "/map",
        SimpleNamespace(clear=lambda: events.append("clear_map_cache")),
    )
    monkeypatch.setattr(
        navigation,
        "_s6_command",
        lambda service, action, **kwargs: events.append(("s6", service, action))
        or f"{service} {action}",
    )

    result = navigation.navigation_start(
        navigation.NavigationStartRequest(mode="nav", map_name="factory")
    )

    assert result.ok
    assert "clear_map_cache" not in events


def test_navigation_start_launches_localization_only(monkeypatch):
    events = []

    monkeypatch.setattr(
        navigation,
        "_request_s6_service_down",
        lambda service: events.append(("request_down", service))
        or f"{service} down requested",
    )
    monkeypatch.setattr(
        navigation,
        "_clear_navigation_runtime_files",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        navigation,
        "_force_stop_navigation_processes",
        lambda: events.append("force") or "",
    )
    monkeypatch.setattr(
        navigation,
        "_write_runtime_file",
        lambda path, content: events.append(("write", path, content)),
    )
    monkeypatch.setattr(
        navigation,
        "_s6_command",
        lambda service, action, **kwargs: events.append(("s6", service, action))
        or f"{service} {action}",
    )
    monkeypatch.setattr(
        navigation,
        "_start_localization_process",
        lambda map_name: events.append(("localize", map_name))
        or "localization launched",
    )

    result = navigation.navigation_start(
        navigation.NavigationStartRequest(mode="localize", map_name="factory")
    )

    assert result.ok
    assert result.message == "localization launched"
    assert events == [
        ("request_down", "ai_worker_navigation"),
        "force",
        ("s6", "ai_worker_navigation", "down"),
        "clear",
        ("write", "/run/navigation_type", "localize"),
        ("write", "/run/launch_args/ai_worker_navigation", "map_name:=factory"),
        ("localize", "factory"),
    ]


def test_navigation_stop_clears_runtime_and_forces_process_group(monkeypatch):
    events = []

    monkeypatch.setattr(
        navigation,
        "_request_s6_service_down",
        lambda service: events.append(("request_down", service))
        or "down requested",
    )
    monkeypatch.setattr(
        navigation,
        "_clear_navigation_runtime_files",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        navigation,
        "_s6_command",
        lambda service, action, **kwargs: events.append(("s6", service, action))
        or "s6 down",
    )
    monkeypatch.setattr(
        navigation,
        "_force_stop_navigation_processes",
        lambda: events.append("force") or "forced",
    )

    result = navigation.navigation_stop()

    assert result.ok
    assert result.message == "down requested\nforced\ns6 down"
    assert events == [
        ("request_down", "ai_worker_navigation"),
        "force",
        ("s6", "ai_worker_navigation", "down"),
        "clear",
    ]


def test_navigation_force_stop_kills_process_group_with_separator(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        return 0, "forced"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setattr(navigation, "_stop_localization_processes", lambda: "")

    assert navigation._force_stop_navigation_processes() == "forced"
    script = captured["command"][-1]
    assert 'kill -TERM -"${PGID}"' in script
    assert 'kill -KILL -"${PGID}"' in script
    assert 'kill -TERM -- -"${PGID}"' not in script
    assert "pkill" not in script


def test_navigation_localization_status_uses_dash_compatible_process_group_check(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        return 0, "up (pid 123 pgid 123) 7 seconds"

    monkeypatch.setattr(navigation, "_exec", fake_exec)

    status = navigation._localization_status()

    assert status.is_up
    assert status.pid == 123
    assert status.uptime_seconds == 7
    script = captured["command"][-1]
    assert 'kill -0 -"${PGID}"' in script
    assert 'kill -0 -- -"${PGID}"' not in script


def test_navigation_stop_defers_busy_s6_lock(monkeypatch):
    from fastapi import HTTPException

    events = []

    monkeypatch.setattr(
        navigation,
        "_request_s6_service_down",
        lambda service: events.append(("request_down", service))
        or "down requested",
    )
    monkeypatch.setattr(
        navigation,
        "_force_stop_navigation_processes",
        lambda: events.append("force") or "forced",
    )
    monkeypatch.setattr(
        navigation,
        "_clear_navigation_runtime_files",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        navigation,
        "_s6_command",
        lambda service, action, **kwargs: events.append(("s6", service, action))
        or (_ for _ in ()).throw(
            HTTPException(503, "s6-rc: fatal: unable to take locks: Resource busy")
        ),
    )

    result = navigation.navigation_stop()

    assert result.ok
    assert "down requested" in result.message
    assert "forced" in result.message
    assert "down sync deferred" in result.message
    assert events == [
        ("request_down", "ai_worker_navigation"),
        "force",
        ("s6", "ai_worker_navigation", "down"),
        "clear",
    ]


def test_navigation_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/navigation/status" in paths
    assert "/navigation/start" in paths
    assert "/navigation/goal/wait" in paths
    assert "/navigation/goals/wait" in paths
    assert "/navigation/initial-pose" in paths
    assert "/navigation/nomotion-update" in paths
    assert "/navigation/global-localization" in paths
    assert "/navigation/amcl/design-localization-params" in paths
    assert "/navigation/maps/pgm/save" in paths
    assert "/navigation/maps/annotations" in paths
    assert "/navigation/maps/annotations/save" in paths
    assert "/navigation/topics/ws" in paths
    assert "/navigation/spots" in paths
    assert "/navigation/missions" in paths
    assert "/navigation/missions/{map_name}" in paths
    assert "/navigation/missions/{map_name}/bt" in paths
    assert "/navigation/missions/{map_name}/bt/default" in paths
    assert "/navigation/missions/{map_name}/duplicate" in paths
    assert "/navigation/missions/{map_name}/rename" in paths


def test_navigation_spots_crud(monkeypatch, tmp_path):
    monkeypatch.setattr(navigation_spots, "SPOTS_ROOT", tmp_path)

    created = navigation_spots.create_spot(
        navigation_spots.SpotCreateRequest(
            id="table_a",
            map_name="factory",
            label="Table A",
            pose=navigation_spots.SpotPose(x=1.0, y=2.0, yaw=0.5),
        )
    )
    assert created.id == "table_a"

    listed = navigation_spots.list_spots("factory")
    assert listed.map_name == "factory"
    assert [spot.id for spot in listed.spots] == ["table_a"]

    updated = navigation_spots.update_spot(
        "table_a",
        navigation_spots.SpotUpdateRequest(
            map_name="factory",
            label="Prep Table",
            linked_bt_tree="prep_table.xml",
        )
    )
    assert updated.label == "Prep Table"
    assert updated.linked_bt_tree == "prep_table.xml"

    deleted = navigation_spots.delete_spot("table_a", map_name="factory")
    assert deleted.ok
    assert navigation_spots.list_spots("factory").spots == []


def test_navigation_spots_rejects_path_like_names(monkeypatch, tmp_path):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_spots, "SPOTS_ROOT", tmp_path)

    with pytest.raises(HTTPException):
        navigation_spots.list_spots("../factory")


def test_navigation_mission_manifest_and_bt_files(monkeypatch, tmp_path):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)

    empty = navigation_missions.load_mission("factory")
    assert empty.exists is False
    assert empty.global_bt == "global.xml"
    assert not hasattr(empty, "compiled_bt")

    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[
                navigation_missions.MissionWaypoint(
                    id="table_a",
                    label="Table A",
                    pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                )
            ],
        ),
    )
    assert saved.exists is True
    assert saved.map_name == "factory"
    assert saved.mission_name == "default"
    assert not hasattr(saved, "compiled_bt")
    assert saved.waypoints[0].local_bt == "locals/table_a.xml"
    assert saved.waypoints[0].local_bt_files == ["locals/table_a.xml"]

    loaded = navigation_missions.load_mission("factory")
    assert loaded.exists is True
    assert loaded.waypoints[0].label == "Table A"

    bt_file = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a.xml",
            content="<root/>",
            waypoint_id="table_a",
            expected_revision=saved.revision,
        ),
    )
    assert bt_file.exists is True
    assert bt_file.revision == saved.revision + 1

    loaded_bt = navigation_missions.load_bt_file(
        "factory",
        path="locals/table_a.xml",
    )
    assert loaded_bt.content == "<root/>"

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as referenced_delete:
        navigation_missions.delete_bt_file(
            "factory",
            path="locals/table_a.xml",
            expected_revision=bt_file.revision,
        )
    assert referenced_delete.value.status_code == 409

    bt_saved = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/orphan.xml",
            content="<root/>",
            expected_revision=bt_file.revision,
        ),
    )
    deleted_bt = navigation_missions.delete_bt_file(
        "factory",
        path="locals/orphan.xml",
        expected_revision=bt_saved.revision,
    )
    assert deleted_bt.exists is False
    assert deleted_bt.revision == bt_saved.revision + 1
    assert (
        tmp_path / "missions" / "factory" / "default" / "locals" / "table_a.xml"
    ).exists()
    assert not (
        tmp_path / "missions" / "factory" / "default" / "locals" / "orphan.xml"
    ).exists()


def test_navigation_missions_migrates_legacy_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    legacy_dir = tmp_path / "missions" / "map_1floor"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "mission.json").write_text(
        '{"waypoints": [], "global_bt": "global.xml"}', encoding="utf-8"
    )
    (legacy_dir / "global.xml").write_text("<root/>", encoding="utf-8")

    missions = navigation_missions.list_missions("map_1floor")

    assert missions.missions == ["default"]
    assert not (legacy_dir / "mission.json").exists()
    assert (legacy_dir / "default" / "mission.json").exists()


def test_navigation_mission_delete_removes_mission_dir(monkeypatch, tmp_path):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[
                navigation_missions.MissionWaypoint(
                    id="table_a",
                    label="Table A",
                    pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                )
            ],
        ),
        mission_name="picnic",
    )
    bt_saved = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a.xml",
            content="<root/>",
            waypoint_id="table_a",
            expected_revision=saved.revision,
        ),
        mission_name="picnic",
    )

    with pytest.raises(HTTPException) as stale:
        navigation_missions.delete_mission(
            "factory",
            mission_name="picnic",
            expected_revision=saved.revision,
        )
    assert stale.value.status_code == 409
    assert (tmp_path / "missions" / "factory" / "picnic").is_dir()

    result = navigation_missions.delete_mission(
        "factory",
        mission_name="picnic",
        expected_revision=bt_saved.revision,
    )

    assert result.deleted is True
    assert result.mission_name == "picnic"
    assert not (tmp_path / "missions" / "factory" / "picnic").exists()

    with pytest.raises(HTTPException) as missing:
        navigation_missions.delete_mission(
            "factory",
            mission_name="picnic",
            expected_revision=bt_saved.revision,
        )
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException):
        navigation_missions.delete_mission(
            "factory", mission_name="../escape", expected_revision=0
        )


def test_navigation_mission_delete_leaves_revision_tombstone(
    monkeypatch, tmp_path
):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
        mission_name="picnic",
    )
    navigation_missions.delete_mission(
        "factory",
        mission_name="picnic",
        expected_revision=saved.revision,
    )

    deleted = navigation_missions.load_mission("factory", mission_name="picnic")
    assert deleted.exists is False
    assert deleted.revision == saved.revision + 1
    with pytest.raises(HTTPException) as stale:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="global.xml",
                content="<root id='stale-session'/>",
                expected_revision=saved.revision,
            ),
            mission_name="picnic",
        )
    assert stale.value.status_code == 409
    with pytest.raises(HTTPException) as missing_revision:
        navigation_missions.save_mission(
            "factory",
            navigation_missions.MissionSaveRequest(waypoints=[]),
            mission_name="picnic",
        )
    assert missing_revision.value.status_code == 409
    assert not (
        tmp_path / "missions" / "factory" / "picnic" / "global.xml"
    ).exists()


@pytest.mark.parametrize("invalid_name", [".", "..", ".revisions", ".TRASH"])
def test_navigation_mission_rejects_storage_control_names(
    monkeypatch, tmp_path, invalid_name
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    sentinel = tmp_path / "missions" / "factory" / "kept" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(HTTPException) as invalid:
        navigation_missions.load_mission(
            "factory",
            mission_name=invalid_name,
        )
    assert invalid.value.status_code == 400
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_navigation_mission_revision_marker_is_symlink_safe(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    map_root = tmp_path / "missions" / "factory"
    outside = tmp_path / "outside"
    map_root.mkdir(parents=True)
    outside.mkdir()
    (map_root / ".revisions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException) as escaped:
        navigation_missions.load_mission("factory", mission_name="inspection")
    assert escaped.value.status_code == 400
    assert list(outside.iterdir()) == []


def test_navigation_mission_corrupt_revision_marker_fails_closed(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    marker = (
        tmp_path
        / "missions"
        / "factory"
        / ".revisions"
        / "inspection.revision"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text("not-a-revision\n", encoding="utf-8")

    with pytest.raises(HTTPException) as corrupt:
        navigation_missions.load_mission("factory", mission_name="inspection")
    assert corrupt.value.status_code == 500


def test_navigation_mission_manifest_symlink_fails_closed(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    outside_manifest = tmp_path / "outside.json"
    secret = "must-not-be-exposed"
    outside_manifest.write_text(
        json.dumps({
            "revision": 9,
            "global_bt": "global.xml",
            "waypoints": [],
            "metadata": {"secret": secret},
        }),
        encoding="utf-8",
    )
    mission_dir = tmp_path / "missions" / "factory" / "inspection"
    mission_dir.mkdir(parents=True)
    (mission_dir / "mission.json").symlink_to(outside_manifest)

    with pytest.raises(HTTPException) as escaped:
        navigation_missions.load_mission(
            "factory", mission_name="inspection"
        )

    assert escaped.value.status_code == 400
    assert secret not in str(escaped.value.detail)
    assert json.loads(outside_manifest.read_text(encoding="utf-8"))["metadata"] == {
        "secret": secret,
    }


def test_navigation_mission_non_regular_manifest_fails_closed(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    manifest_path = (
        tmp_path / "missions" / "factory" / "inspection" / "mission.json"
    )
    manifest_path.mkdir(parents=True)

    with pytest.raises(HTTPException) as invalid:
        navigation_missions.load_mission(
            "factory", mission_name="inspection"
        )

    assert invalid.value.status_code in {400, 500}
    assert "inspection" not in navigation_missions.list_missions("factory").missions


def test_navigation_mission_rename_moves_dir_and_rewrites_name(
    monkeypatch, tmp_path
):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[
                navigation_missions.MissionWaypoint(
                    id="table_a",
                    label="Table A",
                    pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                    local_bt="locals/table_a.xml",
                )
            ],
        ),
        mission_name="picnic",
    )
    bt_saved = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a.xml",
            content="<root/>",
            waypoint_id="table_a",
            expected_revision=saved.revision,
        ),
        mission_name="picnic",
    )

    renamed = navigation_missions.rename_mission(
        "factory",
        navigation_missions.MissionRenameRequest(
            mission_name="picnic",
            new_name="evening",
            expected_revision=bt_saved.revision,
        ),
    )

    assert renamed.exists is True
    assert renamed.mission_name == "evening"
    assert renamed.revision == bt_saved.revision + 1
    assert not (tmp_path / "missions" / "factory" / "picnic").exists()
    new_dir = tmp_path / "missions" / "factory" / "evening"
    assert (new_dir / "locals" / "table_a.xml").read_text() == "<root/>"
    stored = json.loads((new_dir / "mission.json").read_text())
    assert stored["mission_name"] == "evening"

    with pytest.raises(HTTPException) as missing:
        navigation_missions.rename_mission(
            "factory",
            navigation_missions.MissionRenameRequest(
                mission_name="picnic", new_name="ghost"
            ),
        )
    assert missing.value.status_code == 404

    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
        mission_name="other",
    )
    with pytest.raises(HTTPException) as conflict:
        navigation_missions.rename_mission(
            "factory",
            navigation_missions.MissionRenameRequest(
                mission_name="evening",
                new_name="other",
                expected_revision=renamed.revision,
            ),
        )
    assert conflict.value.status_code == 409


def test_navigation_mission_duplicate_copies_manifest_and_bt(
    monkeypatch, tmp_path
):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[
                navigation_missions.MissionWaypoint(
                    id="table_a",
                    label="Table A",
                    pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                    local_bt="locals/table_a.xml",
                )
            ],
        ),
        mission_name="picnic",
    )
    bt_saved = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a.xml",
            content="<root/>",
            waypoint_id="table_a",
            expected_revision=saved.revision,
        ),
        mission_name="picnic",
    )

    copy = navigation_missions.duplicate_mission(
        "factory",
        navigation_missions.MissionDuplicateRequest(
            mission_name="picnic",
            new_name="picnic-copy",
            expected_revision=bt_saved.revision,
        ),
    )

    assert copy.exists is True
    assert copy.mission_name == "picnic-copy"
    assert copy.revision == bt_saved.revision + 1
    assert copy.waypoints[0].label == "Table A"
    copy_dir = tmp_path / "missions" / "factory" / "picnic-copy"
    assert (copy_dir / "locals" / "table_a.xml").read_text() == "<root/>"
    # The copied manifest stores the new name, not the source's.
    stored = json.loads((copy_dir / "mission.json").read_text())
    assert stored["mission_name"] == "picnic-copy"
    # Source is untouched.
    assert (
        tmp_path / "missions" / "factory" / "picnic" / "mission.json"
    ).exists()

    with pytest.raises(HTTPException) as conflict:
        navigation_missions.duplicate_mission(
            "factory",
            navigation_missions.MissionDuplicateRequest(
                mission_name="picnic",
                new_name="picnic-copy",
                expected_revision=bt_saved.revision,
            ),
        )
    assert conflict.value.status_code == 409

    with pytest.raises(HTTPException) as missing:
        navigation_missions.duplicate_mission(
            "factory",
            navigation_missions.MissionDuplicateRequest(
                mission_name="ghost", new_name="ghost-copy"
            ),
        )
    assert missing.value.status_code == 404


def test_navigation_rename_and_duplicate_reissue_target_generations(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    old_target = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
        mission_name="reused",
    )
    old_target = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content="<root id='old-target'/>",
            expected_revision=old_target.revision,
        ),
        mission_name="reused",
    )
    navigation_missions.delete_mission(
        "factory",
        mission_name="reused",
        expected_revision=old_target.revision,
    )
    target_tombstone = navigation_missions.load_mission(
        "factory", mission_name="reused"
    ).revision

    source = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
        mission_name="source",
    )
    renamed = navigation_missions.rename_mission(
        "factory",
        navigation_missions.MissionRenameRequest(
            mission_name="source",
            new_name="reused",
            expected_revision=source.revision,
        ),
    )
    assert renamed.revision == target_tombstone + 1
    source_tombstone = navigation_missions.load_mission(
        "factory", mission_name="source"
    )
    assert source_tombstone.exists is False
    assert source_tombstone.revision == source.revision + 1
    with pytest.raises(HTTPException) as stale_target:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="global.xml",
                content="<root id='stale-target'/>",
                expected_revision=old_target.revision,
            ),
            mission_name="reused",
        )
    assert stale_target.value.status_code == 409

    old_copy = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
        mission_name="reused-copy",
    )
    navigation_missions.delete_mission(
        "factory",
        mission_name="reused-copy",
        expected_revision=old_copy.revision,
    )
    copy_tombstone = navigation_missions.load_mission(
        "factory", mission_name="reused-copy"
    ).revision
    copied = navigation_missions.duplicate_mission(
        "factory",
        navigation_missions.MissionDuplicateRequest(
            mission_name="reused",
            new_name="reused-copy",
            expected_revision=renamed.revision,
        ),
    )
    assert copied.revision == max(renamed.revision, copy_tombstone) + 1
    assert navigation_missions.load_mission(
        "factory", mission_name="reused"
    ).revision == renamed.revision


def test_navigation_duplicate_rejects_nested_symlinks(monkeypatch, tmp_path):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    source = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
        mission_name="source",
    )
    source_dir = tmp_path / "missions" / "factory" / "source"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.xml").write_text("<secret/>", encoding="utf-8")
    (source_dir / "locals").mkdir()
    (source_dir / "locals" / "redirect").symlink_to(
        outside, target_is_directory=True
    )

    with pytest.raises(HTTPException) as rejected:
        navigation_missions.duplicate_mission(
            "factory",
            navigation_missions.MissionDuplicateRequest(
                mission_name="source",
                new_name="copy",
                expected_revision=source.revision,
            ),
        )
    assert rejected.value.status_code == 400
    assert not (tmp_path / "missions" / "factory" / "copy").exists()
    assert (outside / "secret.xml").read_text(encoding="utf-8") == "<secret/>"


@pytest.mark.parametrize("failure_stage", ["copy", "manifest"])
def test_navigation_duplicate_failure_does_not_publish_or_block_target(
    monkeypatch, tmp_path, failure_stage
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    source = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                local_bt="locals/table_a.xml",
            )
        ]),
        mission_name="source",
    )
    source = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a.xml",
            content="<root id='source-local'/>",
            waypoint_id="table_a",
            expected_revision=source.revision,
        ),
        mission_name="source",
    )
    target_name = f"copy-after-{failure_stage}-failure"
    target_dir = tmp_path / "missions" / "factory" / target_name

    with monkeypatch.context() as failure:
        if failure_stage == "copy":
            def fail_copytree(_source, destination, *args, **kwargs):
                partial = Path(destination)
                partial.mkdir(parents=True)
                (partial / "partial.txt").write_text(
                    "incomplete", encoding="utf-8"
                )
                raise OSError("injected copy failure")

            failure.setattr(
                navigation_missions.shutil,
                "copytree",
                fail_copytree,
            )
        else:
            def fail_manifest_write(manifest):
                if manifest.mission_name == target_name:
                    raise HTTPException(500, "injected manifest failure")
                raise AssertionError("unexpected manifest write")

            failure.setattr(
                navigation_missions,
                "_write_manifest",
                fail_manifest_write,
            )

        with pytest.raises(HTTPException) as failed:
            navigation_missions.duplicate_mission(
                "factory",
                navigation_missions.MissionDuplicateRequest(
                    mission_name="source",
                    new_name=target_name,
                    expected_revision=source.revision,
                ),
            )

    assert failed.value.status_code == 500
    assert not target_dir.exists()
    assert target_name not in navigation_missions.list_missions("factory").missions
    reserved = navigation_missions.load_mission(
        "factory", mission_name=target_name
    )
    assert reserved.exists is False
    assert reserved.revision > source.revision

    copied = navigation_missions.duplicate_mission(
        "factory",
        navigation_missions.MissionDuplicateRequest(
            mission_name="source",
            new_name=target_name,
            expected_revision=source.revision,
        ),
    )

    assert copied.exists is True
    assert copied.revision == max(source.revision, reserved.revision) + 1
    assert copied.waypoints[0].local_bt == "locals/table_a.xml"
    assert (
        target_dir / "locals" / "table_a.xml"
    ).read_text(encoding="utf-8") == "<root id='source-local'/>"
    published = navigation_missions.load_mission(
        "factory", mission_name=target_name
    )
    assert published.revision == copied.revision
    assert published.waypoints == copied.waypoints
    stored = json.loads((target_dir / "mission.json").read_text(encoding="utf-8"))
    assert stored["mission_name"] == target_name
    assert stored["revision"] == copied.revision
    assert target_name in navigation_missions.list_missions("factory").missions


def test_navigation_mission_save_prunes_orphan_local_bt_files(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    locals_dir = tmp_path / "missions" / "factory" / "default" / "locals"
    locals_dir.mkdir(parents=True)
    (locals_dir / "table_a.xml").write_text("<root/>", encoding="utf-8")
    (locals_dir / "waypoint_2.xml").write_text("<root/>", encoding="utf-8")
    (locals_dir / "waypoint_5.xml").write_text("<root/>", encoding="utf-8")
    nested_dir = locals_dir / "table_a"
    nested_dir.mkdir()
    (nested_dir / "alternate.xml").write_text("<root/>", encoding="utf-8")
    (nested_dir / "orphan.xml").write_text("<root/>", encoding="utf-8")
    orphan_dir = locals_dir / "removed_waypoint"
    orphan_dir.mkdir()
    (orphan_dir / "main.xml").write_text("<root/>", encoding="utf-8")

    navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[
                navigation_missions.MissionWaypoint(
                    id="table_a",
                    label="Table A",
                    pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                    local_bt="locals/table_a.xml",
                    local_bt_files=[
                        "locals/table_a.xml",
                        "locals/table_a/alternate.xml",
                    ],
                )
            ],
        ),
    )

    # Referenced file survives; leftovers from deleted/renamed waypoints go.
    assert (locals_dir / "table_a.xml").exists()
    assert (nested_dir / "alternate.xml").exists()
    assert not (nested_dir / "orphan.xml").exists()
    assert not orphan_dir.exists()
    assert not (locals_dir / "waypoint_2.xml").exists()
    assert not (locals_dir / "waypoint_5.xml").exists()


def test_navigation_mission_prune_preserves_global_bt_under_locals(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    locals_dir = tmp_path / "missions" / "factory" / "default" / "locals"
    locals_dir.mkdir(parents=True)
    global_path = locals_dir / "global.xml"
    orphan_path = locals_dir / "orphan.xml"
    global_path.write_text("<root id='global'/>", encoding="utf-8")
    orphan_path.write_text("<root id='orphan'/>", encoding="utf-8")

    saved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            global_bt="locals/global.xml",
            waypoints=[],
        ),
    )

    assert saved.global_bt == "locals/global.xml"
    assert global_path.read_text(encoding="utf-8") == "<root id='global'/>"
    assert not orphan_path.exists()


def test_navigation_mission_v1_manifest_promotes_default_to_file_library(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    mission_dir = tmp_path / "missions" / "factory" / "default"
    mission_dir.mkdir(parents=True)
    (mission_dir / "mission.json").write_text(
        json.dumps({
            "schema_version": 1,
            "map_name": "factory",
            "mission_name": "default",
            "global_bt": "global.xml",
            "waypoints": [{
                "id": "table_a",
                "label": "Table A",
                "pose": {"frame_id": "map", "x": 1.0, "y": 2.0, "yaw": 0.5},
                "local_bt": "locals/table_a.xml",
            }],
        }),
        encoding="utf-8",
    )

    loaded = navigation_missions.load_mission("factory")

    assert loaded.schema_version == 2
    assert loaded.waypoints[0].local_bt == "locals/table_a.xml"
    assert loaded.waypoints[0].local_bt_files == ["locals/table_a.xml"]


def test_navigation_mission_preserves_alternates_when_default_changes(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    locals_dir = tmp_path / "missions" / "factory" / "default" / "locals"
    locals_dir.mkdir(parents=True)
    default_path = locals_dir / "table_a.xml"
    alternate_path = locals_dir / "table_a_alt.xml"
    default_path.write_text("<root id='default'/>", encoding="utf-8")
    alternate_path.write_text("<root id='alternate'/>", encoding="utf-8")

    def waypoint(default_bt, *, files_marker=True):
        kwargs = {
            "id": "table_a",
            "label": "Table A",
            "pose": navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
            "local_bt": default_bt,
        }
        if files_marker:
            kwargs["local_bt_files"] = [
                "locals/table_a.xml",
                "locals/table_a_alt.xml",
            ]
        return navigation_missions.MissionWaypoint(**kwargs)

    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[waypoint("locals/table_a.xml")]
        ),
    )
    switched = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[waypoint("locals/table_a_alt.xml")],
            expected_revision=initial.revision,
        ),
    )
    assert switched.waypoints[0].local_bt == "locals/table_a_alt.xml"
    assert switched.waypoints[0].local_bt_files == [
        "locals/table_a_alt.xml",
        "locals/table_a.xml",
    ]
    assert default_path.exists()
    assert alternate_path.exists()

    # A cached v1 client omits local_bt_files entirely. It may change the
    # default pointer, but must not silently delete the v2 library.
    preserved = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            waypoints=[waypoint("locals/table_a.xml", files_marker=False)],
            expected_revision=switched.revision,
        ),
    )
    assert preserved.waypoints[0].local_bt_files == [
        "locals/table_a.xml",
        "locals/table_a_alt.xml",
    ]
    assert alternate_path.exists()

    # An explicit v2 empty list is authoritative; normalization keeps the
    # selected default and prunes the removed alternative.
    removed = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            expected_revision=preserved.revision,
            waypoints=[navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                local_bt="locals/table_a.xml",
                local_bt_files=[],
            )]
        ),
    )
    assert removed.waypoints[0].local_bt_files == ["locals/table_a.xml"]
    assert default_path.exists()
    assert not alternate_path.exists()


def test_navigation_bt_save_registers_waypoint_file_in_manifest(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    pose = navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=pose,
                local_bt="locals/table_a/default.xml",
            )
        ]),
        mission_name="picnic",
    )

    saved = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a//alternate.xml",
            content="<root id='alternate'/>",
            waypoint_id="table_a",
            expected_revision=initial.revision,
        ),
        mission_name="picnic",
    )

    assert saved.path == "locals/table_a/alternate.xml"
    alternate_path = (
        tmp_path
        / "missions"
        / "factory"
        / "picnic"
        / "locals"
        / "table_a"
        / "alternate.xml"
    )
    assert alternate_path.read_text(encoding="utf-8") == "<root id='alternate'/>"
    loaded = navigation_missions.load_mission("factory", mission_name="picnic")
    assert loaded.waypoints[0].local_bt == "locals/table_a/default.xml"
    assert loaded.waypoints[0].local_bt_files == [
        "locals/table_a/default.xml",
        "locals/table_a/alternate.xml",
    ]
    stored = json.loads((
        tmp_path / "missions" / "factory" / "picnic" / "mission.json"
    ).read_text(encoding="utf-8"))
    assert stored["waypoints"][0]["local_bt_files"] == [
        "locals/table_a/default.xml",
        "locals/table_a/alternate.xml",
    ]

    # A later cached-v1 mission save still preserves the file registered by
    # Save As because local_bt_files was omitted, not explicitly cleared.
    cached_v1_save = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Renamed Table",
                pose=pose,
                local_bt="locals/table_a/default.xml",
            )
        ], expected_revision=saved.revision),
        mission_name="picnic",
    )
    assert cached_v1_save.waypoints[0].local_bt_files == [
        "locals/table_a/default.xml",
        "locals/table_a/alternate.xml",
    ]
    assert alternate_path.exists()


def test_navigation_bt_save_keeps_existing_legacy_root_file_compatible(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=navigation_missions.SpotPose(x=0.0, y=0.0, yaw=0.0),
                local_bt="legacy_table.xml",
            )
        ]),
    )

    saved = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="legacy_table.xml",
            content="<root id='legacy'/>",
            waypoint_id="table_a",
            expected_revision=initial.revision,
        ),
    )

    assert saved.exists is True
    assert navigation_missions.load_bt_file(
        "factory", path="legacy_table.xml"
    ).content == "<root id='legacy'/>"


def test_navigation_mission_revision_rejects_stale_library_overwrite(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    pose = navigation_missions.SpotPose(x=0.0, y=0.0, yaw=0.0)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=pose,
                local_bt="locals/table_a/default.xml",
            )
        ]),
    )
    assert initial.revision == 1

    registered = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a/alternate.xml",
            content="<root/>",
            waypoint_id="table_a",
            expected_revision=initial.revision,
        ),
    )
    assert registered.revision == 2

    with pytest.raises(HTTPException) as stale:
        navigation_missions.save_mission(
            "factory",
            navigation_missions.MissionSaveRequest(
                expected_revision=initial.revision,
                waypoints=[navigation_missions.MissionWaypoint(
                    id="table_a",
                    label="Stale Table",
                    pose=pose,
                    local_bt="locals/table_a/default.xml",
                    local_bt_files=["locals/table_a/default.xml"],
                )],
            ),
        )
    assert stale.value.status_code == 409
    current = navigation_missions.load_mission("factory")
    assert current.revision == registered.revision
    assert current.waypoints[0].local_bt_files == [
        "locals/table_a/default.xml",
        "locals/table_a/alternate.xml",
    ]
    assert (
        tmp_path
        / "missions"
        / "factory"
        / "default"
        / "locals"
        / "table_a"
        / "alternate.xml"
    ).exists()


def test_navigation_bt_save_rejects_stale_same_file_overwrite(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=navigation_missions.SpotPose(x=0.0, y=0.0, yaw=0.0),
                local_bt="locals/table_a.xml",
            )
        ]),
    )

    first_session_revision = initial.revision
    second_session_revision = initial.revision
    first_save = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/table_a.xml",
            content="<root id='first-session'/>",
            waypoint_id="table_a",
            expected_revision=first_session_revision,
        ),
    )

    assert first_save.revision == first_session_revision + 1
    with pytest.raises(HTTPException) as stale:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals/table_a.xml",
                content="<root id='second-session'/>",
                waypoint_id="table_a",
                expected_revision=second_session_revision,
            ),
        )
    assert stale.value.status_code == 409
    assert navigation_missions.load_bt_file(
        "factory", path="locals/table_a.xml"
    ).content == "<root id='first-session'/>"
    assert navigation_missions.load_mission("factory").revision == first_save.revision


def test_navigation_bt_save_burns_revision_before_manifest_commit(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
    )
    current = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content="<root id='before-crash'/>",
            expected_revision=initial.revision,
        ),
    )

    with monkeypatch.context() as crash:
        crash.setattr(
            navigation_missions,
            "_write_manifest",
            lambda _manifest: (_ for _ in ()).throw(
                SystemExit("injected crash before manifest commit")
            ),
        )
        with pytest.raises(SystemExit):
            navigation_missions.save_bt_file(
                "factory",
                navigation_missions.MissionBtFileRequest(
                    path="global.xml",
                    content="<root id='after-semantic-write'/>",
                    expected_revision=current.revision,
                ),
            )

    effective = navigation_missions.load_mission("factory")
    assert effective.revision == current.revision + 1
    assert navigation_missions.load_bt_file(
        "factory", path="global.xml"
    ).content == "<root id='after-semantic-write'/>"
    with pytest.raises(HTTPException) as stale:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="global.xml",
                content="<root id='stale-overwrite'/>",
                expected_revision=current.revision,
            ),
        )
    assert stale.value.status_code == 409


def test_navigation_bt_default_burns_revision_before_manifest_commit(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                local_bt="locals/table_a/default.xml",
                local_bt_files=[
                    "locals/table_a/default.xml",
                    "locals/table_a/alternate.xml",
                ],
            )
        ]),
    )
    current = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content=(
                '<root><BehaviorTree ID="MainTree"><MissionStep '
                'waypoint_id="table_a" '
                'local_bt="locals/table_a/default.xml"/>'
                '</BehaviorTree></root>'
            ),
            expected_revision=initial.revision,
        ),
    )

    with monkeypatch.context() as crash:
        crash.setattr(
            navigation_missions,
            "_write_manifest",
            lambda _manifest: (_ for _ in ()).throw(
                SystemExit("injected crash before manifest commit")
            ),
        )
        with pytest.raises(SystemExit):
            navigation_missions.set_default_bt_file(
                "factory",
                navigation_missions.MissionBtDefaultRequest(
                    waypoint_id="table_a",
                    path="locals/table_a/alternate.xml",
                    expected_revision=current.revision,
                ),
            )

    effective = navigation_missions.load_mission("factory")
    assert effective.revision == current.revision + 1
    updated_global = navigation_missions.load_bt_file(
        "factory", path="global.xml"
    ).content
    assert 'local_bt="locals/table_a/alternate.xml"' in updated_global
    with pytest.raises(HTTPException) as stale:
        navigation_missions.set_default_bt_file(
            "factory",
            navigation_missions.MissionBtDefaultRequest(
                waypoint_id="table_a",
                path="locals/table_a/alternate.xml",
                expected_revision=current.revision,
            ),
        )
    assert stale.value.status_code == 409


def test_navigation_bt_delete_burns_revision_before_manifest_commit(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
    )
    current = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/orphan.xml",
            content="<root id='delete-me'/>",
            expected_revision=initial.revision,
        ),
    )
    orphan_path = (
        tmp_path
        / "missions"
        / "factory"
        / "default"
        / "locals"
        / "orphan.xml"
    )
    assert orphan_path.exists()

    with monkeypatch.context() as crash:
        crash.setattr(
            navigation_missions,
            "_write_manifest",
            lambda _manifest: (_ for _ in ()).throw(
                SystemExit("injected crash before manifest commit")
            ),
        )
        with pytest.raises(SystemExit):
            navigation_missions.delete_bt_file(
                "factory",
                path="locals/orphan.xml",
                expected_revision=current.revision,
            )

    assert not orphan_path.exists()
    effective = navigation_missions.load_mission("factory")
    assert effective.revision == current.revision + 1
    with pytest.raises(HTTPException) as stale:
        navigation_missions.delete_bt_file(
            "factory",
            path="locals/orphan.xml",
            expected_revision=current.revision,
        )
    assert stale.value.status_code == 409


def test_navigation_bt_delete_rejects_stale_missing_file(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
    )
    concurrent = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content="<root id='concurrent'/>",
            expected_revision=initial.revision,
        ),
    )

    with pytest.raises(HTTPException) as stale:
        navigation_missions.delete_bt_file(
            "factory",
            path="locals/already-pruned.xml",
            expected_revision=initial.revision,
        )
    assert stale.value.status_code == 409
    assert navigation_missions.load_mission("factory").revision == concurrent.revision


def test_navigation_new_mission_file_upload_reserves_revision(
    monkeypatch, tmp_path
):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    first = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content="<root id='first-session'/>",
            expected_revision=0,
        ),
        mission_name="inspection",
    )
    assert first.revision == 1

    with pytest.raises(HTTPException) as stale:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals/other-session.xml",
                content="<root id='other-session'/>",
                expected_revision=0,
            ),
            mission_name="inspection",
        )
    assert stale.value.status_code == 409
    assert not (
        tmp_path
        / "missions"
        / "factory"
        / "inspection"
        / "locals"
        / "other-session.xml"
    ).exists()

    second = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/waypoint.xml",
            content="<root id='first-session-local'/>",
            expected_revision=first.revision,
        ),
        mission_name="inspection",
    )
    assert second.revision == 2
    finalized = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(
            expected_revision=second.revision,
            waypoints=[navigation_missions.MissionWaypoint(
                id="waypoint",
                label="Waypoint",
                pose=navigation_missions.SpotPose(x=0.0, y=0.0, yaw=0.0),
                local_bt="locals/waypoint.xml",
            )],
        ),
        mission_name="inspection",
    )
    assert finalized.exists is True
    assert finalized.revision == 3


def test_navigation_bt_save_validates_ownership_before_overwrite(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    pose = navigation_missions.SpotPose(x=0.0, y=0.0, yaw=0.0)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="a", label="A", pose=pose, local_bt="locals/a.xml"
            ),
            navigation_missions.MissionWaypoint(
                id="b", label="B", pose=pose, local_bt="locals/b.xml"
            ),
        ]),
    )
    owner_save = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/b.xml",
            content="<root id='owner-b'/>",
            waypoint_id="b",
            expected_revision=initial.revision,
        ),
    )

    with pytest.raises(HTTPException) as missing_revision:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals/b.xml",
                content="<root id='legacy-client-overwrite'/>",
            ),
        )
    assert missing_revision.value.status_code == 409

    with pytest.raises(HTTPException) as missing_owner:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals/b.xml",
                content="<root id='unchecked-overwrite'/>",
                expected_revision=owner_save.revision,
            ),
        )
    assert missing_owner.value.status_code == 409

    valid_overwrite = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="locals/b.xml",
            content="<root id='owner-b-updated'/>",
            waypoint_id="b",
            expected_revision=owner_save.revision,
        ),
    )

    with pytest.raises(HTTPException) as conflict:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals//b.xml",
                content="<root id='owner-a-overwrite'/>",
                waypoint_id="a",
                expected_revision=valid_overwrite.revision,
            ),
        )
    assert conflict.value.status_code == 400
    b_path = (
        tmp_path / "missions" / "factory" / "default" / "locals" / "b.xml"
    )
    assert b_path.read_text(encoding="utf-8") == "<root id='owner-b-updated'/>"
    loaded = navigation_missions.load_mission("factory")
    assert loaded.waypoints[0].local_bt_files == ["locals/a.xml"]
    assert loaded.waypoints[1].local_bt_files == ["locals/b.xml"]

    global_save = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content="<root id='global'/>",
            expected_revision=valid_overwrite.revision,
        ),
    )
    with pytest.raises(HTTPException) as global_collision:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="global.xml",
                content="<root id='local-overwrite'/>",
                waypoint_id="a",
                expected_revision=global_save.revision,
            ),
        )
    assert global_collision.value.status_code == 400
    assert navigation_missions.load_bt_file(
        "factory", path="global.xml"
    ).content == "<root id='global'/>"

    with pytest.raises(HTTPException) as root_level_new_file:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="alternate.xml",
                content="<root/>",
                waypoint_id="a",
                expected_revision=global_save.revision,
            ),
        )
    assert root_level_new_file.value.status_code == 400

    missing_path = (
        tmp_path
        / "missions"
        / "factory"
        / "default"
        / "locals"
        / "missing.xml"
    )
    with pytest.raises(HTTPException) as missing_waypoint:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals/missing.xml",
                content="<root/>",
                waypoint_id="missing",
                expected_revision=global_save.revision,
            ),
        )
    assert missing_waypoint.value.status_code == 404
    assert not missing_path.exists()


def test_navigation_bt_default_requires_owned_file_and_persists(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[
            navigation_missions.MissionWaypoint(
                id="table_a",
                label="Table A",
                pose=navigation_missions.SpotPose(x=1.0, y=2.0, yaw=0.5),
                local_bt="locals/table_a/default.xml",
                local_bt_files=[
                    "locals/table_a/default.xml",
                    "locals/table_a/alternate.xml",
                ],
            )
        ]),
        mission_name="picnic",
    )
    global_save = navigation_missions.save_bt_file(
        "factory",
        navigation_missions.MissionBtFileRequest(
            path="global.xml",
            content=(
                '<root><BehaviorTree ID="MainTree"><Sequence>'
                '<MissionStep waypoint_id="table_a" '
                'local_bt="locals/table_a/default.xml"/>'
                '<MissionStep waypoint_id="table_a" '
                'local_bt="locals/table_a/default.xml"/>'
                '</Sequence></BehaviorTree></root>'
            ),
            expected_revision=initial.revision,
        ),
        mission_name="picnic",
    )

    switched = navigation_missions.set_default_bt_file(
        "factory",
        navigation_missions.MissionBtDefaultRequest(
            waypoint_id="table_a",
            path="locals/table_a//alternate.xml",
            expected_revision=global_save.revision,
        ),
        mission_name="picnic",
    )

    assert switched.waypoints[0].local_bt == "locals/table_a/alternate.xml"
    assert switched.waypoints[0].local_bt_files == [
        "locals/table_a/alternate.xml",
        "locals/table_a/default.xml",
    ]
    reloaded = navigation_missions.load_mission("factory", mission_name="picnic")
    assert reloaded.waypoints[0].local_bt == "locals/table_a/alternate.xml"
    updated_global = navigation_missions.load_bt_file(
        "factory",
        path="global.xml",
        mission_name="picnic",
    ).content
    assert updated_global.count('local_bt="locals/table_a/alternate.xml"') == 2
    assert 'local_bt="locals/table_a/default.xml"' not in updated_global

    with pytest.raises(HTTPException) as unowned:
        navigation_missions.set_default_bt_file(
            "factory",
            navigation_missions.MissionBtDefaultRequest(
                waypoint_id="table_a",
                path="locals/table_a/unowned.xml",
                expected_revision=switched.revision,
            ),
            mission_name="picnic",
        )
    assert unowned.value.status_code == 400
    unchanged = navigation_missions.load_mission("factory", mission_name="picnic")
    assert unchanged.waypoints[0].local_bt == "locals/table_a/alternate.xml"


def test_navigation_mission_rejects_duplicate_waypoints_and_bt_ownership(
    monkeypatch, tmp_path
):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    pose = navigation_missions.SpotPose(x=0.0, y=0.0, yaw=0.0)

    with pytest.raises(HTTPException) as duplicate_id:
        navigation_missions.save_mission(
            "factory",
            navigation_missions.MissionSaveRequest(waypoints=[
                navigation_missions.MissionWaypoint(
                    id="same", label="A", pose=pose, local_bt="locals/a.xml"
                ),
                navigation_missions.MissionWaypoint(
                    id="same", label="B", pose=pose, local_bt="locals/b.xml"
                ),
            ]),
        )
    assert duplicate_id.value.status_code == 400

    with pytest.raises(HTTPException) as duplicate_path:
        navigation_missions.save_mission(
            "factory",
            navigation_missions.MissionSaveRequest(waypoints=[
                navigation_missions.MissionWaypoint(
                    id="a", label="A", pose=pose, local_bt="locals/shared.xml"
                ),
                navigation_missions.MissionWaypoint(
                    id="b", label="B", pose=pose, local_bt="locals/SHARED.xml"
                ),
            ]),
        )
    assert duplicate_path.value.status_code == 400

    with pytest.raises(HTTPException) as alias_path:
        navigation_missions.save_mission(
            "factory",
            navigation_missions.MissionSaveRequest(waypoints=[
                navigation_missions.MissionWaypoint(
                    id="a", label="A", pose=pose, local_bt="locals/shared.xml"
                ),
                navigation_missions.MissionWaypoint(
                    id="b", label="B", pose=pose, local_bt="locals//shared.xml"
                ),
            ]),
        )
    assert alias_path.value.status_code == 400

    with pytest.raises(HTTPException) as global_path:
        navigation_missions.save_mission(
            "factory",
            navigation_missions.MissionSaveRequest(
                global_bt="global.xml",
                waypoints=[navigation_missions.MissionWaypoint(
                    id="a", label="A", pose=pose, local_bt="GLOBAL.xml"
                )],
            ),
        )
    assert global_path.value.status_code == 400


def test_navigation_missions_reject_path_escape(monkeypatch, tmp_path):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)

    with pytest.raises(HTTPException):
        navigation_missions.load_mission("../factory")
    with pytest.raises(HTTPException):
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="../global.xml",
                content="<root/>",
            ),
        )
    with pytest.raises(HTTPException):
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="mission.json",
                content="{}",
            ),
        )


def test_navigation_missions_reject_bt_path_through_intermediate_symlink(
    monkeypatch, tmp_path
):
    from fastapi import HTTPException

    monkeypatch.setattr(navigation_missions, "NAVIGATION_DATA_ROOT", tmp_path)
    initial = navigation_missions.save_mission(
        "factory",
        navigation_missions.MissionSaveRequest(waypoints=[]),
    )
    mission_dir = tmp_path / "missions" / "factory" / "default"
    locals_dir = mission_dir / "locals"
    locals_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "escape.xml"
    outside_file.write_text("<root id='outside'/>", encoding="utf-8")
    (locals_dir / "redirect").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(HTTPException) as read_escape:
        navigation_missions.load_bt_file(
            "factory", path="locals/redirect/escape.xml"
        )
    assert read_escape.value.status_code == 400

    with pytest.raises(HTTPException) as write_escape:
        navigation_missions.save_bt_file(
            "factory",
            navigation_missions.MissionBtFileRequest(
                path="locals/redirect/new.xml",
                content="<root id='write'/>",
                expected_revision=initial.revision,
            ),
        )
    assert write_escape.value.status_code == 400
    assert not (outside_dir / "new.xml").exists()

    with pytest.raises(HTTPException) as delete_escape:
        navigation_missions.delete_bt_file(
            "factory", path="locals/redirect/escape.xml"
        )
    assert delete_escape.value.status_code == 400
    assert outside_file.read_text(encoding="utf-8") == "<root id='outside'/>"


def test_navigation_grid_data_crc32_uses_only_map_data():
    first = {"info": {"width": 2}, "data": [-1, 0, 100, 0]}
    same_data = {"info": {"width": 4}, "data": [-1, 0, 100, 0]}
    changed = {"info": {"width": 2}, "data": [-1, 0, 99, 0]}

    marker = navigation_grid_cache.occupancy_grid_data_crc32(first)
    assert navigation_grid_cache.occupancy_grid_data_crc32(same_data) == marker
    assert navigation_grid_cache.occupancy_grid_data_crc32(changed) != marker


def test_navigation_grid_cache_serializes_only_changed_data():
    cache = navigation_grid_cache.OccupancyGridCache("/map")

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 1]})
    marker, payload = cache.serialized_if_changed(None)
    assert json.loads(payload) == {
        "available": True,
        "data": {"info": {"width": 2}, "data": [0, 1]},
    }
    assert cache.serialized_if_changed(marker) == (marker, None)

    cache.cache_ros_message({"info": {"width": 99}, "data": [0, 1]})
    metadata_marker, metadata_payload = cache.serialized_if_changed(marker)
    assert metadata_marker != marker
    assert json.loads(metadata_payload)["data"]["info"]["width"] == 99

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 2]})
    changed_marker, changed_payload = cache.serialized_if_changed(metadata_marker)
    assert changed_marker != metadata_marker
    assert json.loads(changed_payload)["data"]["data"] == [0, 2]

    cache.clear()
    cleared_marker, cleared_payload = cache.serialized_if_changed(changed_marker)
    assert cleared_marker != changed_marker
    assert json.loads(cleared_payload) == {"available": False}

    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 2]})
    restored_marker, restored_payload = cache.serialized_if_changed(cleared_marker)
    assert restored_marker != cleared_marker
    assert json.loads(restored_payload)["data"]["data"] == [0, 2]


def test_navigation_grid_cache_keeps_full_grids_without_map_msgs(
    monkeypatch, caplog
):
    created_topics = []
    executors = []

    class FakeOccupancyGrid:
        pass

    class FakeQoSProfile:
        def __init__(self, **kwargs):
            self.settings = kwargs

    class FakeNode:
        def __init__(self, name):
            self.name = name

        def get_publishers_info_by_topic(self, topic):
            return []

        def create_subscription(self, message_type, topic, callback, qos):
            created_topics.append((message_type, topic, callback, qos))
            return SimpleNamespace(topic=topic)

    class FakeExecutor:
        def __init__(self):
            self.spun = False
            executors.append(self)

        def add_node(self, node):
            self.node = node

        def spin_once(self, timeout_sec):
            self.timeout_sec = timeout_sec

        def spin(self):
            self.spun = True

    rclpy_stub = types.ModuleType("rclpy")
    rclpy_stub.ok = lambda: True
    rclpy_stub.init = lambda: None
    executors_stub = types.ModuleType("rclpy.executors")
    executors_stub.SingleThreadedExecutor = FakeExecutor
    node_stub = types.ModuleType("rclpy.node")
    node_stub.Node = FakeNode
    qos_stub = types.ModuleType("rclpy.qos")
    qos_stub.DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL="transient")
    qos_stub.QoSProfile = FakeQoSProfile
    qos_stub.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    nav_msgs_stub = types.ModuleType("nav_msgs")
    nav_msgs_msg_stub = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg_stub.OccupancyGrid = FakeOccupancyGrid

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_stub)
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors_stub)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_stub)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_stub)
    monkeypatch.setitem(sys.modules, "nav_msgs", nav_msgs_stub)
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav_msgs_msg_stub)
    monkeypatch.delitem(sys.modules, "map_msgs.msg", raising=False)
    monkeypatch.setitem(sys.modules, "map_msgs", None)

    with caplog.at_level(
        "WARNING", logger="supervisor_api.navigation_topics"
    ):
        navigation_grid_cache._ros_grid_spin()

    assert executors[0].spun is True
    assert {topic for _, topic, _, _ in created_topics} == set(
        navigation_grid_cache.GRID_TOPICS
    )
    assert all(
        message_type is FakeOccupancyGrid
        for message_type, _, _, _ in created_topics
    )
    assert "full OccupancyGrid messages" in caplog.text


@pytest.mark.parametrize("architecture", ["amd64", "arm64"])
def test_navigation_runtime_image_installs_map_msgs(architecture):
    dockerfile = REPO_ROOT / "docker" / f"Dockerfile.{architecture}"

    assert "ros-${ROS_DISTRO}-map-msgs" in dockerfile.read_text(
        encoding="utf-8"
    )


def test_navigation_grid_cache_sends_costmap_deltas_and_resyncs_lagging_clients():
    cache = navigation_grid_cache.OccupancyGridCache(
        "/global_costmap/costmap"
    )
    cache.cache_ros_message({
        "header": {"frame_id": "map"},
        "info": {"width": 4, "height": 2},
        "data": [0, 1, 2, 3, 4, 5, 6, 7],
    })
    full_marker, full_payload = cache.serialized_if_changed(None)
    assert "data" in json.loads(full_payload)

    cache.cache_ros_update({
        "header": {"frame_id": "map", "stamp": {"sec": 2}},
        "x": 1,
        "y": 0,
        "width": 2,
        "height": 2,
        "data": [10, 20, 50, 60],
    })
    first_update_marker, first_update_payload = cache.serialized_if_changed(
        full_marker
    )
    assert json.loads(first_update_payload)["update"]["data"] == [10, 20, 50, 60]

    # A newly connected client receives the current assembled full grid.
    _, reconnect_payload = cache.serialized_if_changed(None)
    assert json.loads(reconnect_payload)["data"]["data"] == [
        0, 10, 20, 3, 4, 50, 60, 7,
    ]

    cache.cache_ros_update({
        "x": 0,
        "y": 1,
        "width": 1,
        "height": 1,
        "data": [99],
    })
    _, current_delta = cache.serialized_if_changed(first_update_marker)
    assert json.loads(current_delta)["update"]["data"] == [99]

    # A client that missed an intermediate delta gets a full resync instead.
    _, lagging_payload = cache.serialized_if_changed(full_marker)
    assert json.loads(lagging_payload)["data"]["data"] == [
        0, 10, 20, 3, 99, 50, 60, 7,
    ]


def test_navigation_grid_cache_compacts_broad_costmap_bounds_to_actual_changes():
    cache = navigation_grid_cache.OccupancyGridCache(
        "/global_costmap/costmap"
    )
    cache.cache_ros_message({
        "header": {"frame_id": "map"},
        "info": {"width": 5, "height": 4},
        "data": [0] * 20,
    })
    full_marker, _ = cache.serialized_if_changed(None)

    broad_update = [0] * 20
    broad_update[1 * 5 + 2] = 50
    broad_update[2 * 5 + 3] = 100
    cache.cache_ros_update({
        "header": {"frame_id": "map", "stamp": {"sec": 2}},
        "x": 0,
        "y": 0,
        "width": 5,
        "height": 4,
        "data": broad_update,
    })

    _, payload = cache.serialized_if_changed(full_marker)
    assert json.loads(payload)["update"] == {
        "header": {"frame_id": "map", "stamp": {"sec": 2}},
        "x": 2,
        "y": 1,
        "width": 2,
        "height": 2,
        "data": [50, 0, 0, 100],
    }


def test_navigation_grid_websocket_sends_cached_original_topic(monkeypatch):
    cache = navigation_grid_cache.OccupancyGridCache("/map")
    cache.cache_ros_message({"info": {"width": 2}, "data": [0, 100]})
    monkeypatch.setitem(navigation_grid_cache.GRID_CACHES, "/map", cache)

    started = []
    monkeypatch.setattr(
        navigation,
        "ensure_ros_grid_subscriber_started",
        lambda: started.append(True),
    )

    class FakeWebSocket:
        def __init__(self):
            self.accepted = False
            self.messages = []

        async def accept(self):
            self.accepted = True

        async def send_text(self, payload):
            self.messages.append(json.loads(payload))

        async def receive(self):
            return {"type": "websocket.disconnect"}

    websocket = FakeWebSocket()
    asyncio.run(asyncio.wait_for(
        navigation.navigation_grid_websocket(websocket, "/map"),
        timeout=1.0,
    ))

    assert websocket.accepted is True
    assert started == [True]
    assert websocket.messages == [{
        "available": True,
        "data": {"info": {"width": 2}, "data": [0, 100]},
    }]


def test_navigation_ros_exec_environment_matches_server(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    assert navigation._ros_exec_environment() == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_goal_passes_ros_environment(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "Goal accepted"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.send_goal(
        navigation.NavigateGoalRequest(
            pose={
                "header": {"frame_id": "map"},
                "pose": {
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "orientation": {
                        "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0,
                    },
                },
            }
        )
    )

    assert result.ok
    assert captured["command"][:4] == [
        "bash", "--noprofile", "--norc", "-c"
    ]
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_initial_pose_publishes_from_ai_worker(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "Published initial pose"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setattr(navigation, "_initialpose_subscription_count", lambda: 1)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.send_initial_pose(
        navigation.InitialPoseRequest(x=1.25, y=-0.5, yaw=0.75)
    )

    assert result.ok
    command_text = captured["command"][-1]
    assert "python3 -c" in command_text
    assert "/initialpose" in command_text
    assert "/request_nomotion_update" in command_text
    assert "Duration(seconds=0.2)" in command_text
    assert "PoseWithCovarianceStamped" in command_text
    assert "std_srvs.srv import Empty" in command_text
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_initial_pose_filters_zenoh_warning(monkeypatch):
    def fake_exec(command, *, environment=None, timeout=None):
        return 0, (
            "\x1b[2m2026-07-16T02:40:40.578237Z\x1b[0m "
            "\x1b[33m WARN\x1b[0m zenoh: Scouting delay elapsed\n"
            "Published initial pose to /initialpose "
            "(-2.351, 0.168, yaw=3.142, subscribers=1)"
        )

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setattr(navigation, "_initialpose_subscription_count", lambda: 1)

    result = navigation.send_initial_pose(
        navigation.InitialPoseRequest(x=-2.351, y=0.168, yaw=3.142)
    )

    assert result.ok
    assert result.message == (
        "Published initial pose to /initialpose "
        "(-2.351, 0.168, yaw=3.142, subscribers=1)"
    )
    assert "WARN" not in result.message


def test_navigation_nomotion_update_calls_amcl_service(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "response:\nstd_srvs.srv.Empty_Response()"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.request_nomotion_update()

    assert result.ok
    command_text = captured["command"][-1]
    assert "ros2 service call" in command_text
    assert "/request_nomotion_update" in command_text
    assert "std_srvs/srv/Empty" in command_text
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_global_localization_calls_amcl_service(monkeypatch):
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        captured["environment"] = environment
        return 0, "response:\nstd_srvs.srv.Empty_Response()"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.request_global_localization()

    assert result.ok
    command_text = captured["command"][-1]
    assert "ros2 service call" in command_text
    assert "/reinitialize_global_localization" in command_text
    assert "std_srvs/srv/Empty" in command_text
    assert captured["environment"] == {
        "ROS_DOMAIN_ID": "30",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def test_navigation_design_localization_sets_amcl_parameters(monkeypatch):
    captured = []

    def fake_exec(command, *, environment=None, timeout=None):
        captured.append((command, environment))
        return 0, "Set parameter successful"

    monkeypatch.setattr(navigation, "_exec", fake_exec)
    monkeypatch.setenv("ROS_DOMAIN_ID", "30")
    monkeypatch.setenv("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    result = navigation.set_design_localization_amcl_parameters()

    assert result.ok
    assert "laser_likelihood_max_dist=2.0" in result.message
    assert "max_beams=80" in result.message
    assert "resample_interval=1" in result.message
    command_text = "\n".join(command[-1] for command, _environment in captured)
    assert "ros2 param set /amcl laser_likelihood_max_dist 2.0" in command_text
    assert "ros2 param set /amcl max_beams 80" in command_text
    assert "ros2 param set /amcl resample_interval 1" in command_text
    assert {tuple(environment.items()) for _command, environment in captured} == {
        (
            ("ROS_DOMAIN_ID", "30"),
            ("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        )
    }


def test_navigation_initial_pose_starts_localization_when_amcl_missing(monkeypatch):
    calls = []

    monkeypatch.setattr(
        navigation,
        "_initialpose_subscription_count",
        lambda: calls.append("count") or 0,
    )
    monkeypatch.setattr(
        navigation,
        "_start_localization_mode",
        lambda map_name: calls.append(("localize", map_name)) or "started",
    )
    monkeypatch.setattr(
        navigation,
        "_publish_initial_pose",
        lambda request: calls.append(("publish", request.map_name)) or "published",
    )

    result = navigation.send_initial_pose(
        navigation.InitialPoseRequest(
            x=1.0,
            y=2.0,
            yaw=0.5,
            map_name="factory",
        )
    )

    assert result.ok
    assert result.message == "published"
    assert calls == ["count", ("localize", "factory"), ("publish", "factory")]


def _container_with_mounts(*destinations):
    return SimpleNamespace(
        attrs={
            "Mounts": [
                {"Destination": destination}
                for destination in destinations
            ]
        }
    )


def test_missing_required_mounts_reports_stale_groot_container():
    container = _container_with_mounts("/legacy_model_mount/groot")

    assert _missing_required_mounts("groot", container) == list(_GROOT_REQUIRED_MOUNTS)


def test_missing_required_mounts_accepts_current_groot_container():
    container = _container_with_mounts(*_GROOT_REQUIRED_MOUNTS)

    assert _missing_required_mounts("groot", container) == []


def test_missing_required_mounts_accepts_current_lerobot_container():
    container = _container_with_mounts(*_LEROBOT_REQUIRED_MOUNTS)

    assert _missing_required_mounts("lerobot", container) == []


def test_backend_container_image_mismatch_detects_old_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.5-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:old"})
    spec = {"image": "robotis/groot-zenoh:1.3.5-arm64"}

    assert _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_image_mismatch_accepts_current_container_image():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.5-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(attrs={"Image": "sha256:new"})
    spec = {"image": "robotis/groot-zenoh:1.3.5-arm64"}

    assert not _backend_container_image_mismatch(
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
    )


def test_backend_container_stale_reason_detects_workspace_mount_mismatch():
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.5-arm64"
            return SimpleNamespace(id="sha256:new")

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/home/robot/old_workspace",
                },
                *[
                    {"Destination": destination}
                    for destination in _GROOT_REQUIRED_MOUNTS
                    if destination != "/workspace"
                ],
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.5-arm64"}

    assert _backend_container_stale_reason(
        "groot",
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
        "/mnt/ssd/cyclo_intelligence/workspace",
    ) == "workspace_mount_mismatch"


def test_backend_container_stale_reason_accepts_repo_symlink_workspace_mount(
    monkeypatch,
    tmp_path,
):
    class FakeImages:
        def get(self, image):
            assert image == "robotis/groot-zenoh:1.3.5-arm64"
            return SimpleNamespace(id="sha256:new")

    host_repo = tmp_path / "host_repo"
    container_repo = tmp_path / "container_repo"
    ssd_workspace = tmp_path / "ssd" / "cyclo_intelligence" / "workspace"
    (host_repo / "docker").mkdir(parents=True)
    (container_repo / "docker").mkdir(parents=True)
    ssd_workspace.mkdir(parents=True)
    (container_repo / "docker" / "workspace").symlink_to(ssd_workspace)

    monkeypatch.setattr(app, "_HOST_PROJECT_DIR_CACHE", str(host_repo / "docker"))
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(container_repo))

    container = SimpleNamespace(
        attrs={
            "Image": "sha256:new",
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": str(host_repo / "docker" / "workspace"),
                },
                *[
                    {"Destination": destination}
                    for destination in _GROOT_REQUIRED_MOUNTS
                    if destination != "/workspace"
                ],
            ],
        }
    )
    spec = {"image": "robotis/groot-zenoh:1.3.5-arm64"}

    assert _backend_container_stale_reason(
        "groot",
        SimpleNamespace(images=FakeImages()),
        container,
        spec,
        str(ssd_workspace),
    ) is None


@pytest.mark.parametrize("source_kind", ["relative", "absolute"])
@pytest.mark.parametrize("wrong_checkout", [False, True])
@pytest.mark.parametrize("destination", [
    "/app/configs/inference_inputs", "/app/lerobot_engine",
    "/policy_runtime", "/catalog", "/robot_client_sdk", "/orchestrator_config",
])
def test_backend_config_mount_source_validation(
    monkeypatch, tmp_path, source_kind, wrong_checkout, destination,
):
    project_dir = tmp_path / "repo" / "docker"
    expected = tmp_path / "repo" / "configs"
    expected.mkdir(parents=True)
    source = "../configs" if source_kind == "relative" else str(expected)
    actual = tmp_path / "other_repo" / "configs" if wrong_checkout else expected
    monkeypatch.setattr(app, "_host_project_dir", lambda: str(project_dir))
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(tmp_path / "repo"))
    container = _container_with_mounts(*app._REQUIRED_BACKEND_MOUNTS["lerobot"])
    for mount in container.attrs["Mounts"]:
        if mount["Destination"] == destination:
            mount["Source"] = str(actual)
    spec = {"config_mounts": {destination: source}}
    reason = (
        "config_mount_source_mismatch=" + destination if wrong_checkout else None
    )
    assert _backend_container_stale_reason("lerobot", None, container, spec, None) == reason
    # Editing the same mounted file must not require recreating the Worker.
    (expected / "act.yaml").write_text(
        "backend: torch\noperations: [{type: identity}]\n"
    )
    assert _backend_container_stale_reason("lerobot", None, container, spec, None) == reason


def test_backend_config_mount_accepts_symlink(monkeypatch, tmp_path):
    host_repo = tmp_path / "host_repo"
    container_repo = tmp_path / "container_repo"
    real_configs = tmp_path / "ssd" / "configs"
    container_repo.mkdir()
    real_configs.mkdir(parents=True)
    (container_repo / "configs").symlink_to(real_configs)
    monkeypatch.setattr(app, "_host_project_dir", lambda: str(host_repo / "docker"))
    monkeypatch.setattr(app, "_CYCLO_REPO_MOUNT", str(container_repo))
    destination = "/app/configs/inference_inputs"
    container = _container_with_mounts(*app._REQUIRED_BACKEND_MOUNTS["lerobot"])
    for mount in container.attrs["Mounts"]:
        if mount["Destination"] == destination:
            mount["Source"] = str(real_configs)
    spec = {"config_mounts": {destination: "../configs"}}
    assert _backend_container_stale_reason("lerobot", None, container, spec, None) is None


def test_backend_config_mount_requires_known_host_project(monkeypatch):
    monkeypatch.setattr(app, "_host_project_dir", lambda: None)
    destination = "/app/configs/inference_inputs"
    container = _container_with_mounts(*app._REQUIRED_BACKEND_MOUNTS["lerobot"])
    spec = {"config_mounts": {destination: "../configs"}}
    assert _backend_container_stale_reason("lerobot", None, container, spec, None) == (
        "config_mount_source_unverified=" + destination
    )


def test_backend_config_mount_is_read_from_compose():
    for backend in ("lerobot", "groot"):
        expected = {
            f"/app/{backend}_engine": f"../cyclo_brain/policy/{backend}/{backend}_engine",
            "/app/policy_manifest.yaml": f"../cyclo_brain/policy/{backend}/manifest.yaml",
            "/policy_runtime": "../cyclo_brain/policy/common/runtime",
            "/catalog": "../cyclo_brain/policy/common/catalog",
            "/zenoh_sdk": "../cyclo_brain/sdk/zenoh_ros2_sdk",
            "/robot_client_sdk": "../cyclo_brain/sdk/robot_client",
            "/orchestrator_config": "../shared/shared/robot_configs",
        }
        if backend == "lerobot":
            expected["/app/extensions"] = "../cyclo_brain/policy/lerobot/extensions"
            expected["/app/configs/inference_inputs"] = (
                "../cyclo_brain/policy/lerobot/configs/inference_inputs"
            )
        assert _BACKENDS[backend]["config_mounts"] == expected


def test_mount_source_for_destination_resolves_workspace_host_path():
    mounts = [
        {"Destination": "/root/ros2_ws/src/cyclo_intelligence", "Source": "/repo"},
        {"Destination": "/workspace", "Source": "/mnt/ssd/cyclo_intelligence/workspace"},
    ]

    assert _mount_source_for_destination(mounts, "/workspace") == (
        "/mnt/ssd/cyclo_intelligence/workspace"
    )


def test_host_workspace_dir_prefers_actual_mount_over_legacy_env(monkeypatch):
    container = SimpleNamespace(
        attrs={
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Source": "/repo/docker/workspace",
                }
            ]
        }
    )
    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _name: container)
    )

    monkeypatch.setenv("HOSTNAME", "self")
    monkeypatch.setenv(
        "CYCLO_WORKSPACE_DIR",
        "/mnt/ssd/cyclo_intelligence/workspace",
    )
    monkeypatch.setattr(app, "_docker_client", lambda: client)
    app._HOST_WORKSPACE_DIR_CACHE = None
    try:
        assert _host_workspace_dir() == "/repo/docker/workspace"
    finally:
        app._HOST_WORKSPACE_DIR_CACHE = None


def test_resolve_groot_trt_paths_defaults_engine_inside_model():
    model, engine = _resolve_groot_trt_paths(
        "/workspace/model/groot/example",
        "",
    )

    assert model == "/workspace/model/groot/example"
    assert engine == "/workspace/model/groot/example/dit_model_bf16.trt"


def test_trt_status_reports_ready_engine(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"
    engine.write_bytes(b"engine")

    status = _trt_status(str(model), str(engine))

    assert status.status == "ready"
    assert status.engine_size_bytes == len(b"engine")


def test_trt_status_reports_missing_engine(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"

    status = _trt_status(str(model), str(engine))

    assert status.status == "missing"


def test_trt_status_reports_stale_oom_build_from_log(tmp_path):
    model = tmp_path / "workspace" / "model" / "groot" / "example"
    model.mkdir(parents=True)
    engine = model / "dit_model_bf16.trt"
    (model / "dit_model_bf16.trt.json").write_text(
        '{"status": "building", "started_at": 1.0, "updated_at": 2.0}'
    )
    (model / "dit_model_bf16.trt.build.log").write_text(
        "=== TensorRT build exited rc=137 at 2026-06-19 06:29:02 ===\n"
    )

    status = _trt_status(str(model), str(engine))

    assert status.status == "failed"
    assert status.returncode == 137
    assert "out-of-memory" in status.message


def test_compose_uses_repo_local_workspace_mounts():
    compose = (REPO_ROOT / "docker" / "docker-compose.yml").read_text()

    assert "CYCLO_WORKSPACE_DIR" not in compose
    assert "CYCLO_HUGGINGFACE_DIR" not in compose
    assert compose.count("./workspace:/workspace") == 6
    assert compose.count("./huggingface:/root/.cache/huggingface") == 6


def test_container_helper_does_not_export_workspace_mount_overrides():
    helper = (REPO_ROOT / "docker" / "container.sh").read_text()

    assert "export CYCLO_WORKSPACE_DIR" not in helper
    assert "export CYCLO_HUGGINGFACE_DIR" not in helper
    assert "CYCLO_SSD_ROOT" not in helper
    assert "CYCLO_STORAGE_MODE" not in helper
    assert "setup_storage" not in helper
    assert "prepare_host_mounts" in helper
    assert "rsync " not in helper
    assert "rsync -aHP" not in helper


def test_bt_node_is_known_user_service():
    _require_known_service("bt_node")


def test_bt_node_robot_type_file_is_written(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))

    _write_bt_robot_type("ffw_sg2_rev1")

    assert target.read_text() == "ffw_sg2_rev1\n"


def test_bt_node_robot_type_defaults_to_sg2():
    assert _validate_bt_robot_type("") == "ffw_sg2_rev1"


def test_bt_node_robot_type_rejects_other_robots():
    try:
        _validate_bt_robot_type("omy_f3m")
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("bt_node should reject unsupported robot types")


def test_bt_node_start_defaults_to_sg2(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(rc=0, stdout="started", stderr="")

    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))
    monkeypatch.setattr(app, "_run", fake_run)

    result = asyncio.run(app.service_start("bt_node"))

    assert result.ok is True
    assert target.read_text() == "ffw_sg2_rev1\n"
    assert calls == [("s6-rc", "-u", "change", "bt_node")]


def test_bt_node_start_rejects_other_robots(monkeypatch, tmp_path):
    target = tmp_path / "bt_node_robot_type"
    calls = []

    async def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(rc=0, stdout="started", stderr="")

    monkeypatch.setattr(app, "_BT_ROBOT_TYPE_FILE", str(target))
    monkeypatch.setattr(app, "_run", fake_run)

    try:
        asyncio.run(app.service_start(
            "bt_node",
            app.ServiceActionRequest(robot_type="omy_f3m"),
        ))
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("bt_node should reject unsupported robot types")

    assert not target.exists()
    assert calls == []


def test_robot_type_validation_rejects_shell_metacharacters():
    try:
        _validate_robot_type("omy_f3m;echo bad")
    except app.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("invalid robot_type should be rejected")


def test_unknown_user_service_is_rejected():
    try:
        _require_known_service("not_a_service")
    except app.HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("unknown service should be rejected")


def test_zenoh_router_is_not_user_managed_service():
    assert "zenoh_router" not in _USER_SERVICES


def test_groot_backend_uses_current_release_image():
    assert (
        _BACKENDS["groot"]["image"]
        == f"robotis/groot-zenoh:1.3.5-{app._BACKEND_ARCH}"
    )


def test_policy_catalog_endpoint_exposes_validated_runtimes():
    catalog = asyncio.run(app.policy_catalog())

    assert catalog["schema_version"] == 1
    assert {runtime["id"] for runtime in catalog["runtimes"]} == {
        "groot",
        "lerobot",
        "rldx",
        "lingbot_vla",
        "abot",
    }
    assert "lerobot:act" in {
        model["policy_id"]
        for runtime in catalog["runtimes"]
        for model in runtime["models"]
    }


def test_compose_configuration_uses_override_and_arch(monkeypatch, tmp_path):
    compose_path = tmp_path / "docker-compose.yml"
    override_path = tmp_path / "docker-compose.override.yml"
    compose_path.write_text("services: {}\n", encoding="utf-8")
    override_path.write_text("services: {}\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout='{"services":{"sample":{}}}',
            stderr="",
        )

    monkeypatch.setattr(app, "_catalog_compose_path", lambda: compose_path)
    monkeypatch.setattr(app.subprocess, "run", fake_run)

    assert _load_compose_configuration() == {"services": {"sample": {}}}
    assert captured["command"] == [
        "docker",
        "compose",
        "-f",
        str(compose_path),
        "-f",
        str(override_path),
        "config",
        "--format",
        "json",
        "--no-path-resolution",
    ]
    assert captured["env"]["ARCH"] == app._BACKEND_ARCH


def test_backend_status_model_exposes_stale_image_status():
    status = app.BackendStatus(
        name="groot",
        image="robotis/groot-zenoh:1.3.5-arm64",
        image_pulled=True,
        image_status="stale",
        container_state="exited",
        raw_state="stale_image",
    )

    assert status.image_status == "stale"


def test_backend_status_preserves_readiness_and_real_errors(monkeypatch):
    container = SimpleNamespace(id="worker-container")
    client = SimpleNamespace(containers=SimpleNamespace(get=lambda _: container))
    monkeypatch.setattr(app, "_docker_client", lambda: client)
    monkeypatch.setattr(app, "_local_backend_image", lambda *_: "local-image")
    monkeypatch.setattr(app, "_host_workspace_dir", lambda: "/workspace")
    monkeypatch.setattr(app, "_backend_container_stale_reason", lambda *_: "")
    monkeypatch.setattr(app, "_container_raw_state", lambda _: "running")
    monkeypatch.setattr(app, "_backend_service_statuses", lambda *_: [
        app.ServiceStatus(name="engine-process", state="up", raw="up")
    ])
    for state in ("waiting", "error"):
        monkeypatch.setattr(app, "_runtime_control_request", lambda _, state=state: {
            "ok": True, "readiness": state, "message": "Worker response pending",
        })
        status = asyncio.run(app.backend_status("lerobot"))
        assert status.worker_readiness == state
        assert status.worker_compatible is False
        assert status.worker_message == "Worker response pending"

    def incompatible(_):
        raise RuntimeError("Worker protocol incompatible")

    monkeypatch.setattr(app, "_runtime_control_request", incompatible)
    status = asyncio.run(app.backend_status("lerobot"))
    assert status.worker_readiness == "error"
    assert status.worker_compatible is False
    assert status.worker_message == "Worker protocol incompatible"

    monkeypatch.setattr(app, "_runtime_control_request", lambda _: {
        "ok": True, "heartbeat_age_s": 0.1, "engine_state": "unloaded",
    })
    assert asyncio.run(app.backend_status("lerobot")).worker_compatible is True

    def missing_runtime(_):
        raise app.PolicyRuntimeUnavailable("Cyclo Policy Runtime is not running or is starting")

    monkeypatch.setattr(app, "_runtime_control_request", missing_runtime)
    status = asyncio.run(app.backend_status("lerobot"))
    assert status.worker_readiness == "waiting"
    assert status.worker_compatible is False
    with pytest.raises(app.HTTPException) as exc:
        app._begin_worker_mutation("lerobot")
    assert exc.value.status_code == 503


def test_absent_runtime_socket_has_distinct_error(tmp_path, monkeypatch):
    monkeypatch.setenv("POLICY_RUNTIME_CONTROL_SOCKET", str(tmp_path / "missing.sock"))
    with pytest.raises(app.PolicyRuntimeUnavailable, match="Start cyclo_intelligence"):
        app._runtime_control_request({"operation": "status"})


def test_backend_start_keeps_running_container(monkeypatch):
    class FakeContainer:
        def __init__(self):
            self.started = False
            self.restarted = False

        def start(self):
            self.started = True

        def restart(self, timeout):
            self.restarted = timeout

    container = FakeContainer()
    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _name: container),
    )
    monkeypatch.setattr(app, "_docker_client", lambda: client)
    monkeypatch.setattr(app, "_host_workspace_dir", lambda: None)
    monkeypatch.setattr(
        app,
        "_backend_container_stale_reason",
        lambda *_args: None,
    )
    monkeypatch.setattr(app, "_container_raw_state", lambda _ctr: "running")
    control_calls = []
    monkeypatch.setattr(
        app,
        "_runtime_control_request",
        lambda payload: control_calls.append(payload) or {
            "ok": True,
            "allowed": True,
            "token": "lease-1",
        },
    )

    result = asyncio.run(app.backend_start("lerobot"))

    assert result.ok is True
    assert result.message == "lerobot_server already running"
    assert container.started is False
    assert container.restarted is False
    assert [call["operation"] for call in control_calls] == [
        "begin_worker_mutation",
        "end_worker_mutation",
    ]


def test_backend_restart_restarts_running_container(monkeypatch):
    class FakeContainer:
        def __init__(self):
            self.restart_timeout = None

        def restart(self, timeout):
            self.restart_timeout = timeout

    container = FakeContainer()
    client = SimpleNamespace(
        containers=SimpleNamespace(get=lambda _name: container),
    )
    monkeypatch.setattr(app, "_docker_client", lambda: client)
    monkeypatch.setattr(app, "_host_workspace_dir", lambda: None)
    monkeypatch.setattr(
        app,
        "_backend_container_stale_reason",
        lambda *_args: None,
    )
    monkeypatch.setattr(app, "_container_raw_state", lambda _ctr: "running")
    control_calls = []
    monkeypatch.setattr(
        app,
        "_runtime_control_request",
        lambda payload: control_calls.append(payload) or {
            "ok": True,
            "allowed": True,
            "token": "lease-2",
        },
    )

    result = asyncio.run(app.backend_restart("lerobot"))

    assert result.ok is True
    assert result.message == "lerobot_server restarted"
    assert container.restart_timeout == 10
    assert [call["operation"] for call in control_calls] == [
        "begin_worker_mutation",
        "end_worker_mutation",
    ]


def test_backend_start_releases_worker_lease_when_start_fails(monkeypatch):
    async def fail_start(*_args, **_kwargs):
        raise RuntimeError("start failed")

    control_calls = []
    monkeypatch.setattr(app, "_ensure_backend_running", fail_start)
    monkeypatch.setattr(
        app,
        "_runtime_control_request",
        lambda payload: control_calls.append(payload) or {
            "ok": True,
            "allowed": True,
            "token": "lease-failure",
        },
    )

    try:
        asyncio.run(app.backend_start("lerobot"))
    except RuntimeError as exc:
        assert str(exc) == "start failed"
    else:
        raise AssertionError("backend_start should propagate the start failure")

    assert [call["operation"] for call in control_calls] == [
        "begin_worker_mutation",
        "end_worker_mutation",
    ]


def test_backend_stop_uses_worker_lease(monkeypatch):
    class FakeContainer:
        def __init__(self):
            self.stop_timeout = None

        def stop(self, timeout):
            self.stop_timeout = timeout

    container = FakeContainer()
    monkeypatch.setattr(
        app,
        "_docker_client",
        lambda: SimpleNamespace(
            containers=SimpleNamespace(get=lambda _name: container),
        ),
    )
    monkeypatch.setattr(app, "_container_raw_state", lambda _ctr: "running")
    control_calls = []
    monkeypatch.setattr(
        app,
        "_runtime_control_request",
        lambda payload: control_calls.append(payload) or {
            "ok": True,
            "allowed": True,
            "token": "lease-stop",
        },
    )

    result = asyncio.run(app.backend_stop("lerobot"))

    assert result.ok is True
    assert container.stop_timeout == 10
    assert [call["operation"] for call in control_calls] == [
        "begin_worker_mutation",
        "end_worker_mutation",
    ]


def test_begin_worker_mutation_rejects_active_worker(monkeypatch):
    monkeypatch.setattr(
        app,
        "_runtime_control_request",
        lambda _payload: {
            "ok": True,
            "allowed": False,
            "reason": "UNLOAD the active policy before changing its worker",
        },
    )

    try:
        app._begin_worker_mutation("lerobot")
    except app.HTTPException as exc:
        assert exc.status_code == 409
        assert "UNLOAD" in exc.detail
    else:
        raise AssertionError("active worker mutation should be rejected")


def test_worker_lifecycle_fails_closed_when_control_socket_is_unavailable(monkeypatch):
    import pytest

    def unavailable(_payload):
        raise RuntimeError("Policy Runtime control socket unavailable: timed out")

    def forbidden_docker():
        raise AssertionError("Docker must not be called without safety approval")

    monkeypatch.setattr(app, "_runtime_control_request", unavailable)
    monkeypatch.setattr(app, "_docker_client", forbidden_docker)
    for endpoint in (app.backend_start, app.backend_restart, app.backend_stop,
                     app.backend_recreate):
        with pytest.raises(app.HTTPException) as failure:
            asyncio.run(endpoint("lerobot"))
        assert failure.value.status_code == 503
        assert "safety check unavailable" in failure.value.detail
        assert "timed out" in failure.value.detail


def test_runtime_control_rejects_invalid_responses_and_closes_socket(monkeypatch):
    import pytest

    class FakeSocket:
        def __init__(self, response):
            self.response = response
            self.closed = False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendall(self, _data):
            pass

        def shutdown(self, _how):
            pass

        def recv(self, _size):
            response, self.response = self.response, b""
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            self.closed = True

    for response in (b"[]", b"null", b"broken", b'{"ok":false,"error":"busy"}',
                     TimeoutError("timed out")):
        client = FakeSocket(response)
        monkeypatch.setattr(app.socket, "socket", lambda *_args: client)
        with pytest.raises(RuntimeError):
            app._runtime_control_request({"operation": "status"})
        assert client.closed


def test_host_project_dir_falls_back_to_compose_container_name(monkeypatch):
    class FakeContainers:
        def __init__(self):
            self.requested = []

        def get(self, name):
            self.requested.append(name)
            if name == "cyclo_intelligence":
                return SimpleNamespace(
                    attrs={
                        "Mounts": [
                            {
                                "Destination": app._CYCLO_REPO_MOUNT,
                                "Source": "/home/rc/workspace/cyclo_intelligence",
                            }
                        ]
                    }
                )
            raise NotFound(name)

    fake_containers = FakeContainers()
    fake_client = SimpleNamespace(containers=fake_containers)

    monkeypatch.setenv("HOSTNAME", "ubuntu")
    monkeypatch.setattr(app, "_docker_client", lambda: fake_client)
    app._HOST_PROJECT_DIR_CACHE = None

    try:
        assert (
            app._host_project_dir()
            == "/home/rc/workspace/cyclo_intelligence/docker"
        )
        assert fake_containers.requested == ["ubuntu", "cyclo_intelligence"]
    finally:
        app._HOST_PROJECT_DIR_CACHE = None


def test_compose_env_uses_current_container_mounts(monkeypatch):
    class FakeContainers:
        def __init__(self):
            self.requested = []

        def get(self, name):
            self.requested.append(name)
            if name != "cyclo_intelligence":
                raise NotFound(name)
            return SimpleNamespace(
                attrs={
                    "Mounts": [
                        {
                            "Destination": "/workspace",
                            "Source": "/mnt/ssd/cyclo_intelligence/workspace",
                        },
                        {
                            "Destination": "/root/.cache/huggingface",
                            "Source": "/mnt/ssd/cyclo_intelligence/huggingface",
                        },
                    ]
                }
            )

    fake_containers = FakeContainers()
    fake_client = SimpleNamespace(containers=fake_containers)

    monkeypatch.setenv("HOSTNAME", "container-id")
    monkeypatch.delenv("CYCLO_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("CYCLO_HUGGINGFACE_DIR", raising=False)
    monkeypatch.setattr(app, "_docker_client", lambda: fake_client)
    app._HOST_WORKSPACE_DIR_CACHE = None
    app._HOST_HUGGINGFACE_DIR_CACHE = None

    try:
        env = _compose_env()
        assert (
            env["CYCLO_WORKSPACE_DIR"]
            == "/mnt/ssd/cyclo_intelligence/workspace"
        )
        assert (
            env["CYCLO_HUGGINGFACE_DIR"]
            == "/mnt/ssd/cyclo_intelligence/huggingface"
        )
        assert env["ARCH"] == app._BACKEND_ARCH
        assert fake_containers.requested == [
            "container-id",
            "cyclo_intelligence",
            "container-id",
            "cyclo_intelligence",
        ]
    finally:
        app._HOST_WORKSPACE_DIR_CACHE = None
        app._HOST_HUGGINGFACE_DIR_CACHE = None


def _navigate_goal_request():
    return navigation.NavigateGoalRequest(
        pose={
            "header": {"frame_id": "map"},
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
            },
        }
    )


def _navigate_through_poses_request(count=2):
    pose = _navigate_goal_request().pose
    return navigation.NavigateThroughPosesRequest(
        poses=[pose for _index in range(count)]
    )


@pytest.mark.parametrize("pose_count", [1, 4])
def test_navigation_through_poses_request_rejects_invalid_batch_size(
    pose_count,
):
    with pytest.raises(ValueError):
        _navigate_through_poses_request(pose_count)


def test_navigation_goal_wait_returns_succeeded(monkeypatch):
    output = "Goal accepted with ID: abc\nGoal finished with status: SUCCEEDED"
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        return 0, output

    monkeypatch.setattr(navigation, "_exec", fake_exec)

    result = navigation.send_goal_and_wait(_navigate_goal_request())

    assert result.ok is True
    assert result.status == "SUCCEEDED"
    assert "timeout 120s ros2 action send_goal" in captured["command"][-1]


def test_navigation_goal_wait_returns_terminal_failure(monkeypatch):
    output = "Goal accepted with ID: abc\nGoal finished with status: ABORTED"
    monkeypatch.setattr(
        navigation,
        "_exec",
        lambda command, *, environment=None, timeout=None: (0, output),
    )

    result = navigation.send_goal_and_wait(_navigate_goal_request())

    assert result.ok is False
    assert result.status == "ABORTED"


def test_navigation_goal_wait_cancels_after_timeout(monkeypatch):
    cancelled = []
    monkeypatch.setattr(
        navigation,
        "_exec",
        lambda command, *, environment=None, timeout=None: (
            124,
            "Goal accepted with ID: abc",
        ),
    )
    monkeypatch.setattr(
        navigation,
        "_cancel_all_navigate_goals",
        lambda: cancelled.append(True) or "Goals cancelled",
    )

    result = navigation.send_goal_and_wait(_navigate_goal_request())

    assert result.ok is False
    assert result.status == "TIMEOUT"
    assert cancelled == [True]


def test_navigation_goal_wait_reports_rejection(monkeypatch):
    monkeypatch.setattr(
        navigation,
        "_exec",
        lambda command, *, environment=None, timeout=None: (
            1,
            "Goal was rejected.",
        ),
    )

    result = navigation.send_goal_and_wait(_navigate_goal_request())

    assert result.ok is False
    assert result.status == "REJECTED"


@pytest.mark.parametrize(
    ("pose_count", "expected_timeout"),
    [(2, 240), (3, 360)],
)
def test_navigation_goals_wait_scales_timeout(
    monkeypatch,
    pose_count,
    expected_timeout,
):
    output = "Goal accepted with ID: abc\nGoal finished with status: SUCCEEDED"
    captured = {}

    def fake_exec(command, *, environment=None, timeout=None):
        captured["command"] = command
        return 0, output

    monkeypatch.setattr(navigation, "_exec", fake_exec)

    result = navigation.send_goals_and_wait(
        _navigate_through_poses_request(pose_count)
    )

    assert result.ok is True
    assert result.status == "SUCCEEDED"
    command = captured["command"][-1]
    assert (
        f"timeout {expected_timeout}s ros2 action send_goal "
        "/navigate_through_poses nav2_msgs/action/NavigateThroughPoses"
    ) in command
    assert '"poses":' in command


def test_navigation_goals_wait_cancels_through_poses_after_timeout(monkeypatch):
    cancelled = []
    monkeypatch.setattr(
        navigation,
        "_exec",
        lambda command, *, environment=None, timeout=None: (
            124,
            "Goal accepted with ID: abc",
        ),
    )
    monkeypatch.setattr(
        navigation,
        "_cancel_all_navigate_through_poses_goals",
        lambda: cancelled.append(True) or "Goals cancelled",
    )

    result = navigation.send_goals_and_wait(
        _navigate_through_poses_request()
    )

    assert result.ok is False
    assert result.status == "TIMEOUT"
    assert cancelled == [True]


@pytest.mark.parametrize(
    "output",
    [
        "Goal accepted with ID: abc\nGoal finished with status: ABORTED",
        "Goal accepted with ID: abc\nGoal finished with status: CANCELED",
        "Goal was rejected.",
    ],
)
def test_navigation_goals_wait_returns_terminal_failure(monkeypatch, output):
    monkeypatch.setattr(
        navigation,
        "_exec",
        lambda command, *, environment=None, timeout=None: (
            1 if "rejected" in output else 0,
            output,
        ),
    )

    result = navigation.send_goals_and_wait(
        _navigate_through_poses_request()
    )

    assert result.ok is False
    assert result.status in {"ABORTED", "CANCELED", "REJECTED"}


def test_navigation_cancel_attempts_both_action_types(monkeypatch):
    calls = []
    monkeypatch.setattr(
        navigation,
        "_cancel_all_navigate_goals",
        lambda: calls.append("pose") or "NavigateToPose cancelled",
    )
    monkeypatch.setattr(
        navigation,
        "_cancel_all_navigate_through_poses_goals",
        lambda: calls.append("through") or "NavigateThroughPoses cancelled",
    )

    result = navigation.cancel_goal()

    assert result.ok is True
    assert calls == ["pose", "through"]
    assert "NavigateToPose cancelled" in result.message
    assert "NavigateThroughPoses cancelled" in result.message


def test_navigation_cancel_attempts_both_before_reporting_error(monkeypatch):
    calls = []

    def fail_pose_cancel():
        calls.append("pose")
        raise navigation.HTTPException(503, "pose cancel failed")

    monkeypatch.setattr(
        navigation,
        "_cancel_all_navigate_goals",
        fail_pose_cancel,
    )
    monkeypatch.setattr(
        navigation,
        "_cancel_all_navigate_through_poses_goals",
        lambda: calls.append("through") or "through cancelled",
    )

    with pytest.raises(navigation.HTTPException) as exc_info:
        navigation.cancel_goal()

    assert calls == ["pose", "through"]
    assert "pose cancel failed" in str(exc_info.value.detail)


def test_bt_support_comes_from_shared_schema():
    assert bt_support.bt_supported_robot_types() == ["ffw_sg2_rev1"]
    assert app._validate_bt_robot_type("") == "ffw_sg2_rev1"
    assert asyncio.run(bt_trees.bt_support_info()).supported_robot_types == [
        "ffw_sg2_rev1",
    ]


def test_bt_support_reports_missing_schema(monkeypatch):
    monkeypatch.setattr(bt_support, "_schema_module", None)
    monkeypatch.setenv("CYCLO_ROBOT_CONFIGS_DIR", "/nonexistent/robot_configs")
    monkeypatch.setenv("COLCON_WS", "/nonexistent/ws")
    monkeypatch.setattr(
        bt_support, "robot_configs_dir_candidates",
        lambda: [Path("/nonexistent/robot_configs")],
    )

    try:
        app._validate_bt_robot_type("")
    except app.HTTPException as exc:
        assert exc.status_code == 503
    else:
        raise AssertionError("missing schema must not silently allow bt_node")


def test_bt_trees_seed_examples_and_never_overwrite(monkeypatch, tmp_path):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "example.xml").write_text("<root/>")
    trees = tmp_path / "trees"
    monkeypatch.setenv("CYCLO_BT_TREES_DIR", str(trees))
    monkeypatch.setenv("CYCLO_BT_EXAMPLE_TREES_DIR", str(examples))
    monkeypatch.setenv("COLCON_WS", str(tmp_path / "no_ws"))

    listed = asyncio.run(bt_trees.list_trees())
    assert listed.directory == str(trees)
    assert [item.name for item in listed.trees] == ["example.xml"]

    (trees / "example.xml").write_text("<root edited='1'/>")
    listed = asyncio.run(bt_trees.list_trees())
    assert asyncio.run(bt_trees.read_tree("example.xml")).content == "<root edited='1'/>"


def test_bt_trees_save_read_and_conflict(monkeypatch, tmp_path):
    trees = tmp_path / "trees"
    monkeypatch.setenv("CYCLO_BT_TREES_DIR", str(trees))
    monkeypatch.setenv("COLCON_WS", str(tmp_path / "no_ws"))
    monkeypatch.delenv("CYCLO_BT_EXAMPLE_TREES_DIR", raising=False)

    saved = asyncio.run(bt_trees.save_tree(
        bt_trees.BtTreeSaveRequest(filename="pick-and-place", content="<root/>"),
    ))
    assert saved.ok is True
    assert saved.name == "pick-and-place.xml"
    assert saved.path == str(trees / "pick-and-place.xml")
    assert asyncio.run(bt_trees.read_tree("pick-and-place.xml")).content == "<root/>"

    try:
        asyncio.run(bt_trees.save_tree(
            bt_trees.BtTreeSaveRequest(filename="pick-and-place.xml", content="<x/>"),
        ))
    except app.HTTPException as exc:
        assert exc.status_code == 409
        assert exc.detail["code"] == "file_exists"
    else:
        raise AssertionError("saving over an existing tree must require overwrite")

    overwritten = asyncio.run(bt_trees.save_tree(
        bt_trees.BtTreeSaveRequest(filename="pick-and-place.xml", content="<x/>", overwrite=True),
    ))
    assert overwritten.ok is True
    assert asyncio.run(bt_trees.read_tree("pick-and-place")).content == "<x/>"


def test_bt_trees_reject_unsafe_names_and_missing_files(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLO_BT_TREES_DIR", str(tmp_path / "trees"))
    monkeypatch.setenv("COLCON_WS", str(tmp_path / "no_ws"))
    monkeypatch.delenv("CYCLO_BT_EXAMPLE_TREES_DIR", raising=False)

    for bad in ("", "../escape.xml", "a/b.xml", "bad name.xml"):
        try:
            asyncio.run(bt_trees.save_tree(bt_trees.BtTreeSaveRequest(filename=bad, content="")))
        except app.HTTPException as exc:
            assert exc.status_code == 400, bad
        else:
            raise AssertionError(f"{bad!r} must be rejected")

    try:
        asyncio.run(bt_trees.read_tree("missing.xml"))
    except app.HTTPException as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("missing tree must 404")


def test_bt_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}
    assert {"/bt/support", "/bt/trees", "/bt/trees/{name}"} <= paths


def test_bt_trees_seed_from_the_pre_1_4_source_directory(monkeypatch, tmp_path):
    # Releases before 1.4.0 saved trees into orchestrator/orchestrator/bt/trees
    # of the bind-mounted checkout; they must survive the upgrade.
    ws = tmp_path / "ws"
    legacy = ws / "src" / "cyclo_intelligence" / "orchestrator" / "orchestrator" / "bt" / "trees"
    legacy.mkdir(parents=True)
    (legacy / "user_task.xml").write_text("<root user='1'/>")
    (legacy / "notes.txt").write_text("ignored")
    trees = tmp_path / "trees"
    monkeypatch.setenv("CYCLO_BT_TREES_DIR", str(trees))
    monkeypatch.setenv("COLCON_WS", str(ws))
    monkeypatch.delenv("CYCLO_BT_EXAMPLE_TREES_DIR", raising=False)
    monkeypatch.delenv("CYCLO_BT_LEGACY_TREES_DIR", raising=False)

    listed = asyncio.run(bt_trees.list_trees())
    assert [item.name for item in listed.trees] == ["user_task.xml"]
    assert asyncio.run(bt_trees.read_tree("user_task.xml")).content == "<root user='1'/>"

    # The copy is one-way: later edits stay in the user directory only.
    asyncio.run(bt_trees.save_tree(
        bt_trees.BtTreeSaveRequest(filename="user_task.xml", content="<root user='2'/>", overwrite=True),
    ))
    assert (legacy / "user_task.xml").read_text() == "<root user='1'/>"
    assert asyncio.run(bt_trees.read_tree("user_task.xml")).content == "<root user='2'/>"
