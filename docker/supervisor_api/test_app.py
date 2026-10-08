import asyncio
import importlib.util
import json
import sys
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
_GROOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["groot"]
_LEROBOT_REQUIRED_MOUNTS = app._REQUIRED_BACKEND_MOUNTS["lerobot"]


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


def test_navigation_parses_binary_pgm():
    data = b"P5\n# map\n2 2\n255\n" + bytes([0, 127, 254, 255])

    assert navigation._parse_pgm(data) == (
        2,
        2,
        255,
        [0, 127, 254, 255],
    )


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


def test_navigation_routes_are_registered():
    paths = {route.path for route in app.app.routes if hasattr(route, "path")}

    assert "/navigation/status" in paths
    assert "/navigation/start" in paths
    assert "/navigation/maps/pgm/save" in paths
    assert "/navigation/topics/ws" in paths


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
