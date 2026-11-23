import pytest

from cyclo_data.converter import pipeline_worker as worker


def test_default_and_custom_output_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'LEROBOT_OUTPUT_ROOT', tmp_path)
    paths = worker.lerobot_output_dirs('/recordings/nested/task', '', True, True)
    assert paths == {v: tmp_path / f'task_lerobot_{v}' for v in ('v21', 'v30')}
    assert worker.lerobot_output_dirs('/recordings/task', 'peanut', False, True) == {
        'v30': tmp_path / 'peanut_v30'}


@pytest.mark.parametrize('name', ['../out', '/abs', 'a/b', 'a\\b', '.', '..', ' ', 'a\nb'])
def test_rejects_invalid_names(name):
    with pytest.raises(ValueError):
        worker.lerobot_output_dirs('/recordings/task', name, True, True)


def test_checks_only_selected_outputs_and_never_writes_on_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'LEROBOT_OUTPUT_ROOT', tmp_path)
    (tmp_path / 'custom_v30').mkdir()
    marker = tmp_path / 'custom_v30' / 'previous'
    marker.write_bytes(b'keep')
    assert set(worker.lerobot_output_dirs('/task', 'custom', True, False)) == {'v21'}
    with pytest.raises(ValueError, match='already exists'):
        worker.lerobot_output_dirs('/task', 'custom', True, True)
    assert marker.read_bytes() == b'keep'
    assert not (tmp_path / 'custom_v21').exists()
    (tmp_path / 'custom_v21').symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        worker.lerobot_output_dirs('/task', 'custom', True, False)
