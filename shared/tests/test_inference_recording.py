import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from shared.inference_recording import allocate_folder, folder_name, model_name, validate_folder


def episode(root, session='saved', robot='test'):
    path = root / folder_name(session) / '0'
    path.mkdir(parents=True)
    (path / 'episode_info.json').write_text(json.dumps({
        'robot_type': robot, 'format_version': 'robotis_v2',
    }))
    return path


def test_allocation_is_atomic_and_uses_local_time_and_model(monkeypatch, tmp_path):
    clock = object()
    monkeypatch.setattr('shared.inference_recording.time.localtime', lambda: clock)

    def format_time(fmt, tm):
        assert fmt == '%y%m%d_%H%M' and tm is clock
        return '260928_1630'

    monkeypatch.setattr('shared.inference_recording.time.strftime', format_time)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: allocate_folder(tmp_path, '/models/peanut'), range(8)))
    assert len(set(ids)) == 8
    assert '260928_1630_peanut' in ids
    assert '260928_1630_peanut_02' in ids
    assert all((tmp_path / folder_name(value)).is_dir() for value in ids)


@pytest.mark.parametrize(('path', 'expected'), [
    ('/models/peanut/', 'peanut'),
    ('/models/peanut/checkpoints/010000/pretrained_model', 'peanut'),
    ('/models/peanut model..v2', 'peanut_model_v2'),
    ('', 'unknown_model'),
    ('/models/..', 'unknown_model'),
    ('/models/' + 'a' * 200, 'a' * 140),
])
def test_model_name_is_safe(path, expected):
    assert model_name(path) == expected
    assert folder_name(f'260928_1630_{expected}') == f'260928_1630_{expected}'


@pytest.mark.parametrize('value', ['', '../escape', '/absolute', 'a/b', 'a..b'])
def test_rejects_bad_ids(value):
    with pytest.raises(ValueError):
        folder_name(value)


@pytest.mark.parametrize('session', ['saved', '260928_1630_peanut'])
def test_folder_validation(tmp_path, session):
    path = episode(tmp_path, session=session)
    assert validate_folder(tmp_path, session, 'test') == path.parent
    with pytest.raises(ValueError, match='robot_type'):
        validate_folder(tmp_path, session, 'other')
    (path / 'episode_info.json').write_text('{broken')
    with pytest.raises(ValueError, match='Cannot read'):
        validate_folder(tmp_path, session, 'test')


def test_rejects_symlinks_and_missing_metadata(tmp_path):
    path = episode(tmp_path)
    (tmp_path / folder_name('link')).symlink_to(path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        validate_folder(tmp_path, 'link', 'test')
    (path / 'linked').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='symlink'):
        validate_folder(tmp_path, 'saved', 'test')
    (path / 'linked').unlink()
    (path / 'episode_info.json').unlink()
    (path / 'partial.mcap').write_bytes(b'partial')
    with pytest.raises(ValueError, match='no readable metadata'):
        validate_folder(tmp_path, 'saved', 'test')


def test_allocation_permission_failure_is_not_silently_redirected(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError('denied')
    monkeypatch.setattr(type(tmp_path), 'mkdir', denied)
    with pytest.raises(PermissionError):
        allocate_folder(tmp_path)
