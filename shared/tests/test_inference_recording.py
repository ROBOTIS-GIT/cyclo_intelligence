import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from shared.inference_recording import allocate_folder, folder_name, validate_folder


def episode(root, session='saved', robot='test'):
    path = root / folder_name(session) / '0'
    path.mkdir(parents=True)
    (path / 'episode_info.json').write_text(json.dumps({
        'robot_type': robot, 'format_version': 'robotis_v2',
    }))
    return path


def test_allocation_is_atomic_and_uses_utc(monkeypatch, tmp_path):
    monkeypatch.setattr('shared.inference_recording.time.strftime', lambda fmt, tm: '20260921_010203')
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: allocate_folder(tmp_path), range(8)))
    assert len(set(ids)) == 8
    assert '20260921_010203' in ids
    assert all((tmp_path / folder_name(value)).is_dir() for value in ids)


@pytest.mark.parametrize('value', ['', '../escape', '/absolute', 'a/b', 'a..b'])
def test_rejects_bad_ids(value):
    with pytest.raises(ValueError):
        folder_name(value)


def test_folder_validation(tmp_path):
    path = episode(tmp_path)
    assert validate_folder(tmp_path, 'saved', 'test') == path.parent
    with pytest.raises(ValueError, match='robot_type'):
        validate_folder(tmp_path, 'saved', 'other')
    (path / 'episode_info.json').write_text('{broken')
    with pytest.raises(ValueError, match='Cannot read'):
        validate_folder(tmp_path, 'saved', 'test')


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
