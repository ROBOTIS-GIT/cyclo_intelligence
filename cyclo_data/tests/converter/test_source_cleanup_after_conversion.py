import json
import logging
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cyclo_data.converter import pipeline_worker
from cyclo_data.converter.pipeline_worker import (
    _delete_raw_source_episodes,
    _delete_sources_after_validated_conversion,
)


def test_output_root_defaults_and_rejects_workspace_escape(tmp_path, monkeypatch):
    allowed_root = tmp_path / 'workspace' / 'lerobot'
    allowed_root.mkdir(parents=True)
    monkeypatch.setattr(pipeline_worker, 'LEROBOT_OUTPUT_ROOT', allowed_root)

    assert pipeline_worker.resolve_lerobot_output_root('') == allowed_root
    assert pipeline_worker.resolve_lerobot_output_root(
        str(allowed_root / 'round_1')
    ) == allowed_root / 'round_1'

    with pytest.raises(ValueError, match='absolute path'):
        pipeline_worker.resolve_lerobot_output_root('round_1')
    with pytest.raises(ValueError, match='must remain within'):
        pipeline_worker.resolve_lerobot_output_root(
            str(allowed_root / '..' / 'model')
        )


def _write_raw_episode(task_dir: Path, index: int) -> None:
    episode_dir = task_dir / str(index)
    episode_dir.mkdir(parents=True)
    (episode_dir / f'{index}_0.mcap').write_bytes(b'mcap')
    (episode_dir / 'metadata.yaml').write_text(
        yaml.safe_dump(
            {
                'rosbag2_bagfile_information': {
                    'duration': {'nanoseconds': 1_000_000_000},
                }
            }
        ),
        encoding='utf-8',
    )
    (episode_dir / 'episode_info.json').write_text(
        json.dumps({'episode_index': index, 'episode_success': True}),
        encoding='utf-8',
    )


def _write_v21_output(
    output_root: Path,
    task_name: str,
    count: int,
    *,
    data_episode_indices=None,
    declare_video: bool = False,
    write_videos: bool = True,
) -> None:
    output = output_root / f'{task_name}_lerobot_v21'
    (output / 'meta').mkdir(parents=True)
    (output / 'data' / 'chunk-000').mkdir(parents=True)
    features = {}
    video_path = None
    if declare_video:
        features['observation.images.rgb.cam_left_head'] = {'dtype': 'video'}
        video_path = (
            'videos/chunk-{episode_chunk:03d}/{video_key}/'
            'episode_{episode_index:06d}.mp4'
        )
    (output / 'meta' / 'info.json').write_text(
        json.dumps({
            'total_episodes': count,
            'total_frames': count,
            'chunks_size': 1000,
            'data_path': (
                'data/chunk-{episode_chunk:03d}/'
                'episode_{episode_index:06d}.parquet'
            ),
            'video_path': video_path,
            'features': features,
        }),
        encoding='utf-8',
    )
    (output / 'meta' / 'episodes.jsonl').write_text(
        ''.join(
            json.dumps({'episode_index': index}) + '\n'
            for index in range(count)
        ),
        encoding='utf-8',
    )
    indices = (
        list(range(count))
        if data_episode_indices is None
        else list(data_episode_indices)
    )
    for episode_index in indices:
        pq.write_table(
            pa.table({'episode_index': [episode_index]}),
            output / 'data' / 'chunk-000'
            / f'episode_{episode_index:06d}.parquet',
        )
        if declare_video and write_videos:
            video = (
                output / 'videos' / 'chunk-000'
                / 'observation.images.rgb.cam_left_head'
                / f'episode_{episode_index:06d}.mp4'
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'mp4')


def _write_v30_output(
    output_root: Path,
    task_name: str,
    count: int,
    *,
    metadata_episode_indices=None,
    data_episode_indices=None,
    declare_video: bool = False,
    write_video: bool = True,
) -> None:
    output = output_root / f'{task_name}_lerobot_v30'
    (output / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (output / 'data' / 'chunk-000').mkdir(parents=True)
    features = {}
    video_path = None
    if declare_video:
        features['observation.images.rgb.cam_left_head'] = {'dtype': 'video'}
        video_path = (
            'videos/{video_key}/chunk-{chunk_index:03d}/'
            'file-{file_index:03d}.mp4'
        )
    (output / 'meta' / 'info.json').write_text(
        json.dumps({
            'total_episodes': count,
            'total_frames': count,
            'video_path': video_path,
            'features': features,
        }),
        encoding='utf-8',
    )
    metadata_indices = (
        list(range(count))
        if metadata_episode_indices is None
        else list(metadata_episode_indices)
    )
    metadata = {
        'episode_index': metadata_indices,
        'length': [1] * len(metadata_indices),
        'data/chunk_index': [0] * len(metadata_indices),
        'data/file_index': [0] * len(metadata_indices),
    }
    if declare_video:
        metadata['videos/observation.images.rgb.cam_left_head/chunk_index'] = (
            [0] * len(metadata_indices)
        )
        metadata['videos/observation.images.rgb.cam_left_head/file_index'] = (
            [0] * len(metadata_indices)
        )
    pq.write_table(
        pa.table(metadata),
        output / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet',
    )
    data_indices = (
        list(range(count))
        if data_episode_indices is None
        else list(data_episode_indices)
    )
    pq.write_table(
        pa.table({'episode_index': data_indices}),
        output / 'data' / 'chunk-000' / 'file-000.parquet',
    )
    if declare_video and write_video:
        video = (
            output / 'videos' / 'observation.images.rgb.cam_left_head'
            / 'chunk-000' / 'file-000.mp4'
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b'mp4')


def test_cleanup_deletes_only_raw_episodes_after_all_outputs_validate(tmp_path):
    task_dir = tmp_path / 'raw' / 'pick_jelly'
    _write_raw_episode(task_dir, 0)
    _write_raw_episode(task_dir, 4)
    (task_dir / 'README.md').write_text('keep me', encoding='utf-8')
    (task_dir / 'task_metadata.json').write_text('{}', encoding='utf-8')

    output_root = tmp_path / 'lerobot'
    _write_v21_output(output_root, task_dir.name, 2)
    _write_v30_output(output_root, task_dir.name, 2)

    deleted = _delete_sources_after_validated_conversion(
        task_dir,
        source_folders=[],
        convert_v21=True,
        convert_v30=True,
        logger=logging.getLogger('test-cleanup'),
        output_root=output_root,
    )

    assert deleted == 2
    assert task_dir.is_dir()
    assert (task_dir / 'README.md').read_text(encoding='utf-8') == 'keep me'
    assert (task_dir / 'task_metadata.json').is_file()
    assert not (task_dir / '0').exists()
    assert not (task_dir / '4').exists()
    assert (output_root / f'{task_dir.name}_lerobot_v21').is_dir()
    assert (output_root / f'{task_dir.name}_lerobot_v30').is_dir()


def test_validation_failure_preserves_every_raw_episode(tmp_path):
    task_dir = tmp_path / 'raw' / 'pick_jelly'
    _write_raw_episode(task_dir, 0)
    _write_raw_episode(task_dir, 1)

    output_root = tmp_path / 'lerobot'
    # Deliberately incomplete: metadata claims only one converted episode.
    _write_v21_output(output_root, task_dir.name, 1)

    with pytest.raises(RuntimeError, match='episode count mismatch'):
        _delete_sources_after_validated_conversion(
            task_dir,
            source_folders=[],
            convert_v21=True,
            convert_v30=False,
            logger=logging.getLogger('test-cleanup'),
            output_root=output_root,
        )

    assert (task_dir / '0' / '0_0.mcap').is_file()
    assert (task_dir / '1' / '1_0.mcap').is_file()


def test_v21_missing_per_episode_parquet_preserves_raw_sources(tmp_path):
    task_dir = tmp_path / 'raw' / 'pick_jelly'
    _write_raw_episode(task_dir, 0)
    _write_raw_episode(task_dir, 1)
    output_root = tmp_path / 'lerobot'
    _write_v21_output(
        output_root,
        task_dir.name,
        2,
        data_episode_indices=[0],
    )

    with pytest.raises(RuntimeError, match='missing or empty'):
        _delete_sources_after_validated_conversion(
            task_dir,
            source_folders=[],
            convert_v21=True,
            convert_v30=False,
            logger=logging.getLogger('test-cleanup'),
            output_root=output_root,
        )

    assert (task_dir / '0' / '0_0.mcap').is_file()
    assert (task_dir / '1' / '1_0.mcap').is_file()


def test_v30_incomplete_episode_metadata_preserves_raw_sources(tmp_path):
    task_dir = tmp_path / 'raw' / 'pick_jelly'
    _write_raw_episode(task_dir, 0)
    _write_raw_episode(task_dir, 1)
    output_root = tmp_path / 'lerobot'
    _write_v30_output(
        output_root,
        task_dir.name,
        2,
        metadata_episode_indices=[0],
    )

    with pytest.raises(RuntimeError, match='metadata indices'):
        _delete_sources_after_validated_conversion(
            task_dir,
            source_folders=[],
            convert_v21=False,
            convert_v30=True,
            logger=logging.getLogger('test-cleanup'),
            output_root=output_root,
        )

    assert (task_dir / '0' / '0_0.mcap').is_file()
    assert (task_dir / '1' / '1_0.mcap').is_file()


def test_v30_incomplete_data_episode_coverage_preserves_raw_sources(tmp_path):
    task_dir = tmp_path / 'raw' / 'pick_jelly'
    _write_raw_episode(task_dir, 0)
    _write_raw_episode(task_dir, 1)
    output_root = tmp_path / 'lerobot'
    _write_v30_output(
        output_root,
        task_dir.name,
        2,
        data_episode_indices=[0, 0],
    )

    with pytest.raises(RuntimeError, match='data episode coverage'):
        _delete_sources_after_validated_conversion(
            task_dir,
            source_folders=[],
            convert_v21=False,
            convert_v30=True,
            logger=logging.getLogger('test-cleanup'),
            output_root=output_root,
        )

    assert (task_dir / '0' / '0_0.mcap').is_file()
    assert (task_dir / '1' / '1_0.mcap').is_file()


@pytest.mark.parametrize('version', ['v21', 'v30'])
def test_declared_video_without_artifact_preserves_raw_source(tmp_path, version):
    task_dir = tmp_path / 'raw' / f'pick_jelly_{version}'
    _write_raw_episode(task_dir, 0)
    output_root = tmp_path / 'lerobot'
    if version == 'v21':
        _write_v21_output(
            output_root,
            task_dir.name,
            1,
            declare_video=True,
            write_videos=False,
        )
    else:
        _write_v30_output(
            output_root,
            task_dir.name,
            1,
            declare_video=True,
            write_video=False,
        )

    with pytest.raises(RuntimeError, match='video.*missing or empty'):
        _delete_sources_after_validated_conversion(
            task_dir,
            source_folders=[],
            convert_v21=version == 'v21',
            convert_v30=version == 'v30',
            logger=logging.getLogger('test-cleanup'),
            output_root=output_root,
        )

    assert (task_dir / '0' / '0_0.mcap').is_file()


def test_merge_cleanup_deletes_real_sources_and_unlinks_staging_view(tmp_path):
    source_a = tmp_path / 'raw' / 'source_a'
    source_b = tmp_path / 'raw' / 'source_b'
    _write_raw_episode(source_a, 2)
    _write_raw_episode(source_b, 7)
    merged = tmp_path / 'raw' / 'merged'
    merged.mkdir(parents=True)
    (merged / '0').symlink_to(source_a / '2', target_is_directory=True)
    (merged / '1').symlink_to(source_b / '7', target_is_directory=True)
    (source_a / 'README.md').write_text('keep a', encoding='utf-8')
    (source_b / 'README.md').write_text('keep b', encoding='utf-8')

    output_root = tmp_path / 'lerobot'
    _write_v21_output(output_root, merged.name, 2)

    deleted = _delete_sources_after_validated_conversion(
        merged,
        source_folders=[str(source_a), str(source_b)],
        convert_v21=True,
        convert_v30=False,
        logger=logging.getLogger('test-cleanup'),
        output_root=output_root,
    )

    assert deleted == 2
    assert not (source_a / '2').exists()
    assert not (source_b / '7').exists()
    assert (source_a / 'README.md').is_file()
    assert (source_b / 'README.md').is_file()
    assert not (merged / '0').exists()
    assert not (merged / '1').exists()


def test_staging_failure_rolls_back_every_prior_episode_move(tmp_path, monkeypatch):
    task_dir = tmp_path / 'raw' / 'pick_jelly'
    _write_raw_episode(task_dir, 0)
    _write_raw_episode(task_dir, 1)
    (task_dir / 'README.md').write_text('keep me', encoding='utf-8')

    real_rename = Path.rename
    staged_rename_count = 0

    def fail_second_staging_rename(source, target):
        nonlocal staged_rename_count
        target = Path(target)
        if source.name.isdigit() and target.parent.name.startswith('.cyclo_delete_'):
            staged_rename_count += 1
            if staged_rename_count == 2:
                raise OSError('injected staging failure')
        return real_rename(source, target)

    monkeypatch.setattr(Path, 'rename', fail_second_staging_rename)

    with pytest.raises(RuntimeError, match='staging failed and was rolled back'):
        _delete_raw_source_episodes(
            task_dir,
            source_folders=[],
            logger=logging.getLogger('test-cleanup'),
        )

    assert (task_dir / '0' / '0_0.mcap').is_file()
    assert (task_dir / '1' / '1_0.mcap').is_file()
    assert (task_dir / 'README.md').is_file()
    assert list(task_dir.glob('.cyclo_delete_*')) == []


def test_cleanup_failure_happens_only_after_all_sources_are_staged(
    tmp_path,
    monkeypatch,
):
    source_a = tmp_path / 'raw' / 'source_a'
    source_b = tmp_path / 'raw' / 'source_b'
    _write_raw_episode(source_a, 2)
    _write_raw_episode(source_b, 7)
    (source_a / 'README.md').write_text('keep a', encoding='utf-8')
    (source_b / 'README.md').write_text('keep b', encoding='utf-8')

    merged = tmp_path / 'raw' / 'merged'
    merged.mkdir(parents=True)
    (merged / '0').symlink_to(source_a / '2', target_is_directory=True)
    (merged / '1').symlink_to(source_b / '7', target_is_directory=True)

    real_rmtree = pipeline_worker.shutil.rmtree
    cleanup_count = 0

    def fail_second_staging_cleanup(path, *args, **kwargs):
        nonlocal cleanup_count
        if Path(path).name.startswith('.cyclo_delete_'):
            cleanup_count += 1
            if cleanup_count == 2:
                raise OSError('injected cleanup failure')
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        pipeline_worker.shutil,
        'rmtree',
        fail_second_staging_cleanup,
    )

    with pytest.raises(RuntimeError, match='All raw episodes were staged'):
        _delete_raw_source_episodes(
            merged,
            source_folders=[str(source_a), str(source_b)],
            logger=logging.getLogger('test-cleanup'),
        )

    # No later source remains visible merely because an earlier staging-tree
    # removal failed. The failed tree keeps its episode recoverable.
    assert not (source_a / '2').exists()
    assert not (source_b / '7').exists()
    assert not (merged / '0').exists()
    assert not (merged / '1').exists()
    remaining_staging = (
        list(source_a.glob('.cyclo_delete_*'))
        + list(source_b.glob('.cyclo_delete_*'))
        + list(merged.glob('.cyclo_delete_*'))
    )
    assert len(remaining_staging) == 1
    assert any(remaining_staging[0].rglob('7_0.mcap'))
    assert (source_a / 'README.md').is_file()
    assert (source_b / 'README.md').is_file()
