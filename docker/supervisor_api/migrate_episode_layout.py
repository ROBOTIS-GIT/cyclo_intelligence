"""Move legacy map/episode folders into map/observation, map/patrol, or map/mission.

Run with --apply after reviewing the default dry-run. Episode IDs and sample files
are unchanged; only the episode directory and storage_path metadata move.
"""
import argparse
import json
import os
from pathlib import Path


def migrate(root: Path, apply: bool = False):
    moves = []
    for source in sorted(root.glob('*/raw_episode_*/episode.json')):
        episode_dir = source.parent
        episode = json.loads(source.read_text(encoding='utf-8'))
        if episode.get('ended_at') is None:
            raise RuntimeError(f'End the active episode before migrating: {episode_dir}')
        name = episode.get('recording_mission_name') or ''
        kind = episode.get('episode_type')
        if kind not in ('observation', 'patrol', 'mission'):
            kind = 'mission' if name and name != 'ObservationMode' else 'observation'
        destination = episode_dir.parent / kind / episode_dir.name
        if destination.exists():
            raise RuntimeError(f'Destination already exists: {destination}')
        if episode_dir.is_symlink() or any(child.is_symlink() for child in episode_dir.rglob('*')):
            raise RuntimeError(f'Symlink in episode: {episode_dir}')
        moves.append((episode_dir, destination, episode, kind))
    for source, destination, _, kind in moves:
        print(f'{source.relative_to(root)} -> {destination.relative_to(root)} [{kind}]')
    if not apply:
        print(f'Dry run: {len(moves)} episodes. Pass --apply to move them.')
        return len(moves)
    for source, destination, episode, kind in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        old_storage = Path(episode['storage_path'])
        episode.update(episode_type=kind, storage_path=str(old_storage.parent / kind / old_storage.name))
        temp = destination / 'episode.json.tmp'
        try:
            temp.write_text(json.dumps(episode, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            os.replace(temp, destination / 'episode.json')
        except Exception:
            temp.unlink(missing_ok=True)
            destination.rename(source)
            raise
    print(f'Migrated {len(moves)} episodes.')
    return len(moves)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='reasoning dataset root')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    migrate(args.root, args.apply)
