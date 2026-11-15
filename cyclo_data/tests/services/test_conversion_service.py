from pathlib import Path

from cyclo_data.services.conversion_service import ConversionService


def test_existing_output_check_uses_selected_destination(tmp_path):
    dataset_path = tmp_path / 'raw' / 'Task_pick_jelly'
    dataset_path.mkdir(parents=True)
    default_root = tmp_path / 'default'
    selected_root = tmp_path / 'selected'
    selected_meta = (
        selected_root / f'{dataset_path.name}_lerobot_v30' / 'meta'
    )
    selected_meta.mkdir(parents=True)
    (selected_meta / 'info.json').write_text('{}', encoding='utf-8')

    assert ConversionService._existing_lerobot_outputs(
        str(dataset_path),
        convert_v21=False,
        convert_v30=True,
        output_root=selected_root,
    ) == [selected_root / f'{dataset_path.name}_lerobot_v30']
    assert ConversionService._existing_lerobot_outputs(
        str(dataset_path),
        convert_v21=False,
        convert_v30=True,
        output_root=default_root,
    ) == []
