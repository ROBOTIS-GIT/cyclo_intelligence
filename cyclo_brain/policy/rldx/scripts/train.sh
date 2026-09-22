#!/usr/bin/env bash
set -euo pipefail

# Invoke with the isolated RLDX environment activated. No production services
# are started and no robot commands are published by training or smoke checks.
: "${RLDX_REPO:?Set RLDX_REPO to the pinned upstream checkout}"
: "${RLDX_DATASET_PATH:?Set RLDX_DATASET_PATH to the prepared v2.1 dataset}"
: "${RLDX_OUTPUT:?Set RLDX_OUTPUT to a new output directory}"
STEPS=${STEPS:-10}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -e "$RLDX_DATASET_PATH/INCOMPLETE" || -e "$RLDX_OUTPUT" ]]; then
    echo "Dataset is incomplete or output already exists; refusing to overwrite." >&2
    exit 1
fi
cd "$RLDX_REPO"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
torchrun --standalone --nproc_per_node=1 rldx/experiment/launch_train.py \
    --base-model-path "${RLDX_BASE_MODEL:-RLWRLD/RLDX-1-PT-IMG}" \
    --dataset-path "$RLDX_DATASET_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "$SCRIPT_DIR/modality_config.py" \
    --video-length 1 --video-stride 2 --action-horizon 16 \
    --global-batch-size "${BATCH_SIZE:-16}" --gradient-accumulation-steps "${GRAD_ACCUM:-4}" \
    --dataloader-num-workers 2 --learning-rate 0.0001 \
    --max-steps "$STEPS" --save-steps "${SAVE_STEPS:-$STEPS}" --save-total-limit "${SAVE_TOTAL_LIMIT:-1}" \
    --output-dir "$RLDX_OUTPUT" --experiment-name "$(basename "$RLDX_OUTPUT")"
for checkpoint in "$RLDX_OUTPUT/$(basename "$RLDX_OUTPUT")"/checkpoint-*; do
    [[ -d "$checkpoint" ]] || continue
    cp "$RLDX_DATASET_PATH/cyclo_input_metadata.json" "$checkpoint/cyclo_input_metadata.json"
done
