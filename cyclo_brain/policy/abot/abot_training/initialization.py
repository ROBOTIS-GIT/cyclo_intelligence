"""Explicit pretrained loading: strict backbone, dimension-dependent heads."""

HEADS = ("action_model.state_encoder.", "action_model.action_encoder.", "action_model.action_decoder.")
VGGT = ("spatial_model.", "spatial_projector.", "fuser.")
DIMENSION_AXES = {
    "action_model.state_encoder.layer1.weight": 1,
    "action_model.action_encoder.layer1.weight": 1,
    "action_model.action_decoder.layer2.weight": 0,
    "action_model.action_decoder.layer2.bias": 0,
}


def _vocabulary_plan(model, source, target):
    interface = model.qwen_vl_interface
    vlm = interface.model
    embedding = vlm.get_input_embeddings().weight
    output = vlm.get_output_embeddings().weight
    tied = embedding is output or embedding.data_ptr() == output.data_ptr()
    keys = []
    for module in (vlm.get_input_embeddings(), vlm.get_output_embeddings()):
        matches = [name + ".weight" for name, child in model.named_modules() if child is module]
        if len(matches) != 1:
            raise ValueError("Cannot identify unique Qwen input/output embedding keys")
        keys.append(matches[0])
    if tied:
        import torch

        if not torch.equal(source[keys[0]], source[keys[1]]):
            raise ValueError("Tied Qwen embeddings require identical pretrained input/output weights")
    old_rows, new_rows = [], []
    for key in keys:
        left, right = source[key].shape, target[key].shape
        if len(left) != 2 or len(right) != 2 or left[1] != right[1]:
            raise ValueError("Qwen vocabulary restoration cannot change hidden dimensions")
        old_rows.append(right[0])
        new_rows.append(left[0])
    if old_rows[0] != old_rows[1] or new_rows[0] != new_rows[1]:
        raise ValueError("Qwen embedding and LM head vocabulary sizes disagree")
    if new_rows[0] < old_rows[0]:
        raise ValueError("Qwen vocabulary restoration must not truncate rows")
    tokenizer = interface.processor.tokenizer
    vocabulary = tokenizer.get_vocab()
    if (not isinstance(vocabulary, dict) or not vocabulary
            or any(type(i) is not int or not 0 <= i < old_rows[0] for i in vocabulary.values())):
        raise ValueError("Processor token IDs fall outside the original Qwen vocabulary")
    if any(type(i) is not int or not 0 <= i < old_rows[0] for i in tokenizer.all_special_ids):
        raise ValueError("Processor special token IDs fall outside the original Qwen vocabulary")
    return vlm, keys, {"mode": "restore_checkpoint", "original_rows": old_rows[0],
                       "checkpoint_rows": new_rows[0], "restored_rows": new_rows[0] - old_rows[0],
                       "tokenizer_modified": False, "tied_embeddings_preserved": tied}


def load_pretrained(model, weights, *, restore_vocabulary=False):
    target = model.state_dict(keep_vars=True)
    discarded = sorted(k for k in weights if k.startswith(VGGT))
    source = {k: v for k, v in weights.items() if k not in discarded}
    if set(source) != set(target):
        raise ValueError(f"Pretrained keys differ: missing={sorted(set(target) - set(source))}, "
                         f"unexpected={sorted(set(source) - set(target))}")
    vocabulary = None
    vocabulary_keys = []
    if restore_vocabulary:
        vlm, vocabulary_keys, vocabulary = _vocabulary_plan(model, source, target)
    mismatched = [k for k in target if source[k].shape != target[k].shape]
    unsupported = []
    for key in mismatched:
        axis = 0 if key in vocabulary_keys else DIMENSION_AXES.get(key)
        left, right = source[key].shape, target[key].shape
        if (axis is None or len(left) != len(right)
                or any(a != b for i, (a, b) in enumerate(zip(left, right)) if i != axis)):
            unsupported.append(key)
    if unsupported:
        raise ValueError(f"Shapes differ outside input/output dimensions: {unsupported}")
    reset = [prefix for prefix in HEADS if any(k.startswith(prefix) for k in mismatched)]
    # Validate every key and shape before resizing any module. New rows are then
    # overwritten by pretrained tensors, never left randomly initialized.
    if vocabulary and vocabulary["restored_rows"]:
        vlm.resize_token_embeddings(vocabulary["checkpoint_rows"], mean_resizing=False)
        target = model.state_dict()
    # Retain a newly constructed head in full, not just its mismatching tensors.
    merged = {k: target[k] if any(k.startswith(prefix) for prefix in reset) else source[k] for k in target}
    model.load_state_dict(merged, strict=True)
    return {"head_initialization": "reinitialize_mismatched", "reinitialized_heads": reset,
            "mismatched_keys": sorted(mismatched), "discarded_vggt_keys": discarded,
            "vocabulary": vocabulary}
