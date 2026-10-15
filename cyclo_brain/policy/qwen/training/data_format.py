"""Read-only bridge from Collect exports to Qwen3.5 Transformers messages.

Model-owned template (no copied or hand-written chat/vision special tokens):
https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja
Stored records use the same typed content structure; image paths become PIL objects only at load time.
"""
import argparse
import json
from pathlib import Path

FORMAT = "transformers_multimodal_messages"


def messages_for(sample, root):
    """Return fresh typed messages; preserve prompt/target text and source files."""
    from PIL import Image
    if not isinstance(sample, dict):
        raise ValueError("Expected a Collect export record")
    messages = sample.get("messages")
    if set(sample) != {"messages"}:
        raise ValueError("Expected typed messages only; migrate legacy messages/images records first")
    if not isinstance(messages, list) or len(messages) != 3 or not all(isinstance(m, dict) for m in messages):
        raise ValueError("Expected system/user/assistant messages")
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Unexpected message roles")
    for message, types in zip(messages, (["text"], ["image", "text"], ["text"])):
        blocks = message.get("content")
        if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks) or [b.get("type") for b in blocks] != types:
            raise ValueError("Expected typed image/text content blocks")
        for block in blocks:
            kind = block["type"]
            if set(block) != {"type", kind} or not isinstance(block[kind], str) or not block[kind].strip():
                raise ValueError("Expected non-empty text or image path")
    relative = messages[1]["content"][0]["image"]
    image_path = (Path(root) / relative).resolve()
    if not image_path.is_relative_to(Path(root).resolve()):
        raise ValueError("Image path must stay inside the export directory")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    from copy import deepcopy
    result = deepcopy(messages)
    result[1]["content"][0]["image"] = image
    return result


def read_samples(path):
    """Validate before allocating model weights; do not retain decoded images."""
    path = Path(path)
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            messages = messages_for(sample, path.parent)
            messages[1]["content"][0]["image"].close()
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Validate existing Collect exports without writing data or loading a model")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    files = [args.dataset / f"{split}.jsonl" for split in ("train", "val", "test")]
    if not any(path.is_file() for path in files):
        raise ValueError("No train/val/test JSONL files found")
    counts = {path.stem: len(read_samples(path)) for path in files if path.is_file()}
    print(json.dumps({"model_input_format": FORMAT, "samples": counts}, indent=2))


if __name__ == "__main__":
    main()
