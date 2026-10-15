import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from data_format import messages_for, read_samples


class QwenDataFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new("RGB", (16, 12), "red").save(self.root / "frame.jpg")
        self.sample = {"messages": [
            {"role": "system", "content": [{"type":"text", "text":"Choose a mission."}]},
            {"role": "user", "content": [{"type":"image", "image":"frame.jpg"},
                {"type":"text", "text":'{"context":{"active_mission":null},"missions":[{"mission_id":"m2"}]}'}]},
            {"role": "assistant", "content": [{"type":"text", "text":'{"decision":"start","mission_id":"m2","reason":"Red stick visible."}'}]}]}

    def test_conversion_preserves_prompt_target_image_and_files(self):
        source = copy.deepcopy(self.sample)
        path = self.root / "train.jsonl"
        path.write_text(json.dumps(self.sample) + "\n")
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        rows = read_samples(path)
        messages = messages_for(rows[0], self.root)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[1]["content"][1]["text"], self.sample["messages"][1]["content"][1]["text"])
        self.assertEqual(messages[-1]["content"][0]["text"], self.sample["messages"][-1]["content"][0]["text"])
        self.assertEqual(messages[1]["content"][0]["image"].size, (16, 12))
        self.assertEqual(messages[1]["content"][0]["image"].mode, "RGB")
        self.assertEqual(self.sample, source)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)
        messages[1]["content"][0]["image"].close()

    def test_bad_roles_placeholder_and_missing_image_report_line(self):
        variants = [copy.deepcopy(self.sample) for _ in range(3)]
        variants[0]["messages"][2]["role"] = "user"
        variants[1]["messages"][1]["content"] = "legacy string"
        variants[2]["messages"][1]["content"][0]["image"] = "missing.jpg"
        for sample in variants:
            with self.subTest(sample=sample):
                (self.root / "train.jsonl").write_text("\n" + json.dumps(sample))
                with self.assertRaisesRegex(ValueError, "train.jsonl:2:"):
                    read_samples(self.root / "train.jsonl")

    def test_outside_image_and_symlink_are_rejected(self):
        for image in ("../outside.jpg", "link.jpg"):
            if image == "link.jpg":
                (self.root / image).symlink_to(self.root.parent / "outside.jpg")
            self.sample["messages"][1]["content"][0]["image"] = image
            with self.assertRaisesRegex(ValueError, "inside the export directory"):
                messages_for(self.sample, self.root)


if __name__ == "__main__":
    unittest.main()
