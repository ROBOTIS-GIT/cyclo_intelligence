"""Stop structured reasoning after one complete top-level JSON object."""
import json


def completed_object(text):
    start = len(text) - len(text.lstrip())
    if start == len(text) or text[start] != "{":
        return None
    depth = 0
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    value = json.loads(candidate)
                except ValueError:
                    return None
                return candidate if isinstance(value, dict) else None
    return None


class JsonCompletion:
    def __init__(self, processor, input_tokens):
        self.processor = processor
        self.input_tokens = input_tokens
        self.completed = False

    def __call__(self, input_ids, scores, **kwargs):
        text = self.processor.decode(input_ids[0, self.input_tokens:], skip_special_tokens=True)
        self.completed = completed_object(text) is not None
        return self.completed
