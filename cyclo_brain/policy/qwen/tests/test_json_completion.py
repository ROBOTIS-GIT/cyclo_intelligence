import unittest
from qwen_engine.json_completion import completed_object

class JsonCompletionTests(unittest.TestCase):
    def test_stops_at_first_object_before_synthetic_conversation(self):
        answer = '{"decision":"continue","mission_id":"laundry","reason":"Clear."}'
        text = answer + '\nuser\n{"decision":"continue","mission_id":"serve","reason":"Other."}'
        self.assertEqual(completed_object(text), answer)

    def test_braces_and_escaped_quotes_in_reason_do_not_end_the_answer(self):
        text = ' {"reason":"brace } and \\"quoted\\"", "nested":{"x":1}}'
        self.assertEqual(completed_object(text), text.strip())
        self.assertIsNone(completed_object(text[:-1]))

    def test_does_not_recover_prefaces_invalid_or_truncated_json(self):
        for text in ('user\n{}', '{"reason":', '{"reason": broken}', '[]', ''):
            self.assertIsNone(completed_object(text))

    def test_engine_json_completion_is_a_normal_finish_without_eos(self):
        import torch
        from types import SimpleNamespace
        from qwen_engine.engine import QwenEngine
        from qwen_engine.server import GenerateRequest
        class Inputs(dict):
            def to(self, device): return self
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return Inputs(input_ids=torch.tensor([[1]]))
            def decode(self, tokens, **kwargs):
                return ''.join(chr(value) for value in tokens.tolist())
        answer = '{"decision":"continue","mission_id":"laundry","reason":"Clear."}'
        def generate(input_ids, stopping_criteria, **kwargs):
            ids = input_ids
            for char in answer + '\nuser\n{"decision":"cancel"}':
                ids = torch.cat([ids, torch.tensor([[ord(char)]])], dim=1)
                if stopping_criteria(ids, None).all(): break
            return ids
        engine = QwenEngine();engine.processor = Processor()
        engine.model = SimpleNamespace(device='cpu', generation_config=SimpleNamespace(eos_token_id=9), generate=generate)
        result = engine.generate(GenerateRequest(prompt='test', json_object=True, do_sample=False))
        self.assertEqual(result['text'], answer)
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['output_tokens'], len(answer))
