import base64
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from qwen_engine.engine import BusyError, NotReadyError, QwenEngine
from qwen_engine.server import GenerateRequest, create_app, decode_image


class EngineTests(unittest.TestCase):
    def test_idle_and_repeated_unload(self):
        engine = QwenEngine()
        self.assertFalse(engine.status()["ready"])
        with self.assertRaises(NotReadyError):
            engine.generate(GenerateRequest(prompt="hello"))
        engine.unload()
        engine.unload()
        self.assertEqual(engine.state, "unloaded")

    def test_load_generate_and_unload_reject_overlap(self):
        engine = QwenEngine()
        with engine._operation():
            for operation in (engine.load, engine.unload,
                              lambda: engine.generate(GenerateRequest(prompt="hello"))):
                with self.assertRaises(BusyError):
                    operation()
        self.assertFalse(engine.status()["busy"])

    def test_load_failure_releases_partial_state_and_allows_retry(self):
        engine = QwenEngine()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        fake_transformers = SimpleNamespace(AutoProcessor=None, BitsAndBytesConfig=None,
                                            Qwen3_5ForConditionalGeneration=None)
        with patch.dict(sys.modules, torch=fake_torch, transformers=fake_transformers):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    engine.load()
                self.assertIsNone(engine.model)
                self.assertIsNone(engine.processor)
                self.assertEqual(engine.state, "error")
                self.assertFalse(engine.status()["busy"])

    def test_generation_passes_image_and_trims_prompt(self):
        import torch

        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                # Transformers' multimodal processor expects content blocks
                # for every role, including the system message.
                assert all(isinstance(message["content"], list) for message in messages)
                self.messages, self.kwargs = messages, kwargs
                return Inputs(input_ids=torch.tensor([[1, 2, 3]]))

            def decode(self, tokens, **kwargs):
                assert tokens.tolist() == [4, 5]
                return "answer"

        engine = QwenEngine()
        engine.processor = Processor()
        generation_options = {}
        def generate(**kwargs):
            generation_options.update(kwargs)
            return torch.tensor([[1, 2, 3, 4, 5]])
        engine.model = SimpleNamespace(
            device="cpu", generation_config=SimpleNamespace(eos_token_id=[5]),
            generate=generate,
        )
        image = Image.new("RGB", (32, 32))
        result = engine.generate(GenerateRequest(prompt="look"), image)
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["input_tokens"], 3)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertIs(engine.processor.messages[1]["content"][0]["image"], image)
        self.assertFalse(engine.processor.kwargs["enable_thinking"])
        generation_options.clear()
        engine.generate(GenerateRequest(prompt="look", do_sample=False))
        self.assertFalse(generation_options["do_sample"])
        self.assertNotIn("temperature", generation_options)
        engine.max_input_tokens = 2
        with self.assertRaisesRegex(ValueError, "limit"):
            engine.generate(GenerateRequest(prompt="look"))

    def test_conversation_history_and_adapter_restoration(self):
        import torch
        from contextlib import contextmanager
        class Inputs(dict):
            def to(self, device): return self
        engine = QwenEngine()
        seen = []
        def template(messages, **kwargs):
            seen.extend(messages)
            return Inputs(input_ids=torch.tensor([[1]]))
        engine.processor = SimpleNamespace(apply_chat_template=template, decode=lambda *a, **kw: 'hello')
        enabled = [True]
        @contextmanager
        def disable():
            enabled[0] = False
            try: yield
            finally: enabled[0] = True
        def generate(**kwargs):
            self.assertFalse(enabled[0])
            return torch.tensor([[1, 9]])
        engine.model = SimpleNamespace(device='cpu', generation_config=SimpleNamespace(eos_token_id=9),
                                       disable_adapter=disable, generate=generate)
        request = GenerateRequest(prompt='Why?', use_adapter=False,
            history=[{'role':'user','content':'What are you doing?'}, {'role':'assistant','content':'Observing.'}])
        engine.generate(request)
        self.assertTrue(enabled[0])
        self.assertEqual([m['role'] for m in seen], ['system','user','assistant','user'])
        def broken(**kwargs): raise RuntimeError('generation failed')
        engine.model.generate = broken
        with self.assertRaises(RuntimeError): engine.generate(request)
        self.assertTrue(enabled[0])

    def test_unfinished_thinking_is_not_an_answer(self):
        import torch
        # Reuse the generation fixture with a tiny mapping that supports .to().
        class Inputs(dict):
            def to(self, device):
                return self
        engine = QwenEngine()
        engine.processor = SimpleNamespace(
            apply_chat_template=lambda *a, **kw: Inputs(input_ids=torch.tensor([[1]])),
            decode=lambda *a, **kw: "still thinking",
        )
        engine.model = SimpleNamespace(
            device="cpu", generation_config=SimpleNamespace(eos_token_id=9),
            generate=lambda **kw: torch.tensor([[1, 2]]),
        )
        result = engine.generate(GenerateRequest(prompt="look", enable_thinking=True))
        self.assertEqual(result["text"], "")
        self.assertEqual(result["finish_reason"], "length")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = QwenEngine()
        self.client = TestClient(create_app(self.engine))

    def test_liveness_does_not_claim_model_readiness(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertFalse(self.client.get("/health").json()["ready"])
        self.assertEqual(self.client.get("/ready").status_code, 503)
        self.assertEqual(self.client.post("/generate", json={"prompt": "hello"}).status_code, 503)

    def test_invalid_input_is_rejected(self):
        for payload in ({"prompt": " "}, {"prompt": "hi", "max_new_tokens": 99999},
                        {"prompt": "hi", "image_base64": "not-base64"},
                        {"prompt": "hi", "image_url": "file:///etc/passwd"}):
            self.assertEqual(self.client.post("/generate", json=payload).status_code, 422)

    def test_busy_lifecycle_is_conflict_not_queued(self):
        with self.engine._operation():
            self.assertEqual(self.client.post("/load").status_code, 409)
            self.assertEqual(self.client.post("/unload").status_code, 409)
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_image_decode_accepts_png_and_rejects_nonimage(self):
        data = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(data, format="PNG")
        image = decode_image(base64.b64encode(data.getvalue()).decode())
        self.assertEqual(image.size, (32, 32))
        with self.assertRaises(ValueError):
            decode_image(base64.b64encode(b"not an image").decode())


if __name__ == "__main__":
    unittest.main()


class SelectedAdapterTests(unittest.TestCase):
    def test_load_validates_requests_and_passes_selection(self):
        from unittest.mock import Mock
        engine=Mock();engine.load.return_value={'ready':True}
        client=TestClient(create_app(engine))
        self.assertEqual(client.post('/load',json={'model_id':'Qwen/Qwen3.5-4B'}).status_code,422)
        response=client.post('/load',json={'model_id':'Qwen/Qwen3.5-4B','revision':'rev','adapter_path':'/models/custom'})
        self.assertEqual(response.status_code,200)
        self.assertEqual(engine.load.call_args.args[0].adapter_path,'/models/custom')
        baseline=client.post('/load',json={'model_id':'Qwen/Qwen3.5-2B','revision':'15852e8c16360a2fea060d615a32b45270f8a8fc'})
        self.assertEqual(baseline.status_code,200)
        self.assertEqual(engine.load.call_args.args[0].adapter_path,'')
        tiny=client.post('/load',json={'model_id':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17'})
        self.assertEqual(tiny.status_code,200)
        self.assertEqual(engine.load.call_args.args[0].adapter_path,'')
        client.post('/load',json={});engine.load.assert_called_with(None)

    def test_adapter_switch_releases_old_weights_and_does_not_fallback_on_failure(self):
        import json, tempfile
        from pathlib import Path
        from qwen_engine.server import LoadRequest
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',QWEN_ADAPTER_ROOT=tmp):
            adapter=Path(tmp)/'custom';adapter.mkdir()
            (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'Qwen/Qwen3.5-4B'}))
            (adapter/'adapter_model.safetensors').write_bytes(b'weights')
            selection=LoadRequest(model_id='Qwen/Qwen3.5-4B',revision='trained-rev',adapter_path=str(adapter))
            engine=QwenEngine();engine.model=object();engine.processor=object();engine.state='ready'
            base=Mock();base.eval.return_value=base
            adapted=Mock();adapted.eval.return_value=adapted
            cuda=SimpleNamespace(is_available=lambda:True,empty_cache=lambda:None,memory_allocated=lambda:0,memory_reserved=lambda:0)
            torch=SimpleNamespace(cuda=cuda,bfloat16='bf16')
            transformers=SimpleNamespace(AutoProcessor=Mock(),BitsAndBytesConfig=Mock(),Qwen3_5ForConditionalGeneration=Mock())
            transformers.Qwen3_5ForConditionalGeneration.from_pretrained.return_value=base
            peft=SimpleNamespace(PeftConfig=Mock(),PeftModel=Mock())
            peft.PeftConfig.from_pretrained.return_value.base_model_name_or_path=selection.model_id
            peft.PeftModel.from_pretrained.return_value=adapted
            with patch.dict(sys.modules,torch=torch,transformers=transformers,peft=peft):
                health=engine.load(selection)
                self.assertTrue(health['ready']);self.assertEqual(health['adapter'],str(adapter))
                peft.PeftModel.from_pretrained.assert_called_once_with(base,str(adapter),is_trainable=False)
                self.assertEqual(transformers.AutoProcessor.from_pretrained.call_args.kwargs['revision'],'trained-rev')
                engine.load(selection)
                self.assertEqual(peft.PeftModel.from_pretrained.call_count,1)
                # Changing revision forces a reload; a failed reload cannot keep serving old weights.
                transformers.AutoProcessor.from_pretrained.side_effect=RuntimeError('bad model')
                with self.assertRaisesRegex(RuntimeError,'bad model'):
                    engine.load(selection.model_copy(update={'revision':'another'}))
                self.assertFalse(engine.status()['ready']);self.assertIsNone(engine.model)

    def test_2b_baseline_loads_without_peft_adapter(self):
        from unittest.mock import Mock
        from qwen_engine.server import LoadRequest
        selection = LoadRequest(model_id='Qwen/Qwen3.5-2B',
                                revision='15852e8c16360a2fea060d615a32b45270f8a8fc')
        engine = QwenEngine()
        base = Mock()
        base.eval.return_value = base
        cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None,
                               memory_allocated=lambda: 0, memory_reserved=lambda: 0)
        torch = SimpleNamespace(cuda=cuda, bfloat16='bf16')
        transformers = SimpleNamespace(AutoProcessor=Mock(), BitsAndBytesConfig=Mock(),
                                       Qwen3_5ForConditionalGeneration=Mock())
        transformers.Qwen3_5ForConditionalGeneration.from_pretrained.return_value = base
        with patch.dict(sys.modules, torch=torch, transformers=transformers):
            result = engine.load(selection)
        self.assertTrue(result['ready'])
        self.assertEqual(result['model'], 'Qwen/Qwen3.5-2B')
        self.assertIsNone(result['adapter'])
        self.assertEqual(transformers.AutoProcessor.from_pretrained.call_args.kwargs['revision'],
                         selection.revision)
        self.assertEqual(transformers.Qwen3_5ForConditionalGeneration.from_pretrained.call_args.kwargs['revision'],
                         selection.revision)

    def test_explicit_adapter_must_stay_inside_model_root(self):
        from qwen_engine.server import LoadRequest
        engine=QwenEngine()
        with self.assertRaises(ValueError):
            engine.load(LoadRequest(model_id='Qwen/Qwen3.5-4B',revision='rev',adapter_path='/etc'))
