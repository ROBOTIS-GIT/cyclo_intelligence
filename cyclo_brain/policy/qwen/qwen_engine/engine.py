"""Model lifecycle and bounded, single-request VLM inference."""

import gc
import os
from contextlib import contextmanager, nullcontext
from threading import Lock
from time import monotonic


class BusyError(RuntimeError):
    pass


class NotReadyError(RuntimeError):
    pass


class QwenEngine:
    def __init__(self):
        self.model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
        self.adapter_path = os.getenv("QWEN_ADAPTER_PATH", "").strip()
        self.revision = os.getenv("QWEN_MODEL_REVISION", "main")
        self.default_model = (self.model_id, self.revision, self.adapter_path)
        self.quantization = os.getenv("QWEN_QUANTIZATION", "4bit")
        if self.quantization not in {"4bit", "none"}:
            raise ValueError("QWEN_QUANTIZATION must be 4bit or none")
        self.max_input_tokens = int(os.getenv("QWEN_MAX_INPUT_TOKENS", "4096"))
        if self.max_input_tokens < 1:
            raise ValueError("QWEN_MAX_INPUT_TOKENS must be positive")
        self.model = None
        self.processor = None
        self.state = "unloaded"
        self._lock = Lock()

    @contextmanager
    def _operation(self):
        # Reject overlaps instead of queuing stale camera observations or
        # unloading weights while a generation is still using them.
        if not self._lock.acquire(blocking=False):
            raise BusyError("Qwen is busy; retry with a fresh observation")
        try:
            yield
        finally:
            self._lock.release()

    def status(self):
        memory = None
        if self.model is not None:
            import torch
            memory = {
                "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
            }
        return {"model": self.model_id, "revision": self.revision,
                "quantization": self.quantization, "adapter": self.adapter_path or None, "state": self.state,
                "ready": self.model is not None and self.state == "ready", "busy": self._lock.locked(),
                "gpu_memory": memory}

    def load(self, selection=None):
        with self._operation():
            desired = self.default_model if selection is None else (
                selection.model_id, selection.revision, selection.adapter_path)
            if self.model is not None and desired == (self.model_id, self.revision, self.adapter_path):
                return self.status()
            if selection is not None and selection.adapter_path:
                import json
                from pathlib import Path
                root = Path(os.getenv("QWEN_ADAPTER_ROOT", "/workspace/model/qwen")).resolve()
                adapter = Path(selection.adapter_path).resolve()
                for name in ("adapter_config.json", "adapter_model.safetensors"):
                    path = (adapter / name).resolve()
                    if not path.is_relative_to(root) or not path.is_file() or not path.stat().st_size:
                        raise ValueError("Select a completed adapter inside the Qwen model directory")
                config = json.loads((adapter / "adapter_config.json").read_text())
                if config.get("base_model_name_or_path") != selection.model_id:
                    raise ValueError("Adapter base model does not match the selected model")
            self.state = "loading"
            self._release()
            self.model_id, self.revision, self.adapter_path = desired
            try:
                import torch
                from transformers import (
                    AutoProcessor, BitsAndBytesConfig,
                    Qwen3_5ForConditionalGeneration,
                )

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable; check the NVIDIA container runtime")
                options = dict(revision=self.revision, dtype=torch.bfloat16,
                               device_map={"": 0}, attn_implementation="sdpa")
                from pathlib import Path
                vision_overlay = bool(self.adapter_path and any(
                    (Path(self.adapter_path) / name).exists()
                    for name in ("vision_manifest.json", "vision.safetensors", "vision_config.json")))
                if self.quantization == "4bit":
                    options["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=["lm_head", "model.visual"] if vision_overlay else None,
                    )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, revision=self.revision,
                    min_pixels=256 * 32 * 32, max_pixels=768 * 32 * 32,
                )
                self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    self.model_id, **options,
                ).eval()
                if vision_overlay:
                    from .vision_checkpoint import apply_vision_checkpoint
                    visual_config = apply_vision_checkpoint(
                        self.model.model.visual, self.adapter_path, self.model_id, self.revision)
                    self.processor.image_processor.min_pixels = visual_config["min_pixels"]
                    self.processor.image_processor.max_pixels = visual_config["max_pixels"]
                    self.processor.image_processor.size = {
                        "shortest_edge": visual_config["min_pixels"],
                        "longest_edge": visual_config["max_pixels"],
                    }
                if self.adapter_path:
                    from peft import PeftConfig, PeftModel
                    config = PeftConfig.from_pretrained(self.adapter_path)
                    if config.base_model_name_or_path != self.model_id:
                        raise ValueError("LoRA adapter base model does not match QWEN_MODEL_ID")
                    self.model = PeftModel.from_pretrained(
                        self.model, self.adapter_path, is_trainable=False,
                    ).eval()
                # Quantization allocates temporary buffers while loading.
                # Return unused cached blocks so GR00T can share this GPU.
                torch.cuda.empty_cache()
                self.state = "ready"
            except Exception:
                self._release()
                self.state = "error"
                raise
        return self.status()

    def _release(self):
        self.model = None
        self.processor = None
        gc.collect()
        # Import lazily: API and its tests can run without CUDA/model packages.
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self):
        with self._operation():
            self._release()
            self.state = "unloaded"
        return self.status()

    def generate(self, request, image=None):
        with self._operation():
            if self.model is None:
                raise NotReadyError("Load the model with POST /load first")
            import torch

            started = monotonic()
            content = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": request.prompt})
            messages = [{"role": "system", "content": [
                {"type": "text", "text": request.system_prompt}]},
                        *[{"role": turn.role, "content": [{"type": "text", "text": turn.content}]}
                          for turn in getattr(request, "history", [])],
                        {"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=request.enable_thinking,
            )
            input_tokens = inputs["input_ids"].shape[-1]
            if input_tokens > self.max_input_tokens:
                raise ValueError(f"Input has {input_tokens} tokens; limit is {self.max_input_tokens}")
            inputs = inputs.to(self.model.device)
            options = dict(max_new_tokens=request.max_new_tokens, do_sample=request.do_sample)
            if request.do_sample:
                options.update(temperature=0.6 if request.enable_thinking else 0.7,
                               top_p=0.95 if request.enable_thinking else 0.8, top_k=20)
            json_stop = None
            if getattr(request, "json_object", False):
                if request.enable_thinking:
                    raise ValueError("JSON object mode requires thinking to be disabled")
                from transformers import StoppingCriteriaList
                from .json_completion import JsonCompletion
                json_stop = JsonCompletion(self.processor, input_tokens)
                options["stopping_criteria"] = StoppingCriteriaList([json_stop])
            preview = None
            attention = {}
            if getattr(request, 'attention_preview', False) and image is not None:
                try:
                    from .attention_preview import PreviewAttention
                    preview = PreviewAttention(self.model, inputs)
                except Exception as exc:
                    attention['attention_error'] = str(exc)
            try:
                adapter_scope = (self.model.disable_adapter() if not getattr(request, 'use_adapter', True)
                                 and hasattr(self.model, 'disable_adapter') else nullcontext())
                with adapter_scope, torch.inference_mode():
                    output = self.model.generate(**inputs, **options)
            finally:
                if preview is not None: preview.remove()
            if preview is not None:
                try: attention['attention_preview'] = preview.render(image)
                except Exception as exc: attention['attention_error'] = str(exc)
            generated = output[0, input_tokens:]
            text = self.processor.decode(generated, skip_special_tokens=True).strip()
            # Thinking mode can begin with an open <think> in the prompt.
            # Do not turn an unfinished reasoning trace into a final answer.
            if request.enable_thinking:
                text = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            finished = len(generated) > 0 and generated[-1].item() in eos
            if json_stop is not None:
                from .json_completion import completed_object
                answer = completed_object(text)
                if answer is not None:
                    text = answer
                    finished = True
            return {**attention, "model": self.model_id, "revision": self.revision,
                    "adapter": self.adapter_path or None, "text": text,
                    "finish_reason": "stop" if finished else "length",
                    "input_tokens": input_tokens, "output_tokens": len(generated),
                    "elapsed_seconds": round(monotonic() - started, 3)}
