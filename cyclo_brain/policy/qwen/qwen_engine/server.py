"""Local reasoning API. No robot command publishing or mission execution."""

import base64
import binascii
import io
import logging
import warnings
from typing import Literal

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .engine import BusyError, NotReadyError, QwenEngine

LOG = logging.getLogger(__name__)


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    model_id: str = Field(pattern=r"^Qwen/Qwen3\.5-[A-Za-z0-9.-]+$", max_length=120)
    revision: str = Field(min_length=1, max_length=120)
    adapter_path: str = Field(default='', max_length=1024)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    prompt: str = Field(min_length=1, max_length=16000)
    system_prompt: str = Field(
        default="Describe observable facts and distinguish uncertainty. "
                "Return a concise answer to the user's request.",
        min_length=1, max_length=4000,
    )
    # Raw base64, not a URL or server-local path. A camera adapter can encode
    # its own latest frame without giving the model access to arbitrary files.
    image_base64: str | None = Field(default=None, max_length=8 * 1024 * 1024)
    max_new_tokens: int = Field(default=512, ge=16, le=2048)
    enable_thinking: bool = False
    do_sample: bool = True
    attention_preview: bool = False
    json_object: bool = False
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
    use_adapter: bool = True


def decode_image(encoded):
    if encoded is None:
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported")
                if image.width * image.height > 16_000_000:
                    raise ValueError("Image exceeds 16 million pixels")
                image.load()
                return image.convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid base64 JPEG/PNG image") from exc


def create_app(engine=None):
    engine = engine if engine is not None else QwenEngine()
    app = FastAPI(title="Cyclo Qwen Reasoning", version="0.1.0")

    def invoke(operation, *args):
        try:
            return operation(*args)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except NotReadyError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            LOG.exception("Qwen operation failed")
            raise HTTPException(500, "Qwen operation failed; check container logs") from exc

    @app.get("/health")
    def health():
        # Liveness is independent of model loading; idle containers are healthy.
        return engine.status()

    @app.get("/ready")
    def ready():
        status = engine.status()
        if not status["ready"]:
            raise HTTPException(503, status)
        return status

    @app.post("/load")
    def load(request: dict | None = None):
        if request:
            try:
                selection = LoadRequest.model_validate(request)
            except ValueError as exc:
                raise HTTPException(422, "Provide model_id and revision, with an optional adapter_path") from exc
        else:
            selection = None
        return invoke(engine.load, selection)

    @app.post("/unload")
    def unload():
        return invoke(engine.unload)

    @app.post("/generate")
    def generate(request: GenerateRequest):
        image = invoke(decode_image, request.image_base64)
        return invoke(engine.generate, request, image)

    return app


app = create_app()
