"""The judging endpoint. Audio is processed in memory and is never stored."""
import base64
import binascii
from contextlib import asynccontextmanager
import hashlib
import io
import os
from pathlib import Path
import struct
from typing import Literal

import joblib
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from behavior_features import BLOCKS, extract_model_features
from provenance import pipeline_fingerprint
from vad import energy_profile, read_wav, turns_from_profiles

MAX_AUDIO_SECONDS = 900
MAX_AUDIO_BYTES = 32 * 1024 * 1024
MAX_BASE64_LENGTH = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
MAX_REQUEST_BYTES = MAX_BASE64_LENGTH + 4096
MIN_AUDIO_SECONDS = 3.0
ROOT = Path(__file__).resolve().parent


class RequestBodyLimit:
    """Bound JSON bodies before parsing, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                response = JSONResponse({"detail": "Request exceeds the audio size limit"}, 413)
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


@asynccontextmanager
async def lifespan(app):
    path = Path(os.environ.get("MODEL_PATH", str(ROOT / "artifacts/model.joblib")))
    # Only load our own trusted artifact; joblib is not an upload format.
    bundle = joblib.load(path)
    supported_nan_policies = {"native_hgb", "pipeline_imputer"}
    if bundle.get("trained_split") != "train" or bundle.get("nan_policy") not in supported_nan_policies:
        raise RuntimeError("Model must use train-only fitting and an explicit missing-value policy")
    if not bundle.get("vad_config"):
        raise RuntimeError("A frozen VAD configuration is required")
    if set(bundle.get("feature_blocks", ())) - set(BLOCKS):
        raise RuntimeError("Model requires unsupported behavior blocks")
    app.state.bundle = bundle
    app.state.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    app.state.pipeline_sha256 = pipeline_fingerprint(model_path=path)
    with threadpool_limits(limits=1):
        bundle["model"].predict_proba(np.zeros((1, len(bundle["feats"]))))
        yield


app = FastAPI(title="Altur · Detector conversacional", version="0.3.0", lifespan=lifespan)
app.add_middleware(RequestBodyLimit)


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio: str = Field(max_length=MAX_BASE64_LENGTH, description="Base64 WAV, stereo PCM16, 8 kHz")
    format: Literal["wav"] = "wav"


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float = Field(ge=0.5, le=1.0, description="Probability of the returned class; 0.5 is abstention")


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Pydantic's default response can echo the submitted audio. Omit all input.
    errors = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


def abstain():
    # The contract requires a bool. False is only a placeholder at confidence 0.5.
    return {"is_synthetic": False, "confidence": 0.5}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        raw = base64.b64decode(req.audio, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(422, "audio must be valid base64") from None
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "WAV exceeds 32 MiB")
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise HTTPException(422, "audio must contain a WAV file")
    if struct.unpack_from("<I", raw, 4)[0] + 8 > len(raw):
        raise HTTPException(422, "Truncated WAV")
    try:
        info = sf.info(io.BytesIO(raw))
    except (sf.LibsndfileError, ValueError):
        raise HTTPException(422, "Unreadable WAV") from None
    if info.channels == 1:
        return abstain()
    if info.channels != 2 or info.samplerate != 8000 or info.subtype != "PCM_16":
        raise HTTPException(422, "Expected stereo PCM16 WAV at 8000 Hz")
    if info.duration > MAX_AUDIO_SECONDS:
        raise HTTPException(413, "WAV exceeds 900 seconds")
    if info.duration < MIN_AUDIO_SECONDS:
        return abstain()
    try:
        data, sr = read_wav(raw)
    except (sf.LibsndfileError, ValueError):
        raise HTTPException(422, "Unreadable WAV samples") from None
    if not np.any(data):
        return abstain()
    bundle = app.state.bundle
    config = bundle["vad_config"]
    profiles = [energy_profile(data[:, ch], sr, config["frame_ms"]) for ch in (0, 1)]
    turns = turns_from_profiles(profiles, **config)
    if len(turns["turns"]) < 4:
        return abstain()
    features = extract_model_features(turns, bundle.get("feature_blocks", ()))
    if not features or features["n_caller"] + features["n_agent"] < 4:
        return abstain()
    row = np.asarray([[features.get(name, np.nan) for name in bundle["feats"]]], dtype=float)
    p = float(bundle["model"].predict_proba(row)[0, 1])
    is_synthetic = p > bundle["decision_threshold"]
    # Preserve full precision so HTTP AUC/Brier reproduce the offline evaluation.
    return {"is_synthetic": bool(is_synthetic), "confidence": p if is_synthetic else 1 - p}


@app.get("/health")
def health():
    return {"ok": True, "model_sha256": app.state.model_sha256,
            "pipeline_sha256": app.state.pipeline_sha256,
            "features": len(app.state.bundle["feats"]),
            "feature_blocks": app.state.bundle.get("feature_blocks", [])}
