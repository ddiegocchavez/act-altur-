"""Focused audio robustness checks through the judging HTTP endpoint.

Only the three predeclared families are evaluated: gain, telephone resampling
round-trip and duration truncation. Transformed WAVs stay in memory.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import time

import httpx
import numpy as np
import pandas as pd

from evaluate_http import require_local_url
from metrics import evaluate


SCENARIOS = ("clean", "gain_minus_6db", "gain_plus_6db", "resample_8_16_8",
             "truncate_30s", "truncate_60s")


def transform_wav(raw, scenario):
    import soundfile as sf
    from scipy.signal import resample_poly

    data, samplerate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if samplerate != 8000 or data.shape[1] != 2:
        raise ValueError("Expected official stereo 8 kHz audio")
    if scenario == "clean":
        return raw
    if scenario == "gain_minus_6db":
        data *= 10 ** (-6 / 20)
    elif scenario == "gain_plus_6db":
        data = np.clip(data * 10 ** (6 / 20), -1, 1)
    elif scenario == "resample_8_16_8":
        data = resample_poly(resample_poly(data, 2, 1, axis=0), 1, 2, axis=0)
    elif scenario.startswith("truncate_"):
        seconds = int(scenario.removeprefix("truncate_").removesuffix("s"))
        data = data[:seconds * samplerate]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    output = io.BytesIO()
    sf.write(output, data, samplerate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def run(url, data_dir, output):
    require_local_url(url)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    val = manifest.loc[manifest.split == "val"].reset_index(drop=True)
    if len(val) != 71 or not val.anon_id.is_unique:
        raise RuntimeError("Unexpected official val split")
    prediction_rows = []
    report = {"status": "complete", "url": url, "audio_storage": "in_memory_only",
              "scenarios": {}}
    with httpx.Client(base_url=url.rstrip("/"), timeout=httpx.Timeout(120, connect=30),
                      trust_env=False, follow_redirects=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        report["model_sha256"] = health.json()["model_sha256"]
        for scenario in SCENARIOS:
            rows = []
            for item in val.itertuples(index=False):
                raw = (data_dir / "audio" / f"{item.anon_id}.wav").read_bytes()
                audio = base64.b64encode(transform_wav(raw, scenario)).decode("ascii")
                started = time.perf_counter()
                response = client.post("/detect", json={"audio": audio, "format": "wav"})
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                result = response.json()
                confidence = float(result["confidence"])
                probability = confidence if result["is_synthetic"] else 1 - confidence
                rows.append((item.label, probability, confidence, elapsed))
                prediction_rows.append({"scenario": scenario, "anon_id": item.anon_id,
                                        "label": item.label, "p_synthetic": probability,
                                        "confidence": confidence, "latency_s": elapsed})
            labels = np.asarray([label == "synthetic" for label, _, _, _ in rows], dtype=int)
            probability = np.asarray([value for _, value, _, _ in rows])
            latency = np.asarray([value for _, _, _, value in rows])
            scores = evaluate(labels, probability)
            scores.update({"abstentions": sum(confidence == .5 for _, _, confidence, _ in rows),
                           "latency_p50_ms": float(np.median(latency) * 1000),
                           "latency_p95_ms": float(np.percentile(latency, 95) * 1000)})
            report["scenarios"][scenario] = scores
            print(f"{scenario}: {scores['correct']}/71", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output.with_name(output.stem + "_predictions.csv"), index=False)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/audio_robustness.json"))
    args = parser.parse_args()
    run(args.url, args.data_dir, args.output)
