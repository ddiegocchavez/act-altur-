"""Gated, train-only behavior ablations, each measured through POST /detect."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from behavior_features import extract_blocks, extract_model_features
from check_endpoint import check_endpoint
from evaluate_http import evaluate_http
from train import fit_and_evaluate
from vad import turns_from_wav
from verify_local import wait_for_health

EXPERIMENTS = Path("artifacts/experiments")
PHASE3_BLOCKS = ("interruption_recovery", "silence_recovery", "consistency", "drift",
                 "autocorrelation")


def require_phase2():
    phase1 = json.loads(Path("reports/phase1.json").read_text())
    public = json.loads(Path("reports/phase2_public_http.json").read_text())
    local = json.loads(Path("reports/phase2_http.json").read_text())
    if not (phase1["gate"]["passed"] and public.get("transport") == "HTTPS public"
            and public.get("n") == 71 and public.get("http_successes") == 71
            and public.get("http_errors") == 0 and public.get("offline_equivalent")
            and public.get("correct") == local.get("correct") == 66
            and public.get("model_sha256") == local.get("model_sha256")):
        raise RuntimeError("Phase 2 gate requires the 71-call public baseline evaluation")
    return public


def cache_one(job):
    anon_id, config, cache = job
    target = Path(cache) / (anon_id + ".json")
    if target.exists():
        payload = json.loads(target.read_text())
    else:
        payload = turns_from_wav(Path("audio") / (anon_id + ".wav"), **config)
        target.write_text(json.dumps(payload) + "\n")
    return anon_id, payload


def load_vad_turns(manifest, config, workers=6):
    identity = hashlib.sha256(Path("vad.py").read_bytes() + Path("manifest.csv").read_bytes()
                              + json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path("cache") / ("vad_turns_" + identity)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(i), config, str(cache)) for i in manifest.anon_id]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        turns = {}
        for key, value in pool.map(cache_one, jobs):
            turns[key] = value
            if len(turns) % 50 == 0:
                print(f"VAD turns cached: {len(turns)}/353", flush=True)
    return turns, str(cache)


def candidate_http(name, model_path, offline, edges=False, server_python=None):
    url = "http://127.0.0.1:8002"
    log = Path("cache") / ("server_" + name + ".log")
    env = {**os.environ, "MODEL_PATH": str(model_path.resolve()), "PORT": "8002",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with log.open("w") as output:
        process = subprocess.Popen([server_python or sys.executable, "serve.py"], env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_for_health(url, process)
            result = evaluate_http(url, output=Path(f"reports/phase3_{name}_http.json"), offline=offline)
            if edges:
                check_endpoint(url, model_path=model_path, output=Path("reports/phase3_final_edges.json"))
            return result
        except Exception:
            print(log.read_text()[-4000:])
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    public = require_phase2()
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    baseline_path = Path("artifacts/baseline_model.joblib")
    if not baseline_path.exists():
        shutil.copyfile("artifacts/model.joblib", baseline_path)
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != public["model_sha256"]:
        raise RuntimeError("Saved baseline does not match the public evaluation")
    bundle = joblib.load(baseline_path)
    config, base_names = bundle["vad_config"], bundle["feats"]
    manifest = pd.read_csv("manifest.csv")
    turns, cache_dir = load_vad_turns(manifest, config, args.workers)
    base = pd.read_csv("cache/features_vad.csv")
    rows = []
    for row in base.itertuples(index=False):
        payload = turns[row.anon_id]
        f = extract_model_features(payload)
        expected = np.asarray([getattr(row, name) for name in base_names], dtype=float)
        actual = np.asarray([f.get(name, np.nan) for name in base_names], dtype=float)
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-9, equal_nan=True):
            raise RuntimeError("VAD cache does not reproduce the frozen baseline features")
        extra = extract_blocks(payload)
        # Do not train/evaluate on CSV-rounded floats: a tree threshold can
        # distinguish a one-ULP difference. Recompute exactly as /detect does.
        rows.append({"anon_id": row.anon_id, "label": row.label, "split": row.split, **f, **extra})
    frame = pd.DataFrame(rows)
    frame.to_csv("cache/features_behavior.csv", index=False)
    block_names = {block: list(extract_blocks(next(iter(turns.values())), (block,)))
                   for block in PHASE3_BLOCKS}
    baseline_http = candidate_http("baseline", baseline_path, Path("reports/phase1_predictions_val.csv"))
    if baseline_http["correct"] != 66 or not baseline_http["offline_equivalent"]:
        raise RuntimeError("Updated inference code changed the baseline")

    records = []

    def measure(name, blocks):
        names = base_names + [feature for block in blocks for feature in block_names[block]]
        path = EXPERIMENTS / (name + ".joblib")
        offline_scores, probability = fit_and_evaluate(frame, names, path, config, blocks)
        val = frame.loc[frame.split == "val", ["anon_id", "label"]].copy()
        val["p_synthetic"] = probability
        offline = Path(f"reports/phase3_{name}_offline.csv")
        val.to_csv(offline, index=False)
        http = candidate_http(name, path, offline)
        record = {"name": name, "blocks": list(blocks), "features": len(names),
                  "added_features": len(names) - len(base_names), "feature_names_added": names[len(base_names):],
                  "offline": offline_scores, "http": http,
                  "delta_accuracy_pp": 100 * (http["accuracy"] - baseline_http["accuracy"]),
                  "improves_baseline": http["correct"] > baseline_http["correct"],
                  "model_path": str(path)}
        records.append(record)
        print(f"ABLATION {name}: {http['correct']}/71, delta={record['delta_accuracy_pp']:+.2f} pp", flush=True)
        return record

    singles = [measure(block, (block,)) for block in PHASE3_BLOCKS]
    # Predeclared rule: accuracy first. Ties favor fewer features, then AUC/Brier.
    passing = sorted([r for r in singles if r["improves_baseline"]],
                     key=lambda r: (-r["http"]["correct"], r["features"], -r["http"]["auc"], r["http"]["brier"]))
    selected = passing[0] if passing else None
    for candidate in passing[1:]:
        blocks = selected["blocks"] + candidate["blocks"]
        combined = measure("plus_".join(blocks), blocks)
        combined["improves_previous_selected"] = combined["http"]["correct"] > selected["http"]["correct"]
        if combined["improves_previous_selected"]:
            selected = combined
    winner_path = Path(selected["model_path"]) if selected else baseline_path
    shutil.copyfile(winner_path, "artifacts/model.joblib")
    selected_blocks = selected["blocks"] if selected else []
    for record in records:
        record["selected_final"] = bool(selected and record["name"] == selected["name"])
    result = {"phase2_passed": True, "phase2_public_model_sha256": public["model_sha256"],
              "data": {"train": 282, "val": 71, "split": "Official speaker-disjoint train/val; no random CV"},
              "vad_config": config, "vad_cache": cache_dir, "baseline": baseline_http,
              "selection_rule": "Fixed HGB and VAD. Five single-block ablations; strict accuracy improvement. Ties favor fewer features, then AUC/Brier. Forward add individually passing blocks only when accuracy rises again.",
              "validation_note": "Val is reused for block selection, so selected gains are exploratory; there is no independent hidden-set estimate.",
              "ablations": records, "selected_blocks": selected_blocks,
              "selected_name": selected["name"] if selected else "baseline",
              "selected_metrics": selected["http"] if selected else baseline_http,
              "selected_model_sha256": hashlib.sha256(Path("artifacts/model.joblib").read_bytes()).hexdigest(),
              "discarded_blocks": [b for b in PHASE3_BLOCKS if b not in selected_blocks]}
    Path("reports/phase3.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_blocks": selected_blocks, "accuracy": result["selected_metrics"]["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
