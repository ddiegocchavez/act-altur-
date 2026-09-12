"""Robust behavior challengers; keep the clean champion unless gates pass.

Training perturbations are continuous and avoid neighborhoods around the fixed
evaluation shifts. This is a sensitivity experiment over VAD turns, not a
claim about audio from an unseen autonomous caller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from behavior_features import extract_blocks, extract_model_features
from metrics import evaluate
from stress_latency import shift_caller_turns


SEED = 20260912
TRAIN_AUGMENTATIONS = 3
EVALUATION_SHIFTS = (0.0, 0.5, 1.0, 1.5)
# Hold out +/- 0.1 s around every exact nonzero evaluation point.
TRAIN_SHIFT_INTERVALS = ((0.0, 0.4), (0.6, 0.9), (1.1, 1.4), (1.6, 2.0))
PHASE4_BLOCKS = ("silence_recovery", "relative_recovery")
EXCLUDED_ABSOLUTE_LATENCY = {
    "lat_mean", "lat_med", "lat_min", "lat_max", "lat_p10", "lat_p90",
    "lat_under_05", "lat_under_10", "lat_over_25", "lat_cv",
}
ROBUST_HGB_PARAMETERS = dict(max_iter=160, learning_rate=.05, max_leaf_nodes=7,
                             min_samples_leaf=20, l2_regularization=3,
                             early_stopping=False, class_weight="balanced", random_state=SEED)


def sample_training_shift(rng):
    lengths = np.asarray([right - left for left, right in TRAIN_SHIFT_INTERVALS], dtype=float)
    interval = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
    left, right = TRAIN_SHIFT_INTERVALS[interval]
    return float(rng.uniform(left, right))


def jitter_turns(payload, rng, shift_std=.06, boundary_std=.025):
    """Simulate small VAD boundary errors while preserving valid intervals."""
    output = []
    for turn in payload["turns"]:
        common = float(rng.normal(0, shift_std))
        start = max(0., float(turn["start"]) + common + float(rng.normal(0, boundary_std)))
        end = max(start + .03, float(turn["end"]) + common + float(rng.normal(0, boundary_std)))
        output.append({"channel": int(turn["channel"]), "start": round(start, 4), "end": round(end, 4)})
    return {"turns": sorted(output, key=lambda turn: (turn["start"], turn["channel"]))}


def augmented_payload(payload, label, rng):
    result = jitter_turns(payload, rng)
    if label == "synthetic":
        result = shift_caller_turns(result, sample_training_shift(rng))
    return result


def feature_union(payload):
    return extract_model_features(payload, PHASE4_BLOCKS)


def build_frames(manifest, payloads, augmentations=TRAIN_AUGMENTATIONS):
    rng = np.random.default_rng(SEED)
    clean, augmented = [], []
    for row in manifest.itertuples(index=False):
        base = {"anon_id": row.anon_id, "label": row.label, "split": row.split,
                **feature_union(payloads[row.anon_id])}
        clean.append(base)
        augmented.append(base)
        if row.split == "train":
            for index in range(augmentations):
                payload = augmented_payload(payloads[row.anon_id], row.label, rng)
                augmented.append({"anon_id": f"{row.anon_id}__aug{index}", "source_id": row.anon_id,
                                  "label": row.label, "split": "train", **feature_union(payload)})
    return pd.DataFrame(clean), pd.DataFrame(augmented)


def fit_model(frame, feature_names, kind):
    train = frame.loc[frame.split == "train"]
    x = train[feature_names].to_numpy(dtype=float)
    y = (train.label == "synthetic").to_numpy()
    if kind == "hgb":
        model = HistGradientBoostingClassifier(**ROBUST_HGB_PARAMETERS)
    elif kind == "logistic":
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              LogisticRegression(C=.5, max_iter=2000, class_weight="balanced",
                                                 random_state=SEED))
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    with threadpool_limits(limits=1):
        model.fit(x, y)
    return model


def probability_for_manifest(model, names, blocks, manifest, payloads, shift=0.):
    vectors = []
    for row in manifest.itertuples(index=False):
        payload = payloads[row.anon_id]
        if row.label == "synthetic" and shift:
            payload = shift_caller_turns(payload, shift)
        features = extract_model_features(payload, blocks)
        vectors.append([features.get(name, np.nan) for name in names])
    with threadpool_limits(limits=1):
        return model.predict_proba(np.asarray(vectors, dtype=float))[:, 1]


def scenario_report(model, names, blocks, val, payloads):
    y = (val.label == "synthetic").astype(int).to_numpy()
    scenarios, clean_probability = [], None
    for shift in EVALUATION_SHIFTS:
        probability = probability_for_manifest(model, names, blocks, val, payloads, shift)
        if shift == 0:
            clean_probability = probability
        scores = evaluate(y, probability)
        scores.update(acceleration_s=shift,
                      synthetic_recall=float(np.mean(probability[y == 1] > .5)))
        scenarios.append(scores)
    return scenarios, clean_probability


def expected_calibration_error(y, probability, bins=10):
    """Equal-width ECE; reported with its binning because ECE is not unique."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            value += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(value)


def calibration_diagnostic(y, probability):
    """In-sample Platt diagnostic on val; never used for selection or deployment."""
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    platt = LogisticRegression(C=1., max_iter=2000, random_state=SEED).fit(logits, y)
    calibrated = platt.predict_proba(logits)[:, 1]
    def scores(values):
        return {"brier": float(np.mean((values - y) ** 2)),
                "log_loss": float(log_loss(y, values)),
                "ece_10_equal_width": expected_calibration_error(y, values)}
    return {"status": "diagnostic_only", "before": scores(probability),
            "platt_in_sample": scores(calibrated),
            "coefficient": float(platt.coef_[0, 0]), "intercept": float(platt.intercept_[0]),
            "limitation": "Platt was fitted and scored on the same 71-call val set after model selection; these optimistic values are not used to rank or deploy a model."}


def robust_feature_names(base_names, relative_names):
    invariant_base = [name for name in base_names if name not in EXCLUDED_ABSOLUTE_LATENCY]
    return invariant_base + relative_names


def bundle_for(model, names, blocks, config, kind, metrics):
    return {"model": model, "feats": list(names), "vad_config": config,
            "trained_split": "train", "train_calls": 282,
            "nan_policy": "native_hgb" if kind == "hgb" else "pipeline_imputer",
            "decision_threshold": .5, "score_semantics": "P(synthetic)",
            "feature_blocks": list(blocks), "metrics": metrics,
            "robust_training": {"seed": SEED, "augmentations_per_train_call": TRAIN_AUGMENTATIONS,
                                "shift_intervals_s": TRAIN_SHIFT_INTERVALS,
                                "held_out_evaluation_shifts_s": EVALUATION_SHIFTS[1:]},
            "model_kind": kind}


def candidate_rank(record):
    clean = record["scenarios"][0]
    worst = min(record["scenarios"], key=lambda item: item["correct"])
    return (-worst["correct"], -clean["correct"], clean["brier"], record["features"])


def main():
    # Keep data/audio dependencies optional for unit tests and artifact audits.
    from phase3 import candidate_http, load_vad_turns

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--promote", action="store_true",
                        help="Replace artifacts/model.joblib only when the predeclared gates pass")
    args = parser.parse_args()
    manifest = pd.read_csv("manifest.csv")
    if len(manifest.query("split == 'train'")) != 282 or len(manifest.query("split == 'val'")) != 71:
        raise RuntimeError("Unexpected official split sizes")
    phase3 = json.loads(Path("reports/phase3.json").read_text())
    baseline = joblib.load("artifacts/baseline_model.joblib")
    champion = joblib.load("artifacts/model.joblib")
    if hashlib.sha256(Path("artifacts/model.joblib").read_bytes()).hexdigest() != phase3["selected_model_sha256"]:
        raise RuntimeError("Champion differs from the validated Phase 3 artifact")
    payloads, cache = load_vad_turns(manifest, baseline["vad_config"], args.workers)
    clean_frame, augmented_frame = build_frames(manifest, payloads)
    sample = feature_union(next(iter(payloads.values())))
    relative_names = list(extract_blocks(next(iter(payloads.values())), ("relative_recovery",)))
    robust_names = robust_feature_names(baseline["feats"], relative_names)
    specs = [
        ("augmented_current_hgb", champion["feats"], champion.get("feature_blocks", ()), "hgb"),
        ("relative_hgb", robust_names, ("relative_recovery",), "hgb"),
        ("relative_logistic", robust_names, ("relative_recovery",), "logistic"),
    ]
    for names in (champion["feats"], robust_names):
        missing = set(names) - set(sample)
        if missing:
            raise RuntimeError(f"Feature union is incomplete: {sorted(missing)}")
    val = manifest.query("split == 'val'").reset_index(drop=True)
    y_val = (val.label == "synthetic").astype(int).to_numpy()
    champion_scenarios, champion_probability = scenario_report(
        champion["model"], champion["feats"], champion.get("feature_blocks", ()), val, payloads)
    champion_worst = min(item["correct"] for item in champion_scenarios)
    candidates = []
    experiments = Path("artifacts/experiments")
    experiments.mkdir(parents=True, exist_ok=True)
    for name, names, blocks, kind in specs:
        model = fit_model(augmented_frame, names, kind)
        scenarios, clean_probability = scenario_report(model, names, blocks, val, payloads)
        path = experiments / f"phase4_{name}.joblib"
        joblib.dump(bundle_for(model, names, blocks, baseline["vad_config"], kind, scenarios[0]), path)
        predictions = val[["anon_id", "label"]].copy()
        predictions["p_synthetic"] = clean_probability
        predictions.to_csv(Path("reports") / f"phase4_{name}_offline.csv", index=False)
        worst = min(item["correct"] for item in scenarios)
        candidates.append({"name": name, "kind": kind, "features": len(names), "blocks": list(blocks),
                           "artifact": str(path), "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "scenarios": scenarios, "clean_gate_passed": scenarios[0]["correct"] >= 68,
                           "robustness_gate_passed": worst > champion_worst,
                           "worst_case_correct": worst,
                           "calibration": calibration_diagnostic(y_val, clean_probability)})
    eligible = sorted([item for item in candidates if item["clean_gate_passed"]
                       and item["robustness_gate_passed"]], key=candidate_rank)
    selected = eligible[0] if eligible else None
    promoted = False
    http_verification = None
    if args.promote and selected:
        offline = Path("reports") / f"phase4_{selected['name']}_offline.csv"
        http_verification = candidate_http("phase4_final", Path(selected["artifact"]), offline)
        if (not http_verification.get("offline_equivalent")
                or http_verification["correct"] != selected["scenarios"][0]["correct"]):
            raise RuntimeError("Candidate failed HTTP equivalence; champion was not replaced")
        shutil.copyfile(selected["artifact"], "artifacts/model.joblib")
        promoted = True
    report = {
        "status": "complete", "champion_sha256": phase3["selected_model_sha256"],
        "champion_scenarios": champion_scenarios, "champion_worst_case_correct": champion_worst,
        "champion_calibration": calibration_diagnostic(y_val, champion_probability),
        "training": {"seed": SEED, "augmentations_per_train_call": TRAIN_AUGMENTATIONS,
                     "continuous_shift_intervals_s": TRAIN_SHIFT_INTERVALS,
                     "held_out_fixed_shifts_s": EVALUATION_SHIFTS[1:],
                     "note": "Fixed evaluation points and +/-0.1 s neighborhoods are excluded from training."},
        "selection_rule": "Require clean >=68/71 and worst-case accuracy across clean/-0.5/-1.0/-1.5 strictly above champion; then maximize worst case, clean accuracy, Brier and compactness in that order.",
        "candidates": candidates, "selected": selected["name"] if selected else None,
        "promoted": promoted, "http_verification": http_verification,
        "validation_limitation": "Official val has already been used for Phase 3 block selection; all Phase 4 results remain exploratory until hidden evaluation.",
        "vad_cache": cache,
    }
    Path("reports/phase4_robust.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": report["selected"], "promoted": promoted,
                      "champion_worst_case_correct": champion_worst,
                      "candidates": [{"name": item["name"], "clean": item["scenarios"][0]["correct"],
                                      "worst_case": item["worst_case_correct"]} for item in candidates]}, indent=2))


if __name__ == "__main__":
    main()
