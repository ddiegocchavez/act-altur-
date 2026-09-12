"""Audit model shortcuts before adding more behavioral features.

The artifact-only section runs without challenge data. Dataset ablations run
only when the locally generated behavior feature cache is present; no audio or
per-call predictions are written to the report.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from metrics import evaluate
from train import MODEL_PARAMETERS


METADATA = ("anon_id", "label", "split")


def artifact_audit(path):
    bundle = joblib.load(path)
    model, names = bundle["model"], bundle["feats"]
    split_count, gain, missing_left, root_count = Counter(), Counter(), Counter(), Counter()
    thresholds = defaultdict(list)
    for stage in model._predictors:
        for tree in stage:
            for node_index, node in enumerate(tree.nodes):
                if node["is_leaf"]:
                    continue
                name = names[int(node["feature_idx"])]
                split_count[name] += 1
                gain[name] += float(node["gain"])
                thresholds[name].append(float(node["num_threshold"]))
                if node["missing_go_to_left"]:
                    missing_left[name] += 1
                if node_index == 0:
                    root_count[name] += 1
    ranked = []
    for name, count in split_count.most_common():
        values = np.asarray(thresholds[name], dtype=float)
        ranked.append({
            "feature": name,
            "splits": count,
            "total_gain": gain[name],
            "root_splits": root_count[name],
            "missing_go_left_splits": missing_left[name],
            "threshold_median": float(np.median(values)),
            "threshold_p10": float(np.percentile(values, 10)),
            "threshold_p90": float(np.percentile(values, 90)),
        })
    return {
        "path": str(path),
        "features": len(names),
        "feature_blocks": bundle.get("feature_blocks", []),
        "iterations": int(model.n_iter_),
        "features_used": len(ranked),
        "ranked_features": ranked,
    }


def feature_groups(frame):
    names = [name for name in frame if name not in METADATA]
    agent = [name for name in names if name in {"n_agent", "agent_speech_s"} or name.startswith("adur_")]
    caller = [name for name in names if name in {"n_caller", "caller_speech_s", "caller_ratio",
              "turns_per_min", "frag_ratio", "ipause_per_min"}
              or name.startswith(("cdur_", "ipause_"))]
    interaction = [name for name in names if name not in set(agent) | set(caller)
                   and name not in {"total_s"}]
    groups = {
        "agent_only": agent,
        "caller_only": caller,
        "interaction_only": interaction,
    }
    if "silence_fill_wait_med" in names:
        groups["silence_fill_wait_med_only"] = ["silence_fill_wait_med"]
    return groups


def fit_subset(frame, feature_names):
    train = frame.loc[frame.split == "train"]
    val = frame.loc[frame.split == "val"]
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    model.fit(train[feature_names].to_numpy(), (train.label == "synthetic").to_numpy())
    probability = model.predict_proba(val[feature_names].to_numpy())[:, 1]
    return {"features": feature_names, "metrics": evaluate((val.label == "synthetic").astype(int), probability)}


def dataset_audit(path):
    if not path.exists():
        return {"status": "pending", "reason": f"Missing local generated cache: {path}"}
    frame = pd.read_csv(path)
    if len(frame.query("split == 'train'")) != 282 or len(frame.query("split == 'val'")) != 71:
        raise ValueError("Unexpected official split sizes in behavior cache")
    results = {name: fit_subset(frame, columns) for name, columns in feature_groups(frame).items() if columns}
    numeric = [name for name in frame if name not in METADATA]
    missing = frame[list(METADATA)].copy()
    missing_names = []
    for name in numeric:
        output_name = name + "__missing"
        missing[output_name] = frame[name].isna().astype(float)
        missing_names.append(output_name)
    results["missingness_only"] = fit_subset(missing, missing_names)
    missing_rates = []
    for name in numeric:
        human = float(frame.loc[frame.label == "human", name].isna().mean())
        synthetic = float(frame.loc[frame.label == "synthetic", name].isna().mean())
        if human or synthetic:
            missing_rates.append({"feature": name, "human": human, "synthetic": synthetic,
                                  "absolute_gap": abs(human - synthetic)})
    missing_rates.sort(key=lambda row: row["absolute_gap"], reverse=True)
    return {"status": "complete", "ablations": results, "missing_rates": missing_rates}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("cache/features_behavior.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/shortcut_audit.json"))
    args = parser.parse_args()
    report = {
        "purpose": "Detect absolute-timing, channel-only and missing-value shortcuts; no model promotion.",
        "artifacts": [artifact_audit(Path(path)) for path in
                      ("artifacts/baseline_model.joblib", "artifacts/model.joblib")],
        "dataset_ablations": dataset_audit(args.features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "dataset_status": report["dataset_ablations"]["status"],
                      "dominant_features": [item["ranked_features"][0]["feature"]
                                            for item in report["artifacts"]]}, indent=2))


if __name__ == "__main__":
    main()
