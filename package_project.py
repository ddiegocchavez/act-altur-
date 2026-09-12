"""Create a source/model deliverable using an explicit no-dataset allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT_FILES = [
    "README.md", "RESULTS.md", "PLAN.md", "RESEARCH.md", "DATASET_README.md", "Makefile",
    "Dockerfile", "render.yaml", ".gitignore", ".dockerignore", ".python-version", "provenance.py",
    "requirements.txt", "requirements.lock", "requirements-dev.txt",
    "features.py", "vad.py", "train.py", "metrics.py", "phase1.py", "app.py", "serve.py",
    "evaluate_http.py", "check_endpoint.py", "verify_local.py", "verify_docker.py",
    "verify_public.py", "prepare_data.py", "write_results.py", "package_project.py",
    "behavior_features.py", "phase3.py", "stress_latency.py", "test_behavior.py", "verify_selected.py",
    "audit_shortcuts.py", "phase4_robust.py", "test_robustness.py", "audio_robustness.py",
]
REPORTS = [
    "phase1.json", "phase1_preparation.json", "phase2_http.json", "phase2_edges.json",
    "phase2_clean_http.json", "phase2_clean_edges.json", "phase2_docker.json",
    "phase2_public.json", "phase2_public_edges.json",
    "phase2_public_http.json", "phase3.json", "phase3_final_http.json",
    "phase3_final_clean_http.json", "phase3_final_edges.json", "stress_latency.json",
    "shortcut_audit.json",
    "phase4_robust.json",
    "audio_robustness.json",
    "default_vad_correlations_train.csv", "vad_correlations_train.csv",
    "vad_correlations_val.csv", "vad_grid_train.csv",
]


def main():
    files = [Path(name) for name in ROOT_FILES]
    files += [Path("baseline_original") / name for name in ("features.py", "vad.py", "train.py", "app.py")]
    files += [Path("artifacts/model.joblib"), Path("artifacts/baseline_model.joblib"), Path("artifacts/vad_config.json")]
    files += [Path("reports") / name for name in REPORTS if (Path("reports") / name).is_file()]
    checksums = {}
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"Missing or non-regular source artifact: {path}")
        checksums[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    target = Path("dist/altur-detector.zip")
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, str(Path("altur-detector") / path))
        output.writestr("altur-detector/SHA256SUMS.json", json.dumps(checksums, indent=2) + "\n")
    with zipfile.ZipFile(target) as source:
        assert source.testzip() is None
        assert not any(Path(p).suffix == ".wav" or any(part in ("audio", "turns", "cache", ".venv") for part in Path(p).parts) for p in source.namelist())
    print(json.dumps({"file": str(target.resolve()), "bytes": target.stat().st_size,
                      "entries": len(files) + 1, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
