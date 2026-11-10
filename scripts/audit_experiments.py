"""Independent audit of measured evidence; never overwrite formal runs or reports.

Run snapshot for all records; replay uses seed 42 on full cached streams in fresh
processes and recomputes predictive metrics with NumPy/scikit-learn, not River.
Replay time is deliberately not compared with formal benchmark time.
"""

import argparse
import ast
import base64
import collections
import hashlib
import importlib.metadata as metadata
import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/audit"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def close(a, b):
    import numpy as np
    if a is None or b is None:
        return (a is None or bool(np.isscalar(a) and not isinstance(a, str) and np.isnan(a))) and (b is None or bool(np.isscalar(b) and not isinstance(b, str) and np.isnan(b)))
    if isinstance(a, (bool, str, list, dict)):
        return a == b
    return bool(np.isclose(a, b, rtol=1e-10, atol=1e-11, equal_nan=True))


def compare_mapping(actual, expected, keys=None):
    return {k: {"actual": actual.get(k), "expected": expected.get(k)} for k in (keys or expected) if not close(actual.get(k), expected.get(k))}


def snapshot():
    import numpy as np
    import pandas as pd
    from river_lab.data import load_stream, synthetic
    from river_lab.evaluation import fingerprint
    from capy_lab.runtime import configure
    from capy_lab.experiments import compatible
    config = configure()
    manifest = read(ROOT / "data/manifest.json")
    p1config = read(ROOT / "config/river.json")
    issues = []
    checks = {}
    datasets = {}
    for item in manifest["datasets"]:
        rows = load_stream(item)
        assert all(math.isfinite(v) for x, _ in rows for v in x.values())
        assert len(rows) == item["n_samples"]
        assert len({k for x, _ in rows for k in x}) == item["encoded_n_features"]
        if item["kind"] == "synthetic":
            regenerated = list(synthetic(item["dataset"], len(rows), p1config["stream_seed"]))
            normalized = [({str(k): v for k, v in x.items()}, y) for x, y in regenerated]
            assert normalized == rows, item["dataset"]
        datasets[item["dataset"]] = {"samples": len(rows), "sha256": item["sha256_uncompressed"]}
    checks["all_eight_caches_and_both_synthetic_regenerations"] = True
    p1 = [read(p) for p in (ROOT / "results/river/runs").glob("*.json")]
    p2 = [read(p) for p in (ROOT / "results/capymoa/runs").glob("*.json")]
    expected1 = set()
    for d in manifest["datasets"]:
        models = ["HoeffdingTree", "AdaptiveRandomForest", "LinearAdam", "KNN"] if d["task"] == "regression" else ["HoeffdingTree", "AdaptiveRandomForest", "LogisticAdamOvR", "GaussianNB"]
        detectors = ["ADWIN", "PageHinkley", "KSWIN"] if d["task"] == "regression" else ["ADWIN", "DDM", "EDDM"]
        for seed in p1config["model_seeds"]:
            expected1.update((d["dataset"], m, seed, "baseline", "none") for m in models)
            expected1.update((d["dataset"], "HoeffdingTree", seed, "reset", det) for det in detectors)
    actual1 = [tuple(r["summary"][k] for k in ["dataset", "model", "seed", "mode", "detector"]) for r in p1]
    assert set(actual1) == expected1 and len(actual1) == len(expected1) == 168
    expected2 = set()
    for d, m, lib, seed in itertools.product(config["classification_datasets"], config["families"], ["River", "CapyMOA"], config["seeds"]):
        expected2.add(("classification", d, m, lib, seed))
    for stage, ds, ms in [("clustering", config["cluster_datasets"], config["cluster_models"]), ("automl", config["automl_datasets"], config["automl_models"])]:
        expected2.update((stage, d, m, "CapyMOA", seed) for d, m, seed in itertools.product(ds, ms, config["seeds"]))
    actual2 = [tuple(r["summary"][k] for k in ["stage", "dataset", "model", "library", "seed"]) for r in p2]
    assert set(actual2) == expected2 and len(actual2) == len(expected2) == 147
    checks["exact_315_combinations_no_duplicates_or_substitutions"] = True
    code = "import json; from river_lab.evaluation import fingerprint; from lab_support.identity import environment; from pathlib import Path; print(json.dumps({'signature':fingerprint(json.loads(Path('config/river.json').read_text()),json.loads(Path('data/manifest.json').read_text())),'environment':environment(1)}))"
    river_identity=json.loads(subprocess.check_output([str(ROOT/'.venv/bin/python'),'-c',code],cwd=ROOT,text=True))
    sig = river_identity['signature']
    assert all(r["fingerprint"] == sig for r in p1)
    from lab_support.identity import record_digest
    assert all(r.get('record_sha256')==record_digest(r) and r['environment']==river_identity['environment'] for r in p1)
    assert all(compatible(r, config) for r in p2)
    checks["all_current_runtime_calculation_signatures_and_content_seals"] = True
    # Formal JSON summaries and curves must reproduce the exported CSV exactly.
    comparisons = 0
    for relative, records in [("river/regression_metrics", [r for r in p1 if r["summary"]["task"] == "regression" and r["summary"]["mode"] == "baseline"]), ("river/classification_metrics", [r for r in p1 if r["summary"]["task"] == "classification" and r["summary"]["mode"] == "baseline"]), ("river/reset_metrics", [r for r in p1 if r["summary"]["mode"] == "reset"]), *[("capymoa/" + s + "_metrics", [r for r in p2 if r["summary"]["stage"] == s]) for s in ["classification", "clustering", "automl"]]]:
        csv = pd.read_csv(ROOT / "results" / (relative + ".csv"))
        keys = ["dataset", "model", "seed"] + (["mode", "detector"] if relative.startswith("river") else ["library"])
        lookup = {tuple(row[k] for k in keys): row for row in csv.to_dict("records")}
        for r in records:
            s = r["summary"]; row = lookup[tuple(s[k] for k in keys)]
            for k, v in s.items():
                if isinstance(v, (list, dict)):
                    assert json.loads(row[k]) == v
                else:
                    assert close(row[k], v), (relative, keys, k, row[k], v)
                comparisons += 1
    checks["raw_summary_fields_match_six_metrics_csvs"] = comparisons
    for library, records in [("river", p1), ("capymoa", p2)]:
        for r in records:
            s = r["summary"]; curves = r["curves"]
            ns = [p["n_seen"] for p in curves]
            assert ns == sorted(set(ns)) and ns[-1] == s["n_samples"]
            for k in ["accuracy", "macro_f1", "kappa", "mae", "rmse", "r2"]:
                if k in s: assert close(curves[-1][k], s[k]), (s, k)
            if "accuracy" in s:
                prev_n, prev_correct = 0, 0
                for p in curves:
                    correct = p["accuracy"] * p["n_seen"]
                    assert close(correct, round(correct))
                    assert 0 <= round(correct) - prev_correct <= p["n_seen"] - prev_n
                    prev_n, prev_correct = p["n_seen"], round(correct)
            n = s.get("n_scored", s["n_samples"])
            runtime_key = "learner_runtime_s" if library == "river" else "processing_runtime_s"
            assert close(s[runtime_key], s["mean_latency_us"] * n / 1e6)
            assert s["p50_latency_us"] <= s["p95_latency_us"] and s[runtime_key] <= s["total_runtime_s"] + 1e-6
            if library == "river":
                assert s["n_resets"] == len(r["events"])
            else:
                if s["stage"] == "clustering":
                    assert s["n_scored"] == s["n_samples"] - s["calibration"]
                    for key, field in [("ari_all", "mean_window_ari"), ("silhouette_assigned", "mean_window_silhouette"), ("coverage", "mean_window_coverage"), ("quantization_mse", "mean_window_mse")]:
                        values = [p[key] for p in curves if p[key] is not None]
                        assert close(float(np.mean(values)) if values else None, s[field])
                    for snap in r["snapshots"]:
                        assert np.isfinite(snap["centers"]).all()
    checks["all_curve_endpoints_integer_accuracy_latency_and_cluster_averages"] = True
    # Independent aggregation from each run's evidence, without analysis helpers.
    for lib, source, groups, metrics in [
        ("river", "regression", ["dataset", "model"], ["mae", "rmse", "r2", "mean_latency_us", "total_runtime_s"]),
        ("river", "classification", ["dataset", "model"], ["accuracy", "macro_f1", "kappa", "mean_latency_us", "total_runtime_s"]),
        *[("capymoa", s, ["dataset", "model"] + (["library"] if s == "classification" else []), ["mean_window_ari", "mean_window_silhouette", "mean_latency_us", "total_runtime_s"] if s == "clustering" else ["accuracy", "macro_f1", "kappa", "mean_latency_us", "total_runtime_s"]) for s in ["classification", "clustering", "automl"]]]:
        raw = pd.read_csv(ROOT / "results" / lib / (source + "_metrics.csv"))
        summary = pd.read_csv(ROOT / "results" / lib / (source + "_summary.csv"))
        for row in summary.to_dict("records"):
            part = raw
            for k in groups: part = part[part[k] == row[k]]
            assert len(part) == 3
            for m in metrics:
                for suffix, value in [("mean", part[m].mean()), ("std", part[m].std(ddof=1))]:
                    assert close(row[m + "_" + suffix], value), (lib, source, m, suffix)
    checks["five_summary_tables_independent_mean_and_sample_sd"] = True
    # Installed Python sources must match distribution RECORD, not just version strings.
    installed = []
    for env, packages in [(".venv", ["river", "numpy"]), (".venv-capy", ["river", "capymoa", "numpy", "pandas", "scikit-learn"] )]:
        for d in metadata.distributions(path=[str(ROOT / env / "lib/python3.11/site-packages")]):
            name = d.metadata["Name"].lower().replace("_", "-")
            if name not in packages: continue
            checked = 0
            for p in d.files or []:
                if p.suffix != ".py" or not p.hash: continue
                actual = base64.urlsafe_b64encode(hashlib.new(p.hash.mode, d.locate_file(p).read_bytes()).digest()).rstrip(b"=").decode()
                assert actual == p.hash.value, (env, str(p))
                checked += 1
            installed.append({"environment": env, "package": name, "version": d.version, "verified_python_files": checked})
    checks["installed_distribution_record_integrity"] = True
    write(OUT / "snapshot.json", {"checks": checks, "datasets": datasets, "installed": installed, "issues": issues})
    print(json.dumps(checks, indent=2), flush=True)


def replay(job):
    from capy_lab.runtime import configure
    config = configure()
    import numpy as np
    seed = job["seed"]
    random.seed(seed); np.random.seed(seed)
    if job["part"] == 1:
        from river_lab.data import load_stream
        from river_lab.models import make_model, make_detector
        manifest = read(ROOT / "data/manifest.json")
        cfg = read(ROOT / "config/river.json")
        item = next(d for d in manifest["datasets"] if d["dataset"] == job["dataset"])
        rows = load_stream(item)
        task = item["task"]
        detector = make_detector(job["detector"], seed) if job["detector"] else None
        model = make_model(task, job["model"], seed, cfg)
        counts = collections.Counter(); truth = []; predictions = []; alarms = []; replay_signals = []
        mean = 0.; m2 = 0.; n = 0
        for i, (x, y) in enumerate(rows):
            pred = model.predict_one(x)
            if pred is None: pred = counts.most_common(1)[0][0] if counts else ("brickface" if item["dataset"] == "ImageSegments" else False)
            truth.append(y); predictions.append(pred)
            if task == "regression":
                scale = max(math.sqrt(max(0, m2 / (n - 1) if n > 1 else 0)), 1.)
                signal = min(abs(y - pred) / (3 * scale), 1.)
            else: signal = int(pred != y)
            replay_signals.append(signal)
            if detector and i >= cfg["detector_warmup"]:
                detector.update(signal)
                if detector.drift_detected:
                    alarms.append(i); model = make_model(task, job["model"], seed, cfg)
            model.learn_one(x, y)
            if task == "regression":
                n += 1; delta = y - mean; mean += delta / n; m2 += delta * (y - mean)
            else: counts[y] += 1
        record = read(ROOT / "results/river/runs" / f"{job['dataset']}__{seed}__{job['model']}__{job['detector'] or 'baseline'}.json")
        if task == "regression":
            y = np.array(truth); pred = np.array(predictions); loss = np.abs(y - pred)
            metrics = {"mae": float(loss.mean()), "rmse": float(np.sqrt(np.mean((y - pred)**2))), "r2": float(1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2))}
        else:
            loss = np.array(truth) != np.array(predictions)
            metrics = classification_metrics(truth, predictions)
        differences = compare_mapping(metrics, record["summary"], metrics)
        if record["signals"]:
            assert np.allclose(replay_signals, record["signals"], rtol=1e-11, atol=1e-11)
        assert alarms == [e["position"] for e in record["events"]]
        for w in record["windows"]:
            value = float(np.mean(loss[w["start"]:w["end_exclusive"]]))
            if task != "regression": value = 1 - value
            assert close(value, w["score"]), (job, w)
        for point in record["curves"]:
            end = point["n_seen"]; expected = float(np.mean(loss[max(0, end - cfg["rolling_window"]):end]))
            if task != "regression": expected = 1 - expected
            assert close(expected, point["rolling_score"])
            if task == "regression":
                y_prefix = np.asarray(truth[:end]); p_prefix = np.asarray(predictions[:end])
                partial = {"mae": float(np.mean(np.abs(y_prefix - p_prefix))), "rmse": float(np.sqrt(np.mean((y_prefix - p_prefix)**2))), "r2": float(1 - np.sum((y_prefix - p_prefix)**2) / np.sum((y_prefix - y_prefix.mean())**2))}
            else:
                partial = classification_metrics(truth[:end], predictions[:end])
            assert not compare_mapping(partial, point, partial), (job, point, partial)
        extra = {"alarm_positions_match": True, "rolling_scores_and_recovery_windows_match": True}
    elif job["stage"] == "clustering":
        from capy_lab.experiments import clustering_job
        from unittest.mock import patch
        record = read(ROOT / "results/capymoa/runs" / (job["id"] + ".json"))
        with patch("capy_lab.common.np.savez_compressed"):
            current = clustering_job({k: v for k, v in job.items() if k != "part"}, config)
        keys = ["mean_window_ari", "mean_window_silhouette", "mean_window_coverage", "mean_window_mse", "final_micro_clusters", "n_scored"]
        metrics = {k: current["summary"][k] for k in keys}
        differences = compare_mapping(metrics, record["summary"], metrics)
        curve_differences = [{"n_seen": a["n_seen"], "differences": compare_mapping(a, b)} for a, b in zip(current["curves"], record["curves"])]
        write(OUT / "diagnostics" / (job["id"] + ".json"), {"summary_differences": differences, "curve_differences": curve_differences, "replayed_curves": current["curves"], "original_curves": record["curves"]})
        assert len(current["curves"]) == len(record["curves"]) and all(not p["differences"] for p in curve_differences), job
        # Centres may differ by insignificant floating-point noise.
        assert all(a["n_seen"] == b["n_seen"] and np.allclose(a["centers"], b["centers"], rtol=1e-10, atol=1e-10) for a, b in zip(current["snapshots"], record["snapshots"]))
        extra = {"full_clustering_replay_curves_and_snapshots_match": True}
    else:
        from capy_lab.common import classification_data, schema_for
        from capy_lab.models import classifier, automl
        from capymoa.core import LabeledInstance
        x, truth, meta = classification_data(job["dataset"])
        schema = schema_for(x.shape[1], len(meta["class_labels"]))
        is_capy = job["library"] == "CapyMOA"
        if job["stage"] == "automl": model = automl(job["model"], schema, seed, len(truth))
        else: model = classifier(job["library"], job["model"], schema if is_capy else None, x.shape[1], seed, config)
        predictions = []; means = np.zeros(x.shape[1]); m2 = np.zeros(x.shape[1]); count = 0
        linear = job["model"] == "LinearLogistic"
        eliminated_predictions = []; selection = []
        for i, (row, label) in enumerate(zip(x, truth)):
            if linear:
                variance = m2 / count if count else np.ones_like(means)
                before = (row - means) / np.sqrt(np.where(variance > 0, variance, 1.))
            else: before = row
            if job["model"] == "SuccessiveHalving" and model._best_model_idx not in model._rankings[:model._s]:
                eliminated_predictions.append(i)
            pred = model.predict(LabeledInstance.from_array(schema, before, 0)) if is_capy else model.predict_one(dict(enumerate(before)))
            predictions.append(0 if pred is None else int(pred))
            if linear:
                count += 1; delta = row - means; means += delta / count; m2 += delta * (row - means)
                after = (row - means) / np.sqrt(np.where(m2 / count > 0, m2 / count, 1.))
            else: after = row
            if is_capy: model.train(LabeledInstance.from_array(schema, after, int(label)))
            else: model.learn_one(dict(enumerate(after)), int(label))
            if job["stage"] == "automl" and (i + 1) % 200 == 0:
                state = {"n_seen": i + 1}
                if job["model"] == "Bandit":
                    actual = model.policy.get_best_arm_idx(range(len(model.active_models)))
                    state.update({"prediction_arm": actual, "reported_best_model_index": model.get_model_info()["best_model_index"], "policy": model.policy.get_arm_stats(), "candidates": [{"index": j, "model": str(m), "accuracy": model.metrics[j].accuracy()} for j, m in enumerate(model.active_models)]})
                elif job["model"] == "SuccessiveHalving":
                    state.update({"prediction_arm": model._best_model_idx, "active_indices": model._rankings[:model._s]})
                selection.append(state)
        record = read(ROOT / "results/capymoa/runs" / (job["id"] + ".json"))
        metrics = classification_metrics(truth, predictions)
        differences = compare_mapping(metrics, record["summary"], metrics)
        digest = hashlib.sha256(np.asarray(predictions, dtype=np.int32).tobytes()).hexdigest()
        write(OUT / "diagnostics" / (job["id"] + ".json"), {"replayed_metrics": metrics, "original_metrics": {k: record["summary"][k] for k in metrics}, "differences": differences, "replayed_prediction_sha256": digest, "original_prediction_sha256": record["prediction_sha256"]})
        assert digest == record["prediction_sha256"], (job, digest, record["prediction_sha256"])
        for point in record["curves"]:
            end = point["n_seen"]
            rolling = float(np.mean(np.asarray(predictions[max(0, end - config["rolling_window"]):end]) == truth[max(0, end - config["rolling_window"]):end]))
            assert close(rolling, point["rolling_accuracy"])
            # Independently verify all saved checkpoint metrics, including early class discovery.
            partial = classification_metrics(truth[:end], predictions[:end])
            assert not compare_mapping(partial, point, partial), (job, point, partial)
        extra = {"prediction_sha256_match": True, "all_checkpoint_metrics_match_independent_formulas": True, "predictions_from_eliminated_arm_count": len(eliminated_predictions), "first_eliminated_arm_positions": eliminated_predictions[:10]}
        if job["stage"] == "automl": write(OUT / "selector_states" / (job["id"] + ".json"), selection)
    assert not differences, (job, differences)
    result = {"job": job, "metrics": metrics, "differences": differences, **extra}
    name = job.get("id") or f"river__{job['dataset']}__{seed}__{job['model']}__{job['detector'] or 'baseline'}"
    write(OUT / "replays" / (name + ".json"), result)
    print("PASS", job.get("id", str(job)), flush=True)


def classification_metrics(truth, predictions):
    import numpy as np
    # Formula from the confusion matrix, with the union of observed true/predicted labels.
    labels = sorted(set(truth) | set(predictions), key=str)
    indexes = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for y, pred in zip(truth, predictions): cm[indexes[y], indexes[pred]] += 1
    support = cm.sum(axis=1); predicted = cm.sum(axis=0); tp = np.diag(cm)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(labels), dtype=float), where=denominator != 0)
    n = cm.sum(); acc = tp.sum() / n
    pe = float(support @ predicted) / (n * n)
    return {"accuracy": float(acc), "macro_f1": float(f1.mean()), "kappa": float((acc - pe) / (1 - pe)) if pe != 1 else 0.}


def run_replays():
    manifest = read(ROOT / "data/manifest.json")
    cfg = read(ROOT / "config/capymoa.json")
    jobs = []
    for item in manifest["datasets"]:
        models = ["HoeffdingTree", "AdaptiveRandomForest", "LinearAdam", "KNN"] if item["task"] == "regression" else ["HoeffdingTree", "AdaptiveRandomForest", "LogisticAdamOvR", "GaussianNB"]
        jobs.extend({"part": 1, "dataset": item["dataset"], "model": m, "seed": 42, "detector": None} for m in models)
        if item["kind"] == "synthetic":
            ds = ["ADWIN", "PageHinkley", "KSWIN"] if item["task"] == "regression" else ["ADWIN", "DDM", "EDDM"]
            jobs.extend({"part": 1, "dataset": item["dataset"], "model": "HoeffdingTree", "seed": 42, "detector": det} for det in ds)
    for stage, datasets, models, libraries in [("classification", cfg["classification_datasets"], cfg["families"], ["River", "CapyMOA"]), ("clustering", cfg["cluster_datasets"], cfg["cluster_models"], ["CapyMOA"]), ("automl", cfg["automl_datasets"], cfg["automl_models"], ["CapyMOA"])]:
        for ds, m, lib in itertools.product(datasets, models, libraries):
            job = {"part": 2, "stage": stage, "dataset": ds, "model": m, "library": lib, "seed": 42}
            job["id"] = "__".join(str(job[k]) for k in ["stage", "dataset", "seed", "model", "library"])
            jobs.append(job)
    results = []
    for index, job in enumerate(jobs, 1):
        name = job.get("id") or f"river__{job['dataset']}__42__{job['model']}__{job['detector'] or 'baseline'}"
        path = OUT / "jobs" / (name + ".json")
        write(path, job)
        print(f"REPLAY {index}/{len(jobs)} {name}", flush=True)
        with (OUT / "jobs" / (name + ".log")).open("w") as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "replay", "--job", str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        results.append({"name": name, "passed": completed.returncode == 0})
        write(OUT / "replay_status.json", {"planned": len(jobs), "completed": len(results), "passed": sum(r["passed"] for r in results), "results": results})
        if completed.returncode: print("FAIL", (OUT / "jobs" / (name + ".log")).read_text()[-3000:], flush=True)
    print(f"Completed {len(jobs)} full-stream replays; passed {sum(r['passed'] for r in results)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["snapshot", "replays", "replay"])
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "snapshot": snapshot()
    elif args.command == "replay": replay(read(args.job))
    else: run_replays()
