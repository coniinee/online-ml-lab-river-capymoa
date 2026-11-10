"""Sequential prequential evaluation, paired drift experiments and audit outputs."""

import collections
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import river
from river import metrics, stats, utils

from .data import DATA, ROOT, load_stream
from .models import CLASSIFICATION_MODELS, DETECTORS, REGRESSION_MODELS, make_detector, make_model
from lab_support.identity import calculation_signature, seal, valid_record

RESULTS = ROOT / "results" / "river"


def metric_bundle(task):
    if task == "regression":
        return {"mae": metrics.MAE(), "rmse": metrics.RMSE(), "r2": metrics.R2()}
    return {"accuracy": metrics.Accuracy(), "macro_f1": metrics.MacroF1(), "kappa": metrics.CohenKappa()}


def match_drifts(positions, truth, window):
    """One-to-one first alarm matching inside a declared post-change interval."""
    used, matches = set(), []
    for change in truth:
        candidates = [(j, p) for j, p in enumerate(positions) if j not in used and change <= p <= change + window]
        if candidates:
            j, p = candidates[0]
            used.add(j)
            matches.append({"true_position": change, "detected_position": p, "delay": p - change})
        else:
            matches.append({"true_position": change, "detected_position": None, "delay": None})
    return matches, len(positions) - len(used)


def drift_record(item, seed, detector, mode, events, config):
    positions = [e["position"] for e in events]
    truth = item["known_drift_positions"]
    matches, unmatched = match_drifts(positions, truth, config["drift_match_window"])
    delays = [m["delay"] for m in matches if m["delay"] is not None]
    return {
        "task": item["task"], "dataset": item["dataset"], "seed": seed,
        "detector": detector, "mode": mode, "n_samples": item["n_samples"],
        "n_alarms": len(positions), "alarm_positions": json.dumps(positions),
        "known_changes": len(truth) if truth else None,
        "matched_changes": len(delays) if truth else None,
        "missed_changes": len(truth) - len(delays) if truth else None,
        "unmatched_alarms": unmatched if truth else None,
        "mean_delay": float(np.mean(delays)) if delays else None,
        "matches": json.dumps(matches),
    }


def evaluate(rows, item, model_name, seed, config, reset_detector=None):
    task = item["task"]
    init_start = time.perf_counter()
    model = make_model(task, model_name, seed, config)
    model_repr = repr(model)
    detector = make_detector(reset_detector, seed) if reset_detector else None
    initialization_s = time.perf_counter() - init_start
    cumulative = metric_bundle(task)
    rolling = utils.Rolling(metrics.MAE if task == "regression" else metrics.Accuracy, config["rolling_window"])
    target_var = stats.Var()
    label_counts = collections.Counter()
    mode = "reset" if detector else "baseline"
    identity = {"task": task, "dataset": item["dataset"], "model": model_name,
                "seed": seed, "mode": mode, "detector": reset_detector or "none"}
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    sampled_peak_rss = initial_rss
    latencies, curves, events, losses, signals = [], [], [], [], []
    failures = 0
    started = time.perf_counter()
    for i, (x, y) in enumerate(rows):
        tick = time.perf_counter_ns()
        prediction = model.predict_one(x)
        predict_ns = time.perf_counter_ns() - tick
        if prediction is None:
            # A declared label or a majority based only on labels already seen.
            prediction = label_counts.most_common(1)[0][0] if label_counts else ("brickface" if item["dataset"] == "ImageSegments" else False)
        if task == "regression":
            prediction = float(prediction)
            if not math.isfinite(prediction):
                failures += 1
                raise FloatingPointError(f"Non-finite prediction: {identity}, index {i}")
            loss = abs(float(y) - prediction)
            # Only earlier targets determine scale; no lookahead calibration.
            scale = max(math.sqrt(max(0, target_var.get())), 1.0)
            signal = min(loss / (3 * scale), 1.0)
        else:
            loss = float(prediction != y)
            signal = loss
        for metric in cumulative.values():
            metric.update(y, prediction)
        rolling.update(y, prediction)
        losses.append(loss)
        if model_name == "HoeffdingTree" and not detector:
            signals.append(signal)

        tick = time.perf_counter_ns()
        if detector and i >= config["detector_warmup"]:
            detector.update(signal)
            if detector.drift_detected:
                events.append({**identity, "position": i, "signal": signal})
                # Current prediction is already scored. Recreate the full pipeline,
                # then let the replacement learn this example (never predict it twice).
                model = make_model(task, model_name, seed, config)
        model.learn_one(x, y)
        latencies.append(predict_ns + time.perf_counter_ns() - tick)
        if task == "regression":
            target_var.update(float(y))
        else:
            label_counts[y] += 1

        if (i + 1) % config["checkpoint_every"] == 0 or i == len(rows) - 1:
            sampled_peak_rss = max(sampled_peak_rss, process.memory_info().rss)
            curves.append({**identity, "n_seen": i + 1, "rolling_score": rolling.get(),
                           **{key: value.get() for key, value in cumulative.items()}})
    elapsed = time.perf_counter() - started
    latencies = np.asarray(latencies, dtype=float) / 1000
    summary = {
        **identity, "n_samples": len(rows), "initialization_s": initialization_s,
        **{key: value.get() for key, value in cumulative.items()},
        "total_runtime_s": elapsed, "learner_runtime_s": float(latencies.sum() / 1e6),
        "mean_latency_us": float(latencies.mean()), "p50_latency_us": float(np.median(latencies)),
        "p95_latency_us": float(np.quantile(latencies, 0.95)),
        "sampled_peak_process_rss_mb": sampled_peak_rss / 2**20,
        "sampled_process_rss_increase_mb": max(0, sampled_peak_rss - initial_rss) / 2**20,
        "n_resets": len(events), "nonfinite_predictions": failures,
    }
    # Recovery windows use measured losses, and do not feed information into learners.
    windows = []
    w = config["rolling_window"]
    for change in item["known_drift_positions"]:
        for label, a, b in (("before", max(0, change - w), change),
                            ("immediate_after", change, min(len(rows), change + w)),
                            ("recovery", change + config["drift_match_window"] - w,
                             min(len(rows), change + config["drift_match_window"]))):
            part = losses[a:b]
            if part:
                value = float(np.mean(part))
                windows.append({**identity, "change_position": change, "window": label,
                                "start": a, "end_exclusive": b, "score": value if task == "regression" else 1 - value})
    record = {"summary": summary, "curves": curves, "events": events, "windows": windows,
              "signals": signals, "model_parameters_repr": model_repr}
    if detector:
        record["drift_summary"] = drift_record(item, seed, reset_detector, mode, events, config)
    return record


def monitor(signals, item, seed, detector_name, config, baseline):
    detector = make_detector(detector_name, seed)
    events = []
    started = time.perf_counter()
    for i, signal in enumerate(signals):
        if i >= config["detector_warmup"]:
            detector.update(signal)
            if detector.drift_detected:
                events.append({"task": item["task"], "dataset": item["dataset"], "model": "HoeffdingTree",
                               "seed": seed, "detector": detector_name, "mode": "monitor", "position": i, "signal": signal})
    elapsed = time.perf_counter() - started
    summary = drift_record(item, seed, detector_name, "monitor", events, config)
    summary.update({"detector_only_runtime_s": elapsed,
                    "detector_only_mean_latency_us": elapsed / len(signals) * 1e6,
                    **{k: baseline[k] for k in ("mae", "rmse", "r2", "accuracy", "macro_f1", "kappa") if k in baseline}})
    return {"events": events, "drift_summary": summary}


def fingerprint(config, manifest):
    if manifest != json.loads((DATA/'manifest.json').read_text()):
        raise ValueError('Fingerprint manifest differs from the cached data manifest')
    return calculation_signature(config, 1)


def run_all(config, resume=False):
    for key in ['max_samples','checkpoint_every','rolling_window','drift_match_window','forest_trees','knn_window']:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not config['model_seeds'] or len(config['model_seeds']) != len(set(config['model_seeds'])):
        raise ValueError('Model seeds must be nonempty and unique')
    if any(not isinstance(seed,int) or isinstance(seed,bool) for seed in config['model_seeds']):
        raise ValueError('Model seeds must be integers')
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_dir = RESULTS / "runs"
    run_dir.mkdir(exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config"] != config:
        raise ValueError("Data configuration differs; run the prepare command first")
    for item in manifest['datasets']: load_stream(item)
    signature = fingerprint(config, manifest)
    experiment = {
        "config": config, "fingerprint": signature, "river_version": river.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "physical_memory_gb": psutil.virtual_memory().total / 2**30,
        "evaluation": "Sequential predict, score, detect (optional reset), learn; zero-based drift indices",
        "timing": "perf_counter_ns predict+learn; reset runs also include detection+reinitialization. Metrics, checkpoints and RSS sampling excluded from latency, included in total runtime. Data download/encoding and model initialization excluded. No parallel experiments.",
        "memory": "Process RSS sampled at checkpoints; includes data and Python runtime, not model-exclusive memory, not a guaranteed continuous peak. RSS deltas depend on allocator reuse.",
        "monitoring": "Detectors replay the identical HoeffdingTree prequential error signal, without influencing learning. Detector-only replay time is separate, never reported as directly measured combined model time.",
        "regression_signal": "min(abs(y-yhat)/(3*max(past-target sample std,1)),1); scale uses earlier targets only and is retained across model resets",
        "classification_signal": "0/1 prequential error, including multiclass tasks",
        "cold_start": "Regression predicts 0 from untrained models; a classifier returning None uses the past-label majority, or False/brickface at the first example. All examples are scored.",
        "drift_matching": "First alarm per known synthetic change inside inclusive [change, change+1500]. Unmatched alarms and missed changes retained; no drift truth assumed for real data.",
        "repeats": "Three model/detector seeds on the same cached stream; this is not three independent dataset replications. Deterministic model predictive scores repeat.",
    }
    (RESULTS / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    records, monitor_records = [], []
    started = time.perf_counter()
    for item in manifest["datasets"]:
        rows = load_stream(item)
        task = item["task"]
        names = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
        for seed in config["model_seeds"]:
            jobs = [(name, None) for name in names] + [("HoeffdingTree", d) for d in DETECTORS[task]]
            for name, detector_name in jobs:
                run_id = f"{item['dataset']}__{seed}__{name}__{detector_name or 'baseline'}"
                path = run_dir / f"{run_id}.json"
                record = None
                if resume and path.exists():
                    previous = json.loads(path.read_text())
                    expected={'task':task,'dataset':item['dataset'],'model':name,'seed':seed,'detector':detector_name or 'none','mode':'reset' if detector_name else 'baseline'}
                    if valid_record(previous, signature, 1) and all(previous.get('summary',{}).get(k)==v for k,v in expected.items()):
                        record = previous
                if record is None:
                    print(f"RUN {run_id}", flush=True)
                    record = evaluate(rows, item, name, seed, config, detector_name)
                    seal(record, signature, 1)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(record, allow_nan=False) + "\n")
                    temporary.replace(path)
                records.append(record)
                s = record["summary"]
                score = f"MAE={s['mae']:.4f}" if task == "regression" else f"accuracy={s['accuracy']:.4f}"
                print(f"  {score}; {s['total_runtime_s']:.2f}s; resets={s['n_resets']}", flush=True)
                if name == "HoeffdingTree" and detector_name is None:
                    for d in DETECTORS[task]:
                        monitor_records.append(monitor(record["signals"], item, seed, d, config, s))
    summaries = pd.DataFrame([r["summary"] for r in records])
    for task in ("regression", "classification"):
        summaries[(summaries.task == task) & (summaries["mode"] == "baseline")].dropna(axis=1, how="all").to_csv(RESULTS / f"{task}_metrics.csv", index=False)
    summaries[summaries["mode"] == "reset"].to_csv(RESULTS / "reset_metrics.csv", index=False)
    pd.DataFrame([p for r in records for p in r["curves"]]).to_csv(RESULTS / "learning_curves.csv", index=False)
    pd.DataFrame([p for r in records for p in r["windows"]]).to_csv(RESULTS / "recovery_windows.csv", index=False)
    pd.DataFrame([p for r in records + monitor_records for p in r.get("events", [])], columns=["task", "dataset", "model", "seed", "mode", "detector", "position", "signal"]).to_csv(RESULTS / "drift_events.csv", index=False)
    drift_rows = []
    for r in records + monitor_records:
        if "drift_summary" in r:
            row = dict(r["drift_summary"])
            if "summary" in r:
                row.update({key: value for key, value in r["summary"].items() if key not in row})
            drift_rows.append(row)
    pd.DataFrame(drift_rows).to_csv(RESULTS / "drift_summary.csv", index=False)
    experiment["runner_wall_time_s"] = time.perf_counter() - started
    experiment["completed_model_runs"] = len(records)
    experiment["completed_monitor_replays"] = len(monitor_records)
    experiment["complete"] = True
    (RESULTS / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    print(f"COMPLETE: {len(records)} model runs, {len(monitor_records)} detector replays", flush=True)
