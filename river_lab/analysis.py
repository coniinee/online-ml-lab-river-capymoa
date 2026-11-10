"""Summarize measured runs and generate report-ready static figures."""

import json
import os

from .data import DATA, ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

OUT = ROOT / "results" / "river"
FIG = OUT / "figures"
COLORS = ["#276FBF", "#E07A26", "#269B73", "#9B59B6"]


def markdown_table(frame, formats=None):
    """A small Markdown formatter without an extra tabulate dependency."""
    formats = formats or {}
    rows = []
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if pd.isna(value):
                values.append("—")
            elif col in formats:
                values.append(format(value, formats[col]))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |", *rows])


def aggregate(frame, groups, columns):
    result = frame.groupby(groups, sort=False)[columns].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(filter(None, col)) if isinstance(col, tuple) else col for col in result.columns]
    return result


def save_figure(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def panels(frame, task, value, ylabel, filename, title):
    names = frame.dataset.drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, name in zip(axes.flat, names):
        part = frame[frame.dataset == name]
        grouped = part.groupby("model", sort=False)[value].agg(["mean", "std"])
        ax.bar(np.arange(len(grouped)), grouped["mean"], yerr=grouped["std"].fillna(0),
               color=COLORS[:len(grouped)], capsize=4, width=0.65)
        ax.set_xticks(np.arange(len(grouped)), grouped.index, rotation=18, ha="right", fontsize=8)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        if value == "mean_latency_us":
            ax.set_yscale("log")
        if task == "classification" and value == "accuracy":
            ax.set_ylim(0, 1)
    fig.suptitle(title + "\nMean and standard deviation across 3 model seeds", fontsize=13)
    save_figure(fig, filename)


def curves_figure(curves, manifest, task, reset=False):
    names = [d["dataset"] for d in manifest["datasets"] if d["task"] == task]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, name in zip(axes.flat, names):
        subset = curves[(curves.task == task) & (curves.dataset == name)]
        if reset:
            subset = subset[subset.model == "HoeffdingTree"].copy()
            subset["series"] = np.where(subset["mode"] == "baseline", "No reset", subset.detector + " + reset")
        else:
            subset = subset[subset["mode"] == "baseline"].copy()
            subset["series"] = subset.model
        for (label, color) in zip(subset.series.drop_duplicates(), COLORS):
            grouped = subset[subset.series == label].groupby("n_seen").rolling_score.agg(["mean", "std"])
            ax.plot(grouped.index, grouped["mean"], label=label, color=color, lw=1.5)
            ax.fill_between(grouped.index, grouped["mean"] - grouped["std"].fillna(0), grouped["mean"] + grouped["std"].fillna(0), color=color, alpha=0.1)
        item = next(d for d in manifest["datasets"] if d["dataset"] == name)
        for j, point in enumerate(item["known_drift_positions"]):
            ax.axvline(point + 1, ls="--", color="black", lw=1, label="Known change" if j == 0 else None)
        ax.set_title(name)
        ax.set_xlabel("Examples processed")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_ylabel("Rolling MAE (last 500)" if task == "regression" else "Rolling accuracy (last 500)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        if task == "classification":
            ax.set_ylim(0, 1.02)
    suffix = "reset_effect" if reset else "learning_curves"
    fig.suptitle(f"{task.title()}: {'drift handling on a common Hoeffding tree' if reset else 'prequential learning curves'}", fontsize=13)
    save_figure(fig, f"{task}_{suffix}")


def alarm_figure(events, manifest, task, seed):
    names = [d["dataset"] for d in manifest["datasets"] if d["task"] == task]
    detectors = ["ADWIN", "PageHinkley", "KSWIN"] if task == "regression" else ["ADWIN", "DDM", "EDDM"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), layout="constrained")
    for ax, name in zip(axes.flat, names):
        for level, detector in enumerate(detectors):
            points = events[(events.dataset == name) & (events["mode"] == "monitor") & (events.seed == seed) & (events.detector == detector)].position
            ax.scatter(points, [level] * len(points), marker="|", s=130, color=COLORS[level], label=detector)
        item = next(d for d in manifest["datasets"] if d["dataset"] == name)
        for point in item["known_drift_positions"]:
            ax.axvline(point, color="black", ls="--", alpha=0.7)
        ax.set_xlim(0, item["n_samples"])
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks(range(3), detectors)
        ax.set_title(name)
        ax.set_xlabel("Zero-based sample index")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(f"{task.title()}: alarms on identical tree errors (seed {seed})\nDashed lines: known synthetic changes; real-data truth unspecified", fontsize=12)
    save_figure(fig, f"{task}_drift_alarms")


def validate_outputs(reg, clf, resets, drift_table, events, manifest, experiment):
    seeds = experiment["config"]["model_seeds"]
    expected = 4 * 4 * len(seeds)
    checks = {
        "all_regression_baselines_present": len(reg) == expected,
        "all_classification_baselines_present": len(clf) == expected,
        "all_reset_comparisons_present": len(resets) == 8 * 3 * len(seeds),
        "all_detector_comparisons_present": len(drift_table) == 8 * 3 * 2 * len(seeds),
        "all_runs_have_finite_metrics": all(np.isfinite(df[cols].to_numpy()).all() for df, cols in [(reg, ["mae", "rmse", "r2", "mean_latency_us"]), (clf, ["accuracy", "macro_f1", "kappa", "mean_latency_us"])]),
        "classification_scores_in_bounds": bool(clf[["accuracy", "macro_f1"]].ge(0).all().all() and clf[["accuracy", "macro_f1"]].le(1).all().all()),
        "all_alarms_accounted_for": int(drift_table.n_alarms.sum()) == len(events),
        "all_experiments_marked_complete": experiment.get("complete", False),
    }
    sizes = {d["dataset"]: d["n_samples"] for d in manifest["datasets"]}
    checks["identical_sample_counts_for_each_dataset"] = all(int(r.n_samples) == sizes[r.dataset] for df in (reg, clf, resets) for r in df.itertuples())
    checks["all_seed_model_combinations_unique"] = all(not df.duplicated(["dataset", "model", "seed", "detector"]).any() for df in (reg, clf, resets))
    (OUT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks


def analyze():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    manifest = json.loads((DATA / "manifest.json").read_text())
    experiment = json.loads((OUT / "experiment.json").read_text())
    reg = pd.read_csv(OUT / "regression_metrics.csv")
    clf = pd.read_csv(OUT / "classification_metrics.csv")
    resets = pd.read_csv(OUT / "reset_metrics.csv")
    drift_table = pd.read_csv(OUT / "drift_summary.csv")
    events = pd.read_csv(OUT / "drift_events.csv")
    curves = pd.read_csv(OUT / "learning_curves.csv")
    from lab_support.identity import valid_record
    from lab_support.validation import validate_exports
    from .evaluation import fingerprint
    records=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*.json')]
    signature=fingerprint(experiment['config'],manifest)
    if len(records)!=168 or not all(valid_record(r,signature,1) for r in records):
        raise ValueError('River raw records are stale, altered or incomplete')
    validate_exports(OUT,records,1)
    checks = validate_outputs(reg, clf, resets, drift_table, events, manifest, experiment)
    checks['sealed_raw_records_and_all_exports_match']=True
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    for task, frame, scores in (("regression", reg, ["mae", "rmse", "r2"]), ("classification", clf, ["accuracy", "macro_f1", "kappa"])):
        cols = scores + ["total_runtime_s", "mean_latency_us", "p95_latency_us", "sampled_peak_process_rss_mb", "sampled_process_rss_increase_mb"]
        aggregate(frame, ["dataset", "model"], cols).to_csv(OUT / f"{task}_summary.csv", index=False)
        value, label = ("mae", "MAE (lower is better)") if task == "regression" else ("accuracy", "Accuracy (higher is better)")
        panels(frame, task, value, label, f"{task}_performance", f"{task.title()}: predictive performance")
        panels(frame, task, "mean_latency_us", "Latency (µs, log scale)", f"{task}_latency", f"{task.title()}: processing cost")
        curves_figure(curves, manifest, task)
        curves_figure(curves, manifest, task, reset=True)
        alarm_figure(events, manifest, task, experiment["config"]["model_seeds"][0])
    aggregate(drift_table, ["task", "dataset", "detector", "mode"], ["n_alarms", "matched_changes", "missed_changes", "unmatched_alarms", "mean_delay"]).to_csv(OUT / "detector_summary.csv", index=False)
    baseline = pd.concat([reg, clf], ignore_index=True)
    tree = baseline[baseline.model == "HoeffdingTree"]
    paired = resets.merge(tree, on=["task", "dataset", "seed"], suffixes=("_reset", "_baseline"))
    paired["mae_change"] = paired.mae_reset - paired.mae_baseline
    paired["accuracy_change"] = paired.accuracy_reset - paired.accuracy_baseline
    paired["latency_ratio"] = paired.mean_latency_us_reset / paired.mean_latency_us_baseline
    paired.to_csv(OUT / "reset_effect_pairs.csv", index=False)
    aggregate(paired, ["task", "dataset", "detector_reset"], ["mae_change", "accuracy_change", "latency_ratio"]).to_csv(OUT / "reset_effect_summary.csv", index=False)

    from scripts.english_findings import write_river
    write_river(ROOT)
    print(f'Analysis complete: {FIG}, findings.md, summary CSVs; {len(checks)} checks passed.')
