"""Measured, isolated-process classification, clustering and online AutoML."""

import collections
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .runtime import ROOT
from .common import CausalScaler, classification_data, clustering_data, schema_for
from .models import classifier, automl, clusterer
from .configuration import require, validate_config, FAMILIES, AUTOML, CLUSTERS, LIBRARIES
from lab_support.identity import calculation_signature, seal, valid_record

OUT = ROOT / "results/capymoa"


def model_description(model):
    if hasattr(model, "moa_learner"):
        return str(model.moa_learner.getClass().getName()) + " " + str(model.moa_learner.getOptions().getAsCLIString())
    if hasattr(model, "sklearner"):
        return repr(model.sklearner)
    if hasattr(model, "active_models"):
        return json.dumps({'adapter': type(model).__name__, 'candidates': [
            {'id': f'candidate_{i}', 'model': type(m).__name__,
             'parameters': str(m.moa_learner.getOptions().getAsCLIString())}
            for i,m in enumerate(model.active_models)]}, sort_keys=True)
    return repr(model)


def performance(latencies, elapsed, process, rss_start, rss_peak):
    times = np.asarray(latencies) / 1000
    return {"total_runtime_s": elapsed, "processing_runtime_s": float(times.sum() / 1e6),
            "mean_latency_us": float(times.mean()), "p50_latency_us": float(np.median(times)),
            "p95_latency_us": float(np.quantile(times, 0.95)),
            "sampled_peak_process_rss_mib": rss_peak / 2**20,
            "sampled_rss_increase_mib": max(0, rss_peak-rss_start) / 2**20}


def classification_job(job, config):
    from river import metrics
    x, y, meta = classification_data(job["dataset"])
    is_capy = job["library"] == "CapyMOA"
    if is_capy:
        from capymoa.core import LabeledInstance
        schema = schema_for(x.shape[1], len(meta["class_labels"]))
    else:
        schema = None
    init = time.perf_counter()
    model = (automl(job["model"], schema, job["seed"], len(y)) if job["stage"] == "automl" else
             classifier(job["library"], job["model"], schema, x.shape[1], job["seed"], config))
    initialization_s = time.perf_counter() - init
    description = model_description(model)
    scaler = CausalScaler(x.shape[1]) if job["model"] == "LinearLogistic" else None
    scores = {"accuracy": metrics.Accuracy(), "macro_f1": metrics.MacroF1(), "kappa": metrics.CohenKappa()}
    history = collections.deque(maxlen=config["rolling_window"])
    curves, latencies, predictions = [], [], []
    process = psutil.Process(); start_rss = process.memory_info().rss; peak = start_rss
    began = time.perf_counter()
    for i, (row, label) in enumerate(zip(x, y)):
        tick = time.perf_counter_ns()
        before = scaler.transform(row) if scaler else row
        if is_capy:
            # The true label is deliberately absent from the prediction object.
            pred = model.predict(LabeledInstance.from_array(schema, before, 0))
        else:
            pred = model.predict_one(dict(enumerate(before)))
        predict_ns = time.perf_counter_ns() - tick
        pred = 0 if pred is None else int(pred)
        if not 0 <= pred < len(meta["class_labels"]):
            raise ValueError(f"Invalid prediction {pred}")
        for metric in scores.values():
            metric.update(int(label), pred)
        history.append(int(pred == label));predictions.append(pred)
        tick = time.perf_counter_ns()
        if scaler:
            scaler.update(row)
            after = scaler.transform(row)
        else:
            after = row
        if is_capy:
            model.train(LabeledInstance.from_array(schema, after, int(label)))
        else:
            model.learn_one(dict(enumerate(after)), int(label))
        latencies.append(predict_ns + time.perf_counter_ns() - tick)
        if (i + 1) % config["checkpoint_every"] == 0 or i == len(y)-1:
            peak = max(peak, process.memory_info().rss)
            point = {"n_seen": i+1, "rolling_accuracy": float(np.mean(history)),
                     **{k: metric.get() for k, metric in scores.items()}}
            if job["stage"] == "automl":
                info = model.get_model_info() if hasattr(model, "get_model_info") else {'method': job['model'], 'observations': i+1}
                point["selection_state"] = json.dumps(info, default=str)
            curves.append(point)
    elapsed = time.perf_counter()-began
    summary = {**job, "n_samples": len(y), "n_features": x.shape[1], "initialization_s": initialization_s,
               **{k: m.get() for k, m in scores.items()},
               **performance(latencies, elapsed, process, start_rss, peak),
               "data_sha256": meta["dense_sha256"], "known_drift_positions": meta["known_drift_positions"]}
    return {"summary": summary, "curves": curves, "parameters": description,
            "prediction_sha256": hashlib.sha256(np.asarray(predictions,dtype=np.int32).tobytes()).hexdigest(),
            "schema": {"features": meta["feature_names"], "labels": meta["class_labels"]}}


def macro_centers(model, previous, k, seed):
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment
    result = model.get_micro_clustering_result()
    centers = np.asarray(result.get_centers(), dtype=float)
    weights = np.asarray(result.get_weights(), dtype=float)
    if not len(centers):
        return previous, 0
    valid = np.isfinite(centers).all(axis=1) & np.isfinite(weights) & (weights > 0)
    centers, weights = centers[valid], weights[valid]
    count = len(centers)
    if count == 0:
        return previous, 0
    k = min(k, count)
    fresh = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=seed).fit(centers, sample_weight=weights).cluster_centers_
    if previous is not None and len(previous) == len(fresh):
        left, right = linear_sum_assignment(((previous[:,None,:] - fresh[None,:,:])**2).sum(axis=2))
        fresh = fresh[right[np.argsort(left)]]
    return fresh, count


def clustering_job(job, config):
    from capymoa.core import LabeledInstance
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    x, truth, meta = clustering_data(job["dataset"], config)
    calibration = config["cluster_calibration"]
    mu, std = x[:calibration].mean(axis=0), x[:calibration].std(axis=0)
    std = np.where(std > 0, std, 1.0)
    scale = std * math.sqrt(x.shape[1])
    z = (x-mu)/scale
    schema = schema_for(x.shape[1])
    model = clusterer(job["model"], schema, job["seed"])
    description = model_description(model)
    init = time.perf_counter()
    for row in z[:calibration]:
        # MOA ClusTree cannot handle a missing (NaN) class slot. A constant
        # dummy value keeps that slot finite; the true evaluation labels never
        # enter an instance. Centers contain only the feature dimensions.
        model.train(LabeledInstance.from_array(schema, row, 0))
    centers, micro_count = macro_centers(model, None, config["cluster_macro_k"], job["seed"])
    initialization_s = time.perf_counter()-init
    process = psutil.Process(); start_rss = process.memory_info().rss; peak = start_rss
    latencies, curves, snapshots = [], [], []
    assigned, errors = [], []
    began = time.perf_counter()
    for i in range(calibration, len(x)):
        tick = time.perf_counter_ns()
        if centers is None:
            prediction, error = -1, float("nan")
        else:
            distances = ((centers-z[i])**2).sum(axis=1)
            prediction, error = int(np.argmin(distances)), float(distances.min())
        model.train(LabeledInstance.from_array(schema, z[i], 0))
        if (i+1) % config["cluster_refresh"] == 0:
            centers, micro_count = macro_centers(model, centers, config["cluster_macro_k"], job["seed"])
        latencies.append(time.perf_counter_ns()-tick)
        assigned.append(prediction);errors.append(error)
        if (i+1) % config["cluster_eval_every"] == 0 or i == len(x)-1:
            peak = max(peak, process.memory_info().rss)
            a = max(calibration, i+1-config["rolling_window"])
            labels = np.asarray(assigned[a-calibration:i+1-calibration])
            valid = labels >= 0
            silhouette = None
            if valid.sum() > 3 and 1 < len(np.unique(labels[valid])) < valid.sum():
                silhouette = float(silhouette_score(z[a:i+1][valid], labels[valid],
                                    sample_size=min(200,int(valid.sum())), random_state=job["seed"]))
            curves.append({"n_seen": i+1, "ari_all": float(adjusted_rand_score(truth[a:i+1], labels)),
                           "silhouette_assigned": silhouette, "coverage": float(valid.mean()),
                           "quantization_mse": float(np.nanmean(errors[a-calibration:i+1-calibration])) if valid.any() else None,
                           "micro_clusters": micro_count, "macro_clusters": len(centers) if centers is not None else 0})
            snapshots.append({"n_seen": i+1, "centers": (centers*scale+mu).tolist() if centers is not None else []})
    elapsed = time.perf_counter()-began
    def avg(key):
        values = [p[key] for p in curves if p[key] is not None]
        return float(np.mean(values)) if values else None
    summary = {**job, "n_samples": len(x), "n_scored": len(assigned), "calibration": calibration,
               "initialization_s": initialization_s, "mean_window_ari": avg("ari_all"),
               "mean_window_silhouette": avg("silhouette_assigned"), "mean_window_coverage": avg("coverage"),
               "mean_window_mse": avg("quantization_mse"), "final_micro_clusters": micro_count,
               **performance(latencies, elapsed, process, start_rss, peak), "data_sha256": meta["dense_sha256"]}
    return {"summary": summary, "curves": curves, "snapshots": snapshots, "parameters": description,
            "preprocessing": {"calibration_mean": mu.tolist(), "fixed_scale": scale.tolist()}, "dataset": meta}


def signature(config, job=None):
    if job is None:
        return hashlib.sha256(''.join(signature(config, {'stage':s}) for s in ['classification','clustering','automl']).encode()).hexdigest()
    require(job['stage'], {'classification','clustering','automl'}, 'stage')
    return calculation_signature(config, 2, job['stage'])


def compatible(record, config, job=None):
    try:
        summary = record['summary']
        validate_job(summary, config)
        if job is not None and any(summary.get(k) != job.get(k) for k in ['stage','dataset','model','library','seed','id']):
            return False
        return valid_record(record, signature(config, summary), 2)
    except (KeyError, TypeError, ValueError):
        return False


def validate_job(job, config):
    stage=job['stage'];require(stage, {'classification','clustering','automl'}, 'stage')
    require(job['library'], LIBRARIES if stage=='classification' else {'CapyMOA'}, 'library')
    choices = FAMILIES if stage=='classification' else CLUSTERS if stage=='clustering' else AUTOML
    require(job['model'], choices, 'model')
    require(job['dataset'], set(config[{'classification':'classification_datasets','clustering':'cluster_datasets','automl':'automl_datasets'}[stage]]), 'dataset')
    require(job['seed'], set(config['seeds']), 'seed')
    expected='__'.join(str(job[k]) for k in ['stage','dataset','seed','model','library'])
    if job['id'] != expected: raise ValueError('Job id does not match its fields')


def worker(job, config):
    validate_job(job, config)
    random.seed(job["seed"]);np.random.seed(job["seed"])
    result = clustering_job(job,config) if job["stage"] == "clustering" else classification_job(job,config)
    seal(result, signature(config,job), 2)
    path = OUT / "runs" / f"{job['id']}.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, allow_nan=False, default=str)+"\n");temp.replace(path)


def run(config, stages, resume=False):
    validate_config(config)
    for stage in stages: require(stage, {'classification','clustering','automl'}, 'stage')
    # Verify actual cache bytes before any --resume shortcut.
    from river_lab.data import load_stream
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    used=set(config['classification_datasets']+config['automl_datasets']+config['cluster_datasets'])
    for item in manifest['datasets']:
        if item['dataset'] in used: load_stream(item)
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ["runs","jobs","logs"]:(OUT/folder).mkdir(exist_ok=True)
    fingerprint = signature(config)
    jobs = []
    if "classification" in stages:
        for dataset in config["classification_datasets"]:
            for seed in config["seeds"]:
                for index,model in enumerate(config["families"]):
                    libraries = ["River","CapyMOA"] if (seed+index)%2 == 0 else ["CapyMOA","River"]
                    jobs.extend(dict(stage="classification",dataset=dataset,model=model,seed=seed,library=lib) for lib in libraries)
    if "clustering" in stages:
        jobs.extend(dict(stage="clustering",dataset=d,model=m,seed=s,library="CapyMOA") for d in config["cluster_datasets"] for s in config["seeds"] for m in config["cluster_models"])
    if "automl" in stages:
        jobs.extend(dict(stage="automl",dataset=d,model=m,seed=s,library="CapyMOA") for d in config["automl_datasets"] for s in config["seeds"] for m in config["automl_models"])
    started = time.perf_counter(); reused = 0
    for job in jobs:
        job["id"] = "__".join(str(job[k]) for k in ["stage","dataset","seed","model","library"])
        output = OUT / "runs" / f"{job['id']}.json"
        if resume and output.exists() and compatible(json.loads(output.read_text()),config,job):
            reused += 1;continue
        jobfile = OUT/"jobs"/f"{job['id']}.json";jobfile.write_text(json.dumps(job))
        print("RUN",job["id"],flush=True)
        with (OUT/"logs"/f"{job['id']}.log").open("w") as log:
            subprocess.run([sys.executable,str(ROOT/"scripts/run_capy.py"),"worker","--job",str(jobfile)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=config["worker_timeout_s"],cwd=ROOT)
        result = json.loads(output.read_text())["summary"]
        score = result.get("accuracy",result.get("mean_window_ari"))
        print(f"  score={score:.4f}, time={result['total_runtime_s']:.2f}s",flush=True)
    for stage in stages:
        records = [json.loads((OUT/"runs"/f"{j['id']}.json").read_text()) for j in jobs if j["stage"] == stage]
        pd.DataFrame([r["summary"] for r in records]).to_csv(OUT/f"{stage}_metrics.csv",index=False)
        pd.DataFrame([{**{k:r['summary'][k] for k in ['dataset','model','library','seed']},**p} for r in records for p in r["curves"]]).to_csv(OUT/f"{stage}_curves.csv",index=False)
    statefile=OUT/"execution.json"
    state=json.loads(statefile.read_text()) if statefile.exists() else {}
    state.update({"fingerprint":fingerprint,"config":config,"last_invocation_s":time.perf_counter()-started,"last_reused_runs":reused})
    state["completed_stages"]=sorted(set(state.get("completed_stages",[]))|set(stages))
    statefile.write_text(json.dumps(state,indent=2)+"\n")
    print(f"COMPLETE: {len(jobs)} jobs, {reused} reused",flush=True)
