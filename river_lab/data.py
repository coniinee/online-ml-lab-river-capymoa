"""Download, encode and cache a shared ordered stream for every comparison."""

import datetime as dt
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

from river import datasets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
os.environ["RIVER_DATA"] = str(DATA / "raw")

SPECS = {
    "TrumpApproval": ("regression", datasets.TrumpApproval, [], "River packaged FiveThirtyEight approval data"),
    "ChickWeights": ("regression", datasets.ChickWeights, ["chick", "diet"], "River packaged longitudinal chick weights"),
    "Bikes": ("regression", datasets.Bikes, ["station"], "https://maxhalford.github.io/files/datasets/toulouse_bikes.zip"),
    "FriedmanDrift": ("regression", None, [], "River synth.FriedmanDrift, global recurring abrupt (gra)"),
    "Elec2": ("classification", datasets.Elec2, ["day"], "https://maxhalford.github.io/files/datasets/electricity.zip"),
    "Phishing": ("classification", datasets.Phishing, [], "River packaged UCI website phishing data"),
    "ImageSegments": ("classification", datasets.ImageSegments, [], "River packaged UCI image segmentation data (7 classes)"),
    "SEARecurring": ("classification", None, [], "River synth.SEA variant 0 with abrupt label inversion and recurrence"),
}


def encode_features(x, categorical):
    """Stateless encoding; no fitted statistics or future labels are used."""
    out = {}
    for raw_key, value in x.items():
        key = str(raw_key)
        if isinstance(value, dt.datetime):
            hour = value.hour + value.minute / 60
            out[key + "__hour_sin"] = math.sin(2 * math.pi * hour / 24)
            out[key + "__hour_cos"] = math.cos(2 * math.pi * hour / 24)
            out[key + "__weekday_sin"] = math.sin(2 * math.pi * value.weekday() / 7)
            out[key + "__weekday_cos"] = math.cos(2 * math.pi * value.weekday() / 7)
            out[key + "__ordinal"] = float(value.toordinal())
        elif value is None:
            out[key + "__missing"] = 1.0
        elif key in categorical or isinstance(value, str):
            out[f"{key}={value}"] = 1.0
        else:
            number = float(value)
            if math.isfinite(number):
                out[key] = number
            else:
                out[key + "__missing"] = 1.0
    return out


def synthetic(name, n, seed):
    # Zero-based positions: the first example under the new concept has this index.
    changes = [n // 3, 2 * n // 3]
    if name == "FriedmanDrift":
        source = datasets.synth.FriedmanDrift(drift_type="gra", position=tuple(changes), seed=seed)
        yield from itertools.islice(source, n)
    else:
        source = datasets.synth.SEA(variant=0, noise=0.05, seed=seed)
        for i, (x, y) in enumerate(itertools.islice(source, n)):
            yield x, bool(y) ^ (changes[0] <= i < changes[1])


def prepare(config):
    cache = DATA / "processed"
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (task, factory, categorical, source) in SPECS.items():
        print(f"Preparing {name} ...", flush=True)
        n = config["max_samples"]
        dataset = factory() if factory else None
        stream = iter(dataset) if dataset else synthetic(name, n, config["stream_seed"])
        path = cache / f"{name}.jsonl.gz"
        digest = hashlib.sha256()
        feature_names, labels = set(), set()
        count = 0
        with gzip.open(path, "wt", encoding="utf-8") as output:
            for x, y in itertools.islice(stream, n):
                x = encode_features(x, categorical)
                line = json.dumps([x, y], sort_keys=True, ensure_ascii=False) + "\n"
                digest.update(line.encode())
                output.write(line)
                feature_names.update(x)
                if task == "classification":
                    labels.add(y)
                count += 1
        item = {
            "dataset": name, "task": task, "kind": "real" if factory else "synthetic",
            "n_samples": count, "original_n_samples": dataset.n_samples if dataset else n,
            "encoded_n_features": len(feature_names), "source": source,
            "categorical_columns": categorical, "sha256_uncompressed": digest.hexdigest(),
            "cache": str(path.relative_to(ROOT)), "stream_seed": config["stream_seed"] if not factory else None,
            "known_drift_positions": [n // 3, 2 * n // 3] if not factory else [],
            "class_labels": sorted(labels, key=str) if task == "classification" else [],
            "order": "original supplied order; no shuffling" if factory else "generated order",
            "temporal_note": "Static benchmark order, not a validated temporal process" if name in ("Phishing", "ImageSegments") else "Use supplied order; no real-data drift ground truth assumed",
        }
        records.append(item)
        print(f"  cached {count:,} samples, {len(feature_names)} encoded features", flush=True)
    manifest = {"config": config, "datasets": records}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_stream(item):
    path = ROOT / item["cache"]
    digest, rows = hashlib.sha256(), []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            digest.update(line.encode())
            x, y = json.loads(line)
            rows.append((x, y))
    if digest.hexdigest() != item["sha256_uncompressed"] or len(rows) != item["n_samples"]:
        raise ValueError(f"Cache verification failed: {item['dataset']}")
    return rows
