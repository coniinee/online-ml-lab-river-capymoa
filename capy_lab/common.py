"""Shared schemas, causal scaling and data with class labels reserved for evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .runtime import ROOT
from river_lab.data import load_stream


def classification_data(name):
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    item = next(d for d in manifest["datasets"] if d["dataset"] == name)
    rows = load_stream(item)
    features = sorted({key for x, _ in rows for key in x})
    labels = item["class_labels"]
    mapping = {label: i for i, label in enumerate(labels)}
    x = np.asarray([[row.get(key, 0.0) for key in features] for row, _ in rows], dtype=np.float64)
    y = np.asarray([mapping[label] for _, label in rows], dtype=np.int64)
    meta = dict(item, feature_names=features, dense_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
    return x, y, meta


def schema_for(n_features, n_classes=2):
    from capymoa.stream import Schema
    return Schema.from_custom(features=[f"x{i}" for i in range(n_features)] + ["class"], target="class",
                              categories={"class": [str(i) for i in range(n_classes)]})


class CausalScaler:
    def __init__(self, dimension):
        self.n = 0
        self.mean = np.zeros(dimension)
        self.m2 = np.zeros(dimension)

    def transform(self, x):
        variance = self.m2 / self.n if self.n else np.ones_like(self.mean)
        scale = np.sqrt(np.where(variance > 0, variance, 1.0))
        return (x - self.mean) / scale

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)


def clustering_data(name, config):
    if name != "MovingGaussians":
        x, y, meta = classification_data(name)
        return x, y, {**meta, "labels_used_for_training": False}
    rng = np.random.default_rng(config["cluster_synthetic_seed"])
    n = config["cluster_synthetic_n"]
    labels = rng.integers(0, 3, size=n)
    means = np.array([[-4., -2.], [0., 3.], [4., -2.]])
    shifts = [np.zeros((3, 2)), np.array([[2., 0.], [-2., 0.], [0., 2.]]),
              np.array([[0., 2.], [2., -1.], [-2., -1.]])]
    x = np.empty((n, 2))
    for i, label in enumerate(labels):
        phase = min(i // (n // 3), 2)
        x[i] = means[label] + shifts[phase][label] + rng.normal(0, 0.45, 2)
    meta = {"dataset": name, "n_samples": n, "kind": "synthetic", "known_drift_positions": [n // 3, 2 * n // 3],
            "labels_used_for_training": False, "dense_sha256": hashlib.sha256(x.tobytes() + labels.tobytes()).hexdigest()}
    path = ROOT / "data/capymoa/MovingGaussians.npz"
    if not path.exists():
        np.savez_compressed(path, x=x, evaluation_labels=labels)
    else:
        with np.load(path) as cached:
            if not np.array_equal(cached['x'],x) or not np.array_equal(cached['evaluation_labels'],labels):
                raise ValueError('MovingGaussians cache differs from configured generator; remove it explicitly before regenerating')
    return x, labels, meta
