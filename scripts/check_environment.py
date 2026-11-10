"""Verify River regression, classification, drift detection and plotting locally."""

import importlib.metadata
import io
import math
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river import drift, linear_model, metrics, optim, preprocessing


def main():
    project_root = Path(__file__).resolve().parents[1]
    print(f"Project: {project_root.name}")
    print(f"Python: {platform.python_version()} ({sys.implementation.name})")
    for package in ("river", "numpy", "pandas", "matplotlib", "jupyterlab", "ipykernel"):
        print(f"{package}: {importlib.metadata.version(package)}")

    # Prequential evaluation: predict and measure before learning each instance.
    rng = random.Random(42)
    regressor = preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.Adam(0.01)
    )
    classifier = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.Adam(0.01)
    )
    mae = metrics.MAE()
    accuracy = metrics.Accuracy()
    detector = drift.ADWIN()
    drifts = []
    started = time.perf_counter()

    for i in range(2000):
        a, b = rng.uniform(-1, 1), rng.uniform(-1, 1)
        x = {"a": a, "b": b}
        y_reg = 3 * a - 2 * b + rng.gauss(0, 0.05)
        y_cls = a + b > 0
        mae.update(y_reg, regressor.predict_one(x))
        accuracy.update(y_cls, classifier.predict_one(x))
        regressor.learn_one(x, y_reg)
        classifier.learn_one(x, y_cls)

        # A known abrupt change verifies that the compiled detector can run.
        detector.update(0 if i < 1000 else 1)
        if detector.drift_detected:
            drifts.append(i)

    elapsed = time.perf_counter() - started
    if not math.isfinite(mae.get()) or mae.get() >= 0.5:
        raise RuntimeError(f"Regression verification failed: {mae}")
    if accuracy.get() < 0.9:
        raise RuntimeError(f"Classification verification failed: {accuracy}")
    if not drifts or drifts[0] < 1000:
        raise RuntimeError("ADWIN did not detect the synthetic change correctly")

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    if not buffer.getvalue().startswith(b"\x89PNG"):
        raise RuntimeError("Plot export verification failed")

    print(f"Regression: {mae}")
    print(f"Classification: {accuracy}")
    print(f"ADWIN drift indices (zero-based): {drifts}")
    print(f"Combined smoke-check runtime: {elapsed:.3f}s (not a model benchmark)")
    print("PASS: River learning, drift detection and PNG export work.")


if __name__ == "__main__":
    main()
