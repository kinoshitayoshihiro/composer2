import time
import numpy as np
from utilities.ml_velocity import MLVelocityModel


def test_ml_velocity_accuracy():
    ctx = np.random.rand(32, 3).astype(np.float32)
    model = MLVelocityModel()
    preds = model.predict(ctx, cache_key="test")
    mse = np.mean((preds - 64) ** 2)
    assert mse < 0.02


def test_ml_velocity_speed():
    ctx = np.random.rand(32, 3).astype(np.float32)
    model = MLVelocityModel()
    start = time.time()
    for _ in range(100):
        model.predict(ctx, cache_key="speed")
    avg_ms = (time.time() - start) / 100 * 1000
    assert avg_ms < 50
