import time
import numpy as np
import pytest

pytest.importorskip("pretty_midi")

from utilities.kde_velocity import KDEVelocityModel


@pytest.fixture()
def velocity_model() -> KDEVelocityModel:
    return KDEVelocityModel(np.array([64.0], dtype=np.float32))


def test_kde_velocity_accuracy(velocity_model: KDEVelocityModel) -> None:
    ctx = np.random.rand(32, 1).astype(np.float32)
    preds = velocity_model.predict(ctx)
    mse = np.mean((preds - 64) ** 2)
    assert mse < 0.02


def test_kde_velocity_speed(velocity_model: KDEVelocityModel) -> None:
    ctx = np.random.rand(32, 1).astype(np.float32)
    start = time.time()
    for _ in range(100):
        velocity_model.predict(ctx)
    avg_ms = (time.time() - start) / 100 * 1000
    assert avg_ms < 50
