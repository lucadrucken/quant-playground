import numpy as np

from qp.metrics import sharpe


def test_sharpe_returns_float():
    r = np.array([0.01, -0.02, 0.015, 0.005])
    s = sharpe(r)
    assert isinstance(s, float)


def test_sharpe_handles_zero_vol():
    s = sharpe([0.0, 0.0, 0.0])
    assert isinstance(s, float)
