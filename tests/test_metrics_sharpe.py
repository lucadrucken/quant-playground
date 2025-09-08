import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.metrics import sharpe


def test_sharpe_scalar_vs_series_rf_equal():
    r = np.array([0.01, -0.02, 0.015, 0.005])
    rf_daily = 0.04 / 252
    s1 = sharpe(r, risk_free=rf_daily)
    s2 = sharpe(r, risk_free=np.full_like(r, rf_daily, dtype=float))
    assert_allclose(s1, s2, rtol=1e-12, atol=1e-12)


def test_sharpe_handles_nans_and_ddof1():
    r = np.array([0.01, np.nan, -0.02, 0.015])
    s = sharpe(r, risk_free=0.0, ddof=1)
    assert isinstance(s, float)


def test_sharpe_raises_on_shape_mismatch():
    r = np.array([0.01, 0.02, 0.03])
    rf = np.array([0.0001, 0.0001])
    with pytest.raises(ValueError):
        sharpe(r, risk_free=rf)
