import numpy as np
import pytest

from qp.risk import sharpe


def test_sharpe_zero_if_returns_equal_rf():
    """Wenn returns == risk_free → Sharpe muss 0 sein."""
    returns = np.full(100, 0.01)
    rf = 0.01
    result = sharpe(returns, risk_free=rf, periods_per_year=252)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_sharpe_positive_for_positive_excess_returns():
    """Wenn returns im Schnitt > rf → Sharpe > 0."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 1000)  # leicht positive Renditen
    result = sharpe(returns, risk_free=0.0)
    assert result > 0


def test_sharpe_negative_for_negative_excess_returns():
    """Wenn returns im Schnitt < rf → Sharpe < 0."""
    rng = np.random.default_rng(1)
    returns = rng.normal(-0.001, 0.01, 1000)  # leicht negative Renditen
    result = sharpe(returns, risk_free=0.0)
    assert result < 0


def test_sharpe_handles_array_rf():
    """Sharpe soll auch mit zeitvariablem rf laufen."""
    returns = np.array([0.01, 0.02, 0.015])
    rf = np.array([0.005, 0.01, 0.005])
    result = sharpe(returns, risk_free=rf, periods_per_year=252)
    assert isinstance(result, float)


def test_sharpe_raises_if_rf_shape_mismatch():
    """Sharpe soll ValueError werfen, wenn risk_free andere Länge als returns hat."""
    returns = np.array([0.01, 0.02, 0.03])
    rf = np.array([0.01, 0.02])  # falsche Länge
    with pytest.raises(ValueError, match="risk_free"):
        sharpe(returns, risk_free=rf)
