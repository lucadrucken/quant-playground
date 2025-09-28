import numpy as np
import pytest

from qp import var_historical, es_historical


def test_var_historical_zero_for_zero_returns():
    r = np.zeros(1000)
    assert var_historical(r, level=0.99) == pytest.approx(0.0, abs=1e-12)


def test_var_historical_monotonic_in_level():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 5000)
    v95 = var_historical(r, level=0.95)
    v99 = var_historical(r, level=0.99)
    assert v99 >= v95  # höheres Level => größere (schlimmere) Verlustschwelle


def test_var_historical_matches_when_input_is_loss():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.02, 2000)
    v_as_returns = var_historical(r, level=0.975, input_is_loss=False)
    v_as_losses = var_historical(-r, level=0.975, input_is_loss=True)
    assert v_as_returns == pytest.approx(v_as_losses, rel=1e-12, abs=0)


def test_var_historical_known_quantile_matches_numpy():
    r = np.array([0.01, -0.02, 0.03, -0.05, 0.04], dtype=float)
    # Erwartungswert via numpy (Verluste = -returns)
    try:
        expected = float(np.quantile(-r, 0.95, method="linear"))
    except TypeError:
        expected = float(np.quantile(-r, 0.95, interpolation="linear"))
    got = var_historical(r, level=0.95, method="linear")
    assert got == pytest.approx(expected, abs=1e-12)

def test_es_equals_tail_mean_definition():
    r = np.array([0.01, -0.02, 0.03, -0.05, 0.04])
    v = var_historical(r, level=0.99)
    losses = -r
    tail = losses[losses >= v]
    es = es_historical(r, level=0.99)
    assert es == pytest.approx(tail.mean(), rel=1e-12, abs=1e-12)

def test_es_historical_monotonic_in_level():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 5000)
    es95 = es_historical(r, level=0.95)
    es99 = es_historical(r, level=0.99)
    assert es99 >= es95 

def test_es_degenerate_tail_returns_var():
    r = np.full(10, 0.001)   # alle identisch
    v = var_historical(r, 0.99)
    e = es_historical(r, 0.99)
    assert e == pytest.approx(v, abs=1e-12)



