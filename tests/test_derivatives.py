import math
import pytest
from qp import bs_price, binomial_price, bs_greeks, put_from_call, call_from_put, parity_gap, parity_bounds 
from math import exp

def test_put_call_parity_with_dividends():
    S, K, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.2, 1.0
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put")
    # C - P = S e^{-qT} - K e^{-rT}
    assert abs((c - p) - (S*exp(-q*T) - K*exp(-r*T))) < 1e-6

def test_limits_T_zero_and_sigma_zero():
    S, K, r, q, T = 100, 90, 0.02, 0.00, 1.0
    # T=0 -> Intrinsic
    assert bs_price(S, K, r, q, 0.2, 0.0, "call") == max(S-K, 0.0)
    # sigma=0 -> deterministic forward difference (discounted)
    assert bs_price(S, K, r, q, 0.0, T, "call") == max(S*exp(-q*T) - K*exp(-r*T), 0.0)

S, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.2, 1.0

def test_convergence_to_black_scholes():
    """Binomial model price for a European call should approximate Black–Scholes."""
    bs_val = bs_price(S, K, r, q, sigma, T, option_type="call")
    binom_val = binomial_price(S, K, r, q, sigma, T, steps=1000,
                               option_type="call", american=False)
    assert abs(bs_val - binom_val) < 1e-2

def test_american_call_equals_european_without_dividends():
    """American Call without dividends = European Call."""
    call_eu = binomial_price(S, K, r, q, sigma, T, steps=300,
                             option_type="call", american=False)
    call_am = binomial_price(S, K, r, q, sigma, T, steps=300,
                             option_type="call", american=True)
    assert abs(call_eu - call_am) < 1e-6

def test_american_put_greater_than_european():
    """American Put >= European Put."""
    put_eu = binomial_price(S, K, r, q, sigma, T, steps=300,
                            option_type="put", american=False)
    put_am = binomial_price(S, K, r, q, sigma, T, steps=300,
                            option_type="put", american=True)
    assert put_am >= put_eu

def test_put_call_parity_european():
    """European Put-Call-Parity should work for binomial model."""
    call = binomial_price(S, K, r, q, sigma, T, steps=300,
                          option_type="call", american=False)
    put  = binomial_price(S, K, r, q, sigma, T, steps=300,
                          option_type="put", american=False)
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-2


def test_bs_greeks_gamma_vega_symmetry():
    S, K, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.2, 1.0
    gc = bs_greeks(S, K, r, q, sigma, T, "call")
    gp = bs_greeks(S, K, r, q, sigma, T, "put")
    assert gc["gamma"] == pytest.approx(gp["gamma"], rel=1e-12, abs=1e-12)
    assert gc["vega"]  == pytest.approx(gp["vega"], rel=1e-12, abs=1e-12)

def test_bs_greeks_delta_matches_fd():
    S, K, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.2, 1.0
    g = bs_greeks(S, K, r, q, sigma, T, "call")
    h = 1e-4 
    
    p_up = bs_price(S + h, K, r, q, sigma, T, "call")
    p_down = bs_price(S - h, K, r, q, sigma, T, "call")
    delta_fd = (p_up - p_down) / (2 * h)
    assert g["delta"] == pytest.approx(delta_fd, rel=1e-9, abs=1e-10)

def test_parity_inversion_put_from_call_and_call_from_put():
    S, K, r, q, sigma, T = 100, 100, 0.02, 0.01, 0.25, 1.0
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put")
    assert put_from_call(c, S, K, r, q, T) == pytest.approx(p, rel=1e-12, abs=1e-12)
    assert call_from_put(p, S, K, r, q, T) == pytest.approx(c, rel=1e-12, abs=1e-12)

def test_parity_gap_near_zero_for_bs_price():
    S, K, r, q, sigma, T = 100, 100, 0.03, 0.00, 0.2, 0.75
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put") 
    assert parity_gap(c, p, S, K, r, q, T) == pytest.approx(0.0, rel=1e-12, abs=1e-12)

def test_parity_bounds_contain_bs_price():
    S, K, r, q, sigma, T = 100, 100, 0.03, 0.00, 0.2, 0.75
    c = bs_price(S, K, r, q, sigma, T, "call")
    p = bs_price(S, K, r, q, sigma, T, "put") 
    (call_lb, call_ub), (put_lb, put_ub) = parity_bounds(S, K, r, q, T)
    assert call_lb <= c <= call_ub
    assert put_lb <= p <= put_ub



    

