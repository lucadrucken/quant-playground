import math
from qp.options import bs_price, binomial_price
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
    # sigma=0 -> deterministisch diskontierter Forward-Unterschied
    assert bs_price(S, K, r, q, 0.0, T, "call") == max(S*exp(-q*T) - K*exp(-r*T), 0.0)

S, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.2, 1.0

def test_convergence_to_black_scholes():
    """Europäischer Call im Binomialmodell sollte für große steps ~ BS sein."""
    bs_val = bs_price(S, K, r, q, sigma, T, option_type="call")
    binom_val = binomial_price(S, K, r, q, sigma, T, steps=1000,
                               option_type="call", american=False)
    assert abs(bs_val - binom_val) < 1e-2

def test_american_call_equals_european_without_dividends():
    """American Call ohne Dividende = European Call."""
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
    """Europäische Put-Call-Parität sollte im Binomialmodell gelten."""
    call = binomial_price(S, K, r, q, sigma, T, steps=300,
                          option_type="call", american=False)
    put  = binomial_price(S, K, r, q, sigma, T, steps=300,
                          option_type="put", american=False)
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-2
