# tests/test_bonds.py
import numpy as np
import pytest
from qp import bond_price, macaulay_duration, modified_duration, dollar_duration, convexity

# simple bond params (annual payments)
FV   = 1000
MAT  = 5
CR   = 0.05   # 5% coupon
YTM  = 0.05   # 5% yield
FREQ = 1      # annual

def test_par_bond_equals_face_value():
    # 5% coupon, 5% YTM, annual payments -> price ~ face value
    price = bond_price(face_value=FV, maturity=MAT, coupon_rate=CR, ytm=YTM, freq=FREQ)
    assert abs(price - FV) < 1e-8

def test_zero_coupon_reduces_correctly():
    # zero coupon reduces to discounted face value
    price = bond_price(face_value=FV, maturity=3, coupon_rate=0.0, ytm=0.04, freq=FREQ)
    assert abs(price - FV / (1.04 ** 3)) < 1e-10

def test_price_monotonic_in_yield():
    # Higher yield -> lower price (monotonicity)
    p_low  = bond_price(FV, MAT, CR, 0.04, FREQ)
    p_high = bond_price(FV, MAT, CR, 0.06, FREQ)
    assert p_low > p_high

def test_frequency_effect_consistency():
    # Same coupon/yield but semiannual discounting vs annual -> slight difference
    p_annual = bond_price(FV, MAT, CR, YTM, 1)
    p_semi   = bond_price(FV, MAT, CR, YTM, 2)
    assert p_semi != p_annual
    # both should be near par for CR == YTM
    assert abs(p_annual - FV) < 1e-6
    assert abs(p_semi   - FV) < 1.0   # a few cents difference allowed

def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bond_price(0, MAT, CR, YTM, FREQ)        # face_value must be > 0
    with pytest.raises(ValueError):
        bond_price(FV, 0, CR, YTM, FREQ)         # maturity must be > 0
    with pytest.raises(ValueError):
        bond_price(FV, MAT, CR, YTM, 0)          # freq must be positive integer   


def test_macaulay_duration_positive():
    """Macaulay duration should be positive and <= maturity."""
    d_mac = macaulay_duration(FV, MAT, CR, YTM, FREQ)
    assert d_mac > 0
    assert d_mac <= MAT

def test_modified_duration_relation():
    """Modified duration = Macaulay / (1 + y/freq)."""
    d_mac = macaulay_duration(FV, MAT, CR, YTM, FREQ)
    d_mod = modified_duration(FV, MAT, CR, YTM, FREQ)
    assert np.isclose(d_mod, d_mac / (1 + YTM / FREQ), rtol=1e-8)

def test_dollar_duration_matches_product():
    """Dollar duration ≈ Modified Duration * Price."""
    price = bond_price(FV, MAT, CR, YTM, FREQ)
    d_mod = modified_duration(FV, MAT, CR, YTM, FREQ)
    d_dol = dollar_duration(FV, MAT, CR, YTM, FREQ)
    assert np.isclose(d_dol, price * d_mod, rtol=1e-8)

def test_convexity_positive():
    """Convexity should be positive for a standard coupon bond."""
    cx = convexity(FV, MAT, CR, YTM, FREQ)
    assert cx > 0