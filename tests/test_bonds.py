# tests/test_bonds.py
import math
from qp import bond_price

def test_par_bond_equals_face_value():
    # 5% Kupon, 5% YTM, jährliche Zahlungen -> Preis ~ Nennwert
    price = bond_price(face_value=1000, maturity=5, coupon_rate=0.05, ytm=0.05, freq=1)
    assert abs(price - 1000) < 1e-8

def test_zero_coupon_reduces_correctly():
    # Kupon 0 
    price = bond_price(face_value=1000, maturity=3, coupon_rate=0.0, ytm=0.04, freq=1)
    assert abs(price - 1000 / (1.04 ** 3)) < 1e-10

def test_price_monotonic_in_yield():
    # Höherer Yield => niedrigerer Preis
    p_low = bond_price(1000, 10, 0.05, 0.04, 2)
    p_high = bond_price(1000, 10, 0.05, 0.06, 2)
    assert p_low > p_high

def test_frequency_effect_consistency():
    # Gleicher Kupon/Yield, aber halbjährlich diskontiert -> Preis leicht anders als annual
    p_annual = bond_price(1000, 5, 0.05, 0.05, 1)
    p_semi   = bond_price(1000, 5, 0.05, 0.05, 2)
    assert p_semi != p_annual
    # und beide sollten nahe am Par sein (Toleranz großzügig)
    assert abs(p_semi - 1000) < 1.0 and abs(p_annual - 1000) < 1e-6

def test_invalid_inputs_raise():
    import pytest
    with pytest.raises(ValueError):
        bond_price(0, 5, 0.05, 0.05, 1)          # face_value <= 0
    with pytest.raises(ValueError):
        bond_price(1000, 0, 0.05, 0.05, 1)       # n < 1 (nach Rundung)
    with pytest.raises(ValueError):
        bond_price(1000, 5, 0.05, 0.05, 0)       # freq < 1
