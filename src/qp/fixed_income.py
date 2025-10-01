from typing import Union

import numpy as np 

def bond_price(
        face_value: Union[int, float], 
        maturity: Union[int, float], 
        coupon_rate: float,
        ytm: float,
        freq: int = 1, 
) -> float: 
    """
    Preis eines Plain-Vanilla-Bonds mit 'freq' Zahlungen/Jahr.
    Es wird n = round(maturity*freq) verwendet (kein Accrued/Dirty-Clean).
    """
    if face_value <= 0: 
        raise ValueError("face_value muss > 0 sein")
    if not isinstance(freq, int) or freq < 1: 
        raise ValueError("frew muss eine postitive ganze Zahl sein")

    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity*freq muss >= 1 sein")

    c = coupon_rate * float(face_value) / freq
    y = ytm / freq
    if np.isclose(1 + y, 0.0):
        raise ValueError("ytm führt zu Division durch 0")

    t = np.arange(1, n + 1, dtype=float)
    disc = (1 + y) ** t

    pv_coupons = np.sum(c / disc)
    pv_face = float(face_value) / ((1 + y) ** n)
    return float(pv_coupons + pv_face)

def macaulay_duration(
        face_value: Union[int, float], 
        maturity: Union[int, float], 
        coupon_rate: float,
        ytm: float,
        freq: int = 1, 
) -> float: 
    """
    Macaulay Duration in years.
    - ytm: nominal p.a., with 'freq' compounding
    - freq: Payments/Year (1=annual, 2=semiannual, ...)
    """
    if face_value <= 0: 
        raise ValueError("face_value needs to be > 0")
    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity*freq needs to be >= 1")

    F = float(face_value)
    c = F * coupon_rate / freq 
    y = ytm / freq 

    t = np.arange(1, n + 1, dtype=float)
    cf = np.full(n, c, dtype=float)
    cf[-1] += F
    
    disc = (1.0 + y) ** t
    pv_cf = cf / disc 
    P = float(np.sum(pv_cf))

    weights_time_years = (t / freq) * pv_cf
    D_mac = float(np.sum(weights_time_years)) / P
    return D_mac    

def modified_duration(
        face_value: Union[int, float], 
        maturity: Union[int, float], 
        coupon_rate: float,
        ytm: float,
        freq: int = 1,      
) -> float:
    """
    Modified Duration (Years)
    Relation: D_mod = D_mac / (1 + ytm/freq)
    """
    D_mac = macaulay_duration(face_value, maturity, coupon_rate, ytm, freq)
    y = ytm / freq
    return D_mac / (1 + y)

def dollar_duration(
        face_value: Union[int, float], 
        maturity: Union[int, float], 
        coupon_rate: float,
        ytm: float,
        freq: int = 1,      
) -> float:
    """
    Dollar Duration = D_mod * Price 
    """
    P = bond_price(face_value, maturity, coupon_rate, ytm, freq)
    D_mod = modified_duration(face_value, maturity, coupon_rate, ytm, freq)
    return P * D_mod    

def convexity(
        face_value: Union[int, float], 
        maturity: Union[int, float], 
        coupon_rate: float,
        ytm: float,
        freq: int = 1,
) -> float:
    """
    """
    if face_value <= 0: 
        raise ValueError("face_value needs to be > 0")
    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity*freq needs to be >= 1")
    
    F = float(face_value)
    c = F * coupon_rate / freq 
    y = ytm / freq

    t = np.arange(1, n + 1, dtype=float)
    cf = np.full(n, c, dtype=float)
    cf[-1] += F
    
    disc_pow_t = (1.0 + y) ** t
    P = float(np.sum(cf / disc_pow_t))

    num = np.sum(cf * t * (t + 1) / ((1.0 + y) ** (t + 2)))
    C = float(num / (P *(freq ** 2)))
    return C    