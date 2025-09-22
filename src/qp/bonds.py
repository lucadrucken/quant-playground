from typing import Union

import numpy as np 

def bond_price(
        face_value: Union[int, float], 
        maturity: Union[int, float], 
        coupon_rate: float,
        ytm: float,
        freq: int = 1, 
) -> float: 
    """"
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

