from typing import Union
import numpy as np

def bond_price(face_value: Union[int, float],
               maturity: Union[int, float],
               coupon_rate: float,
               ytm: float,
               freq: int = 1) -> float:
    """
    Price of a plain-vanilla fixed-coupon bond with `freq` payments per year.

    Parameters
    ----------
    face_value : int or float
        Redemption value (e.g., 100 or 1000).
    maturity : int or float
        Time to maturity in years.
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.05 for 5%).
    ytm : float
        Annual yield-to-maturity as a decimal (nominal with compounding `freq`).
    freq : int, default 1
        Coupon payments per year (1=annual, 2=semiannual, 4=quarterly).

    Returns
    -------
    float
        Clean price (present value of coupons + redemption), same unit as `face_value`.

    Notes
    -----
    - Uses `n = round(maturity * freq)` periods (no accrued interest; no dirty/clean split).
    - Assumes a flat yield and level coupons.
    """
    if face_value <= 0:
        raise ValueError("face_value must be > 0")
    if maturity <= 0:
        raise ValueError("maturity must be > 0 (years)")
    if not isinstance(freq, int) or freq < 1:
        raise ValueError("freq must be a positive integer")

    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity * freq must be >= 1")

    c = face_value * coupon_rate / freq
    y = ytm / freq
    if np.isclose(1.0 + y, 0.0):
        raise ValueError("ytm leads to division by zero")

    t = np.arange(1, n + 1, dtype=float)
    disc = (1.0 + y) ** t
    pv_coupons = np.sum(c / disc)
    pv_face = face_value / ((1.0 + y) ** n)
    return float(pv_coupons + pv_face)


def macaulay_duration(face_value: Union[int, float],
                      maturity: Union[int, float],
                      coupon_rate: float,
                      ytm: float,
                      freq: int = 1) -> float:
    """
    Macaulay duration in years.

    Parameters
    ----------
    face_value : int or float
        Redemption value
    maturity : int or float
        Time to maturity in years
    coupon_rate : float
        Annual coupon rate
    ytm : float
        Annual yield-to-maturity
    freq : int, default 1
        Coupon payments per year

    Returns
    -------
    float
        Macaulay duration (cash-flow weighted average time in years).

    Notes
    -----
    Uses `n = round(maturity * freq)` and nominal compounding at `freq`.
    """
    if face_value <= 0:
        raise ValueError("face_value must be > 0")
    if maturity <= 0:
        raise ValueError("maturity must be > 0 (years)")
    if not isinstance(freq, int) or freq < 1:
        raise ValueError("freq must be a positive integer")

    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity * freq must be >= 1")

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
    return float(np.sum(weights_time_years) / P)


def modified_duration(face_value: Union[int, float],
                      maturity: Union[int, float],
                      coupon_rate: float,
                      ytm: float,
                      freq: int = 1) -> float:
    """
    Modified duration (per 1.00 change in yield, annualized).

    Parameters
    ----------
    face_value : int or float
        Redemption value
    maturity : int or float
        Time to maturity in years
    coupon_rate : float
        Annual coupon rate
    ytm : float
        Annual yield-to-maturity
    freq : int, default 1
        Coupon payments per year

    Returns
    -------
    float
        Modified duration.

    Notes
    -----
    `D_mod = D_mac / (1 + ytm/freq)`.
    Approximates proportional price change: ΔP / P ≈ -D_mod * Δy (for small Δy).
    """
    D_mac = macaulay_duration(face_value, maturity, coupon_rate, ytm, freq)
    return float(D_mac / (1.0 + ytm / freq))


def dollar_duration(face_value: Union[int, float],
                    maturity: Union[int, float],
                    coupon_rate: float,
                    ytm: float,
                    freq: int = 1) -> float:
    """
    Dollar duration (PV01-scaled), i.e., -dP/dy.

    Parameters
    ----------
    face_value : int or float
        Redemption value
    maturity : int or float
        Time to maturity in years
    coupon_rate : float
        Annual coupon rate
    ytm : float
        Annual yield-to-maturity
    freq : int, default 1
        Coupon payments per year

    Returns
    -------
    float
        Price change per 1.00 change in yield (positive for plain bonds).

    Notes
    -----
    Also: `DollarDuration ≈ ModifiedDuration * Price`.
    """
    P = bond_price(face_value, maturity, coupon_rate, ytm, freq)
    D_mod = modified_duration(face_value, maturity, coupon_rate, ytm, freq)
    return float(P * D_mod)


def convexity(face_value: Union[int, float],
              maturity: Union[int, float],
              coupon_rate: float,
              ytm: float,
              freq: int = 1) -> float:
    """
    Convexity (per 1.00 change in yield, annualized).

    Parameters
    ----------
    face_value : int or float
        Redemption value
    maturity : int or float
        Time to maturity in years
    coupon_rate : float
        Annual coupon rate
    ytm : float
        Annual yield-to-maturity
    freq : int, default 1
        Coupon payments per year

    Returns
    -------
    float
        Convexity.

    Notes
    -----
    Discrete-time definition with nominal compounding `freq`. Improves the
    Taylor approximation: P(y+Δy) ≈ P - D_mod*P*Δy + 0.5*Convexity*P*(Δy)^2.
    """
    if face_value <= 0:
        raise ValueError("face_value must be > 0")
    if maturity <= 0:
        raise ValueError("maturity must be > 0 (years)")
    if not isinstance(freq, int) or freq < 1:
        raise ValueError("freq must be a positive integer")

    n = int(round(maturity * freq))
    if n < 1:
        raise ValueError("maturity * freq must be >= 1")

    F = float(face_value)
    c = F * coupon_rate / freq
    y = ytm / freq

    t = np.arange(1, n + 1, dtype=float)
    cf = np.full(n, c, dtype=float)
    cf[-1] += F

    disc_pow_t = (1.0 + y) ** t
    P = float(np.sum(cf / disc_pow_t))

    num = np.sum(cf * t * (t + 1.0) / ((1.0 + y) ** (t + 2.0)))
    return float(num / (P * (freq ** 2)))
