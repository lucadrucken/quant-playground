from typing import Iterable, Union

import numpy as np

ArrayLike = Union[Iterable[float], np.ndarray]


def sharpe(
    returns: ArrayLike,
    risk_free: Union[float, ArrayLike] = 0.0,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like of float
        Period returns (e.g., daily simple returns).
    risk_free : float or array-like, default 0.0
        Risk-free rate per period. Either a constant or a time series
        of the same length as `returns`. If you have an annualized rate,
        convert to per-period before calling.
    periods_per_year : int, default 252
        Annualization factor (252 for daily, 12 for monthly, etc.).
    ddof : int, default 1
        Delta degrees of freedom for standard deviation (sample = 1, population = 0).

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if the standard deviation
        is zero or undefined.

    Notes
    -----
    - Risk-free rate must be specified per period.
    - Annualization is performed by multiplying the mean excess return by
    `periods_per_year` and scaling volatility by `sqrt(periods_per_year)`.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]

    if np.isscalar(risk_free):
        excess = r - float(risk_free)
    else:
        rf = np.asarray(risk_free, dtype=float)
        if rf.shape != r.shape:
            raise ValueError("risk_free must be scalar or have the same shape as returns")
        excess = r - rf

    mean = np.nanmean(excess)
    std = np.nanstd(excess, ddof=ddof)

    if std == 0 or not np.isfinite(std):
        return 0.0

    return (mean / std) * np.sqrt(periods_per_year)
