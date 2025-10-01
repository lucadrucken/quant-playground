from typing import Iterable, Union

import numpy as np

ArrayLike = Union[Iterable[float], np.ndarray]


def var_historical(
    returns: ArrayLike,
    level: float = 0.99,
    input_is_loss: bool = False,
    method: str = "linear",
) -> float:
    """
    Historical (non-parametric) Value-at-Risk for 1 period.

    Parameters
    ----------
   returns : array-like
        Period returns or losses (see `input_is_loss`).
    level : float, default 0.99
        Confidence level (e.g. 0.99 for 99%).
    input_is_loss : bool, default False
        False: `returns` are treated as returns and negated to losses.
        True: `returns` are treated as losses (no negation).
    method : str, default "linear"
        Quantile method for `numpy.quantile`.

    Returns
    -------
    float
        VaR as positive loss.

    Notes
    -----
    - One-period VaR (no scaling to daily/weekly/monthly horizons).
    - NaN values are removed before computation.
    """
    x = np.asarray(list(returns), dtype=float) 
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")

    losses = x if input_is_loss else -x
    q = float(np.clip(level, 0.0, 1.0))

    # calculate VaR at the level
    try:
        v = np.quantile(losses, q, method=method)
    except TypeError:
        v = np.quantile(losses, q, interpolation=method)

    return float(v)


def es_historical(
    returns: ArrayLike,
    level: float = 0.99,
    input_is_loss: bool = False,
    method: str = "linear",
) -> float:
    """
    Historical Expected Shortfall (Conditional VaR) for 1 period.

    Parameters
    ----------
    returns : array-like
        Period returns or losses (see `input_is_loss`).
    level : float, default 0.99
        Confidence level (e.g. 0.95 or 0.99).
    input_is_loss : bool, default False
        False: `returns` are treated as returns and negated to losses.
        True: `returns` are treated as losses (no negation).
    method : str, default "linear"
        Quantile method for `numpy.quantile`.

    Returns
    -------
    float
        Expected Shortfall (positive loss).
    
    Notes
    -----
    - One-period Expected Shortfall (no scaling to daily/weekly/monthly horizons).
    - NaN values are removed before computation.
    """
    x = np.asarray(list(returns), dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")

    losses = x if input_is_loss else -x
    q = float(np.clip(level, 0.0, 1.0))

    # calculate VaR at the level
    try:
        var = np.quantile(losses, q, method=method)
    except TypeError:
        var = np.quantile(losses, q, interpolation=method)

    # select tail losses beyond VaR
    tail_losses = losses[losses >= var]
    if tail_losses.size == 0:
        return float(var)

    return float(np.mean(tail_losses))


