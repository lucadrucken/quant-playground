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
    Historical (non-parametric) Value-at-Risk für 1 Periode.

    Parameters
    ----------
    returns : Iterable[float] | np.ndarray
        Periodische einfache Renditen ODER Verluste (siehe `input_is_loss`).
    level : float, default 0.99
        Konfidenzniveau (z. B. 0.95 oder 0.99).
        Beispiel: 0.99 → 99%-VaR, d. h. die schlechtesten 1% werden überschritten.
    input_is_loss : bool, default False
        False: `returns` werden als Renditen interpretiert und zu Verlusten negiert.
        True: `returns` werden als Verluste interpretiert (keine Negation).
    method : str, default "linear"
        Quantil-Methode für `numpy.quantile` (z. B. "linear", "lower",
        "higher", "midpoint", "nearest").

    Returns
    -------
    float
        VaR als positiver Verlust.
        Gibt `np.nan` zurück, wenn keine Werte vorhanden sind.

    Notes
    -----
    - 1-Perioden-VaR (keine Skalierung auf Tages-/Wochen-/Monats-Horizonte).
    - NaNs werden entfernt.
    """
    x = np.asarray(list(returns), dtype=float)  # list() erlaubt auch Generatoren
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")

    losses = x if input_is_loss else -x
    q = float(np.clip(level, 0.0, 1.0))

    # numpy >=1.22 benutzt "method", ältere Versionen "interpolation"
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
    returns : Iterable[float] | np.ndarray
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


