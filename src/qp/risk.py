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





