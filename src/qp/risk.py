# src/qp/risk.py

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
    Annualisierte Sharpe Ratio.

    Parameters
    ----------
    returns : Iterable[float] oder np.ndarray
        Periodische einfache Renditen (z. B. täglich).
    risk_free : float oder ArrayLike, optional (default=0.0)
        Entweder ein konstanter risikofreier Satz pro Periode
        oder eine zeitvariable Serie gleicher Länge wie returns.
    periods_per_year : int, optional (default=252)
        Annualisierungsfaktor (252 für tägliche Daten, 12 für monatliche).
    ddof : int, optional (default=1)
        Freiheitsgrade für die Standardabweichung (Sample-Std).

    Returns
    -------
    float
        Annualisierte Sharpe Ratio. Gibt 0.0 zurück, wenn die
        Standardabweichung 0 oder nicht definiert ist.

    Notes
    -----
    - Der risikofreie Satz muss per Periode angegeben werden.
      Falls du einen annualisierten rf hast, vorher konvertieren.
    - Annualisierung erfolgt mit sqrt(periods_per_year).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]

    if np.isscalar(risk_free):
        excess = r - float(risk_free)
    else:
        rf = np.asarray(risk_free, dtype=float)
        if rf.shape != r.shape:
            raise ValueError("risk_free muss skalar sein oder gleiche Form wie returns")
        excess = r - rf

    mean = np.nanmean(excess)
    std = np.nanstd(excess, ddof=ddof)

    if std == 0 or not np.isfinite(std):
        return 0.0

    return (mean / std) * np.sqrt(periods_per_year)


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
