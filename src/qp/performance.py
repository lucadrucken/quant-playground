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
