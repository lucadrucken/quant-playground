import numpy as np

# annualisierte sharpe ratio (konstant oder zeitvariablen risk-free)
# returns: array oder Serie mit periodischen Returns
# risk_free: Skalar oder Serie gleiche Länge wie returns
# periods_per_year: Annualisierungsfaktor
# ddof: 1 = Stichproben-Std


def sharpe(returns, risk_free=0.0, periods_per_year=252, ddof=1):
    r = np.asarray(returns, dtype=float)
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
