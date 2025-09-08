def sharpe(returns, risk_free=0.0):
    import numpy as np 
    r = np.asarray(returns, dtype=float)
    excess = r - risk_free 
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std): 
        return 0.0 
    return (excess.mean() / std) * (252 ** 0.5) 
