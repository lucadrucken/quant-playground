from math import log, sqrt, exp, erf
from typing import Literal

def _norm_cdf(x: float) -> float:
    """Standardnormal-CDF ohne SciPy (über die Fehlerfunktion)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

def _d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return _d1(S, K, r, q, sigma, T) - sigma * sqrt(T)

def bs_price(
    S: float,   #Spot Underlying 
    K: float,   #Strike 
    r: float,   #risk-free 
    q: float,   #dividend-yield
    sigma: float,   #Volatility 
    T: float,   #Maturity in years
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black–Scholes Preis einer EUROPAEISCHEN Option (kont. r, q).
    - S, K > 0; T in Jahren; sigma >= 0.
    - option_type: "call" oder "put".
    """
    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    # deterministischer Grenzfall sigma=0
    if sigma <= 0.0:
        fwd_disc = S * exp(-q * T) - K * exp(-r * T)
        return max(fwd_disc, 0.0) if option_type == "call" else max(-fwd_disc, 0.0)

    d1 = _d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return S * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        return K * exp(-r * T) * _norm_cdf(-d2) - S * exp(-q * T) * _norm_cdf(-d1)



def binomial_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int = 200,
    option_type: Literal["call", "put"] = "call",
    american: bool = False,
) -> float:
    """
    Cox–Ross–Rubinstein Binomialpreis (europäisch/amerikanisch).
    r, q kontinuierlich; T in Jahren; sigma in Dezimal.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")

    dt = T / steps
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    disc = exp(-r * dt)
    p = (exp((r - q) * dt) - d) / (u - d)
    # numerische Robustheit
    if not (0.0 <= p <= 1.0):
        # bei extremen Parametern kann p leicht außerhalb [0,1] fallen
        p = min(1.0, max(0.0, p))

    # Terminal-Payoffs
    # S_{N,j} = S * u^j * d^{N-j}
    values = []
    for j in range(steps + 1):
        S_T = S * (u ** j) * (d ** (steps - j))
        payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
        values.append(payoff)

    # Backward induction
    for n in range(steps - 1, -1, -1):
        new_vals = []
        for j in range(n + 1):
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                # aktueller Spot am Knoten (n,j): S * u^j * d^{n-j}
                S_nj = S * (u ** j) * (d ** (n - j))
                ex = (S_nj - K) if option_type == "call" else (K - S_nj)
                ex = max(ex, 0.0)
                new_vals.append(max(cont, ex))
            else:
                new_vals.append(cont)
        values = new_vals

    return float(values[0])
