from math import log, sqrt, exp, erf, pi
from typing import Literal, Tuple, Dict

def _norm_cdf(x: float) -> float:
    """Standardnormal-CDF über die Fehlerfunktion"""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)

def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

def _d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return _d1(S, K, r, q, sigma, T) - sigma * sqrt(T)

def bs_price(
    S: float,   #Spot Underlying 
    K: float,   #Strike 
    r: float,   #risk-free 
    q: float,   #dividend-yield
    sigma: float,  #Volatility 
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
    d2 = _d2(S, K, r, q, sigma, T)

    if option_type == "call":
        return S * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        return K * exp(-r * T) * _norm_cdf(-d2) - S * exp(-q * T) * _norm_cdf(-d1)


def bs_greeks(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call",
) -> Dict[str, float]:
    """
    Black–Scholes Greeks für europäische Optionen (kont. r, q).
    Returns dict: price, delta, gamma, vega, theta, rho, phi

    Einheiten:
      - vega: pro 1.00 (=100%-Punkte) Vol-Änderung; pro 1% ≈ vega/100
      - theta: pro Jahr; pro Tag ≈ theta/365
      - rho/phi: pro 1.00 Zins-/Dividend-Änderung
    """

    price = bs_price(S, K, r, q, sigma, T, option_type)

    if T <= 0.0 or sigma <= 0.0:
        return {"price": float(price), 
                "delta": 0.0, 
                "gamma": 0.0, 
                "vega": 0.0, 
                "theta": 0.0, 
                "rho": 0.0, 
                "phi": 0.0}

    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(S, K, r, q, sigma, T)

    Nd1, Nd2 = _norm_cdf(d1), _norm_cdf(d2)
    nd1 = _norm_pdf(d1)
    disc_r, disc_q = exp(-r * T), exp(-q * T)

    gamma = disc_q * nd1 / (S * sigma * sqrt(T))
    vega  = S * disc_q * nd1 * sqrt(T)

    if option_type == "call":
        delta = disc_q * Nd1
        theta = (- S * disc_q * nd1 * sigma / (2.0 * sqrt(T))
                 - r * K * disc_r * Nd2
                 + q * S * disc_q * Nd1)
        rho   = K * T * disc_r * Nd2
        phi   = - T * S * disc_q * Nd1
    else:
        delta = disc_q * (Nd1 - 1.0)  # = -disc_q * N(-d1)
        theta = (- S * disc_q * nd1 * sigma / (2.0 * sqrt(T))
                 + r * K * disc_r * (1.0 - Nd2)
                 - q * S * disc_q * (1.0 - Nd1))
        rho   = - K * T * disc_r * (1.0 - Nd2)  # = -K T e^{-rT} N(-d2)
        phi   = + T * S * disc_q * (1.0 - Nd1)  # = +T S e^{-qT} N(-d1)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta),
        "rho":   float(rho),
        "phi":   float(phi),
    }


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



def put_from_call(
        call: float, 
        S: float, 
        K: float, 
        r: float, 
        q: float, 
        T: float
    ) -> float:
    """Berechne europäischen Put aus Call-Preis (Put–Call-Parität)."""
    return call - S * exp(-q * T) + K * exp(-r * T)

def call_from_put(
        put: float, 
        S: float, 
        K: float, 
        r: float, 
        q: float, 
        T: float
    ) -> float:
    """Berechne europäischen Call aus Put-Preis (Put–Call-Parität)."""
    return put + S * exp(-q * T) - K * exp(-r * T)

def parity_gap(
        call: float, 
        put: float, 
        S: float, 
        K: float, 
        r: float, 
        q: float, 
        T: float
    ) -> float:
    """
    Differenz zwischen LHS und RHS der Put–Call-Parität.
    0 ~ exakt erfüllt; ≠0 bedeutet Verletzung.
    """
    lhs = call - put
    rhs = S * exp(-q * T) - K * exp(-r * T)
    return lhs - rhs

def parity_bounds(
        S: float, 
        K: float, 
        r: float, 
        q: float, 
        T: float
    )-> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Berechne obere und untere Grenzen für europäische Call- und Put-Preise.
    Returns: (call_lower, call_upper), (put_lower, put_upper)
    """
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)

    call_lb = max(0.0, S * disc_q - K * disc_r)
    call_ub = S * disc_q

    put_lb  = max(0.0, K * disc_r - S * disc_q)
    put_ub  = K * disc_r

    return (call_lb, call_ub), (put_lb, put_ub)



