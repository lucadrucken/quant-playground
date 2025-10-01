from math import log, sqrt, exp, erf, pi
from typing import Literal, Tuple, Dict

# --- helpers ---------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x * x)

def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

def _d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return _d1(S, K, r, q, sigma, T) - sigma * sqrt(T)

# --- pricing ---------------------------------------------------------------

def bs_price(
    S: float,   # spot
    K: float,   # strike
    r: float,   # risk-free (cont.)
    q: float,   # dividend yield (cont.)
    sigma: float,  # volatility (annualized)
    T: float,   # time to maturity in years
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Black–Scholes price for a **European** option with continuous `r` and `q`.

    Parameters
    ----------
    S, K : float
        Spot and strike (both > 0).
    r, q : float
        Continuously-compounded risk-free rate and dividend yield (annual).
    sigma : float
        Volatility (annualized, sigma >= 0).
    T : float
        Time to maturity in years (T >= 0).
    option_type : {"call", "put"}, default "call"
        Option type.

    Returns
    -------
    float
        Option price.

    Notes
    -----
    - For `T <= 0`, returns intrinsic value.
    - For `sigma == 0`, returns discounted forward intrinsic (deterministic limit).
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    # deterministic edge case sigma = 0
    if sigma <= 0.0:
        fwd_disc = S * exp(-q * T) - K * exp(-r * T)
        return max(fwd_disc, 0.0) if option_type == "call" else max(-fwd_disc, 0.0)

    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(S, K, r, q, sigma, T)

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
    Cox–Ross–Rubinstein binomial price (European or American).

    Parameters
    ----------
    S, K : float
        Spot and strike (> 0).
    r, q : float
        Continuous risk-free rate and dividend yield (annual).
    sigma : float
        Annualized volatility (sigma >= 0).
    T : float
        Time to maturity in years (T >= 0).
    steps : int, default 200
        Number of time steps (>= 1).
    option_type : {"call", "put"}, default "call"
        Option type.
    american : bool, default False
        If True, allows early exercise.

    Returns
    -------
    float
        Option price.

    Notes
    -----
    - Uses risk-neutral probability `p = (e^{(r-q)Δt} - d) / (u - d)`.
    - Clamps `p` into [0, 1] for numerical robustness in extreme cases.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")

    if T <= 0.0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    dt = T / steps
    u  = exp(sigma * sqrt(dt))
    d  = 1.0 / u
    disc = exp(-r * dt)
    p = (exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        p = min(1.0, max(0.0, p))  # numeric safeguard

    # terminal payoffs
    values = []
    for j in range(steps + 1):
        S_T = S * (u ** j) * (d ** (steps - j))
        payoff = max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
        values.append(payoff)

    # backward induction
    for n in range(steps - 1, -1, -1):
        new_vals = []
        for j in range(n + 1):
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                S_nj = S * (u ** j) * (d ** (n - j))
                ex = (S_nj - K) if option_type == "call" else (K - S_nj)
                ex = max(ex, 0.0)
                new_vals.append(max(cont, ex))
            else:
                new_vals.append(cont)
        values = new_vals

    return float(values[0])

# --- greeks ----------------------------------------------------------------

def bs_greeks(
    S: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: Literal["call", "put"] = "call",
) -> Dict[str, float]:
    """
    Black–Scholes Greeks for European options (continuous `r`, `q`).

    Returns a dict with: `price`, `delta`, `gamma`, `vega`, `theta`, `rho`, `phi`.

    Units
    -----
    - vega: per 1.00 change in volatility (per 1% ≈ vega/100)
    - theta: per year (per day ≈ theta/365)
    - rho/phi: per 1.00 change in `r`/`q`

    Notes
    -----
    - For `T <= 0` or `sigma <= 0`, Greeks are returned as 0.0 and `price`
      reflects intrinsic/forward-intrinsic.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    price = bs_price(S, K, r, q, sigma, T, option_type)

    if T <= 0.0 or sigma <= 0.0:
        return {
            "price": float(price), "delta": 0.0, "gamma": 0.0, "vega": 0.0,
            "theta": 0.0, "rho": 0.0, "phi": 0.0
        }

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
        delta = disc_q * (Nd1 - 1.0)              # = -disc_q * N(-d1)
        theta = (- S * disc_q * nd1 * sigma / (2.0 * sqrt(T))
                 + r * K * disc_r * (1.0 - Nd2)
                 - q * S * disc_q * (1.0 - Nd1))
        rho   = - K * T * disc_r * (1.0 - Nd2)    # = -K T e^{-rT} N(-d2)
        phi   = + T * S * disc_q * (1.0 - Nd1)    # = +T S e^{-qT} N(-d1)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta),
        "rho":   float(rho),
        "phi":   float(phi),
    }

# --- parity utilities ------------------------------------------------------

def put_from_call(call_price: float, S: float, K: float, r: float, q: float, T: float) -> float:
    """European put price implied by put–call parity (continuous r, q)."""
    call = call_price
    return call - S * exp(-q * T) + K * exp(-r * T)

def call_from_put(put_price: float, S: float, K: float, r: float, q: float, T: float) -> float:
    """European call price implied by put–call parity (continuous r, q)."""
    put = put_price
    return put + S * exp(-q * T) - K * exp(-r * T)

def parity_gap(call: float, put: float, S: float, K: float, r: float, q: float, T: float) -> float:
    """
    Deviation from put–call parity (LHS - RHS).

    Returns
    -------
    float
        `call - put - (S e^{-qT} - K e^{-rT})`. 0 means exact parity.
    """
    lhs = call - put
    rhs = S * exp(-q * T) - K * exp(-r * T)
    return lhs - rhs

def parity_bounds(S: float, K: float, r: float, q: float, T: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    No-arbitrage lower/upper bounds for European call and put (continuous r, q).

    Returns
    -------
    (call_lower, call_upper), (put_lower, put_upper) : tuple[tuple[float, float], tuple[float, float]]
    """
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)

    call_lb = max(0.0, S * disc_q - K * disc_r)
    call_ub = S * disc_q

    put_lb  = max(0.0, K * disc_r - S * disc_q)
    put_ub  = K * disc_r

    return (call_lb, call_ub), (put_lb, put_ub)

