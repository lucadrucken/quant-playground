# Quant-Notebooks & Tools
*Luca Druckenmueller*

This project contains implementations of classical financial models I learned during my studies.


---

## Installation

```bash
git clone https://github.com/lucadrucken/quant-playground.git
cd quant-playground
pip install -r requirements.txt   
# installs qp as well because of '-e .' → all models available via top-level imports
```

---

## Quickstart

### Performance
```python
from qp import sharpe
import numpy as np

returns = np.array([0.001, -0.0005, 0.0008, 0.0012, -0.0009, 0.0004, 0.0015])

rf_annual = 0.02
rf_daily = rf_annual / 252

sr = sharpe(returns, risk_free=rf_daily, periods_per_year=252)
print(f"Annualized Sharpe = {sr:.2f}")
```

### Risk
```python
from qp import var_historical
import numpy as np

returns = np.array([0.004, -0.006, 0.001, -0.012, 0.003, 0.007, -0.004, 0.002])

var99 = var_historical(returns, level=0.99)
print(f"VaR(99%) per period = {var99:.4f}")
```

### Bond Pricing
```python
from qp import bond_price

price = bond_price(face_value=1000, maturity_years=5,
                   coupon_rate=0.05, ytm=0.04, freq=2)
print(price)  # -> ~1035.74
```

### Black–Scholes (European Options)
```python
from qp import bs_price

call = bs_price(100, 100, 0.02, 0.01, 0.2, 1.0, "call")
put  = bs_price(100, 100, 0.02, 0.01, 0.2, 1.0, "put")
print(call, put)
```

### Binomial Model (European & American)
```python
from qp import binomial_price

eu_put = binomial_price(100, 100, 0.05, 0.0, 0.2, 1.0,
                        steps=500, option_type="put", american=False)
am_put = binomial_price(100, 100, 0.05, 0.0, 0.2, 1.0,
                        steps=500, option_type="put", american=True)
print(eu_put, am_put)  # American Put >= European Put
```

---

## Tests

All models are covered by unit tests using Pytest:

```bash
pytest -q
```

---

## Project Structure

```
QUANT-PLAYGROUND/
├── notebooks/         # Jupyter notebooks (experiments, demos)
│   ├── 00_notebook.ipynb       # general playground
│   └── 01_risk_demo.ipynb      # risk metrics demo
│
├── src/qp/            # main package (all models)
│   ├── __init__.py 
│   ├── bonds.py       # bond pricing models
│   ├── options.py     # option pricing models (Black–Scholes, Binomial)
│   ├── performance.py # performance metrics
│   └── risk.py        # risk measures
│
├── tests/             # pytest unit tests
│   ├── test_bonds.py
│   ├── test_options.py
│   ├── test_risk_sharpe.py
│   └── test_risk_var.py
│
├── requirements.txt   # dependencies
├── pyproject.toml     # build configuration
├── README.md 
└── .gitignore  
```