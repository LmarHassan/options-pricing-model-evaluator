# options-pricing-model-evaluator
A Python-based quantitative finance project for pricing and evaluating financial options using multiple mathematical models and historical market data.

# Features

* Implements Black-Scholes, Black-76, CRR Binomial, Monte Carlo, Bachelier, Binary and Asian option pricing models.
* Retrieves historical stock-price data using `yfinance` and calculates historical volatility.
* Compares model performance using RMSE and combines selected models to produce a weighted option-price estimate.
* Includes a Flask-based interface for running evaluations and viewing previous results.

# Technologies

Python, Flask, NumPy, pandas and yfinance.

# Running the Project

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Flask application in your browser.

