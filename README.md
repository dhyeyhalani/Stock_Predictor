# Stock Predictor

Predicting next-day moves of **Reliance Industries (RELIANCE.NS)** with technical indicators, a random forest and LSTMs.

I built this to learn time-series ML properly: feature engineering on OHLCV data, avoiding lookahead, and checking models against baselines instead of trusting a nice-looking chart.

## What's in the notebook

1. **Data**: daily OHLCV from Yahoo Finance (2008 to today) plus data quality checks
2. **Features**: ~35 indicators written from scratch in pandas: Ichimoku, ALMA, Vortex, ADX, TSI, Zero-Lag MACD, Stochastic, Bollinger Bands, ATR %, MFI, OBV slope, TTM Squeeze, Fair Value Gaps, returns and volume ratios
3. **Lookahead test**: recomputes the features on truncated data and checks that nothing changes
4. **Feature selection**: correlation filter, then random forest importance (training data only)
5. **LSTM classifier**: will tomorrow close higher? Compared with a majority-class baseline and the random forest
6. **LSTM regression**: predicts the next-day log return, turns it back into a price and compares it with the naive *tomorrow = today* forecast
7. **Keras Tuner (Hyperband)**: hyperparameter search for the regression model

Split is chronological: 70% train / 15% val / 15% test. Scalers are fitted on train only.

## Run it

Easiest is Google Colab: open `Stock_Predictor.ipynb` and run all.

Locally:

```bash
git clone https://github.com/dhyeyhalani/Stock_Predictor.git
cd Stock_Predictor
pip install -r requirements.txt
jupyter notebook Stock_Predictor.ipynb
```

Change `TICKER` at the top to try another stock. Set `RUN_TUNER = True` to re-run the hyperparameter search (~50 min on a Colab GPU).

## v2: what I fixed

The first version reported ~67% accuracy / 0.72 ROC-AUC. That was too good to be true. It came from bugs:

- **Open and Close were swapped.** yfinance returns columns as `Close, High, Low, Open, Volume` and I renamed them by position. The model was effectively predicting today's open-to-close move using today's high and low. That's leakage.
- **Ichimoku chikou span** (`ICS_26`) is the close shifted *back* 26 days, which is future data. It's removed now.
- **Bollinger Bands and ATR were silently skipped** because `pandas_ta` column names changed between versions. All indicators are now computed manually, with no `pandas_ta` dependency.
- **Price-level features** (raw OBV, raw Ichimoku lines, raw BB bands) are replaced with ratios and oscillators, so test data doesn't fall outside the scaler's range.
- **Sequence labels were off by one day**, and val/test lost their first 60 days. Fixed.
- **Regression target scaler was fitted on all data**, including test. It's now fitted on train only, and the target is the return instead of the raw price.
- The tuned model had `Dense(units=2)` and hardcoded hyperparameters. It now uses a single output and the tuner's best values.
- Added baselines, a confusion matrix, loss curves, seeds and a lookahead test.

After the fixes, accuracy is much closer to the baseline. That's the honest result for daily stock direction.

## Next ideas

- Walk-forward validation instead of a single split
- Market context features (NIFTY 50, INR/USD, sector index)
- Multi-day return targets
- A simple backtest with transaction costs

---

*Learning project, not financial advice.*
