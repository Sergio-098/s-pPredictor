import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Rf
from sklearn.metrics import precision_score

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

del sp500["Stock Splits"]
del sp500["Dividends"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1991-01-01":].copy()

horizons = [2, 5, 7, 30, 50, 90]
predictors = ["Open", "High", "Low", "Close", "Volume"]

sp500["daily_return"] = (sp500["Close"] / sp500["Close"].shift(1)) - 1
sp500["prev_volume"] = sp500["Volume"].shift(1)
sp500["prev_high"] = sp500["High"].shift(1)
sp500["prev_low"] = sp500["Low"].shift(1)
sp500["prev_open"] = sp500["Open"].shift(1)
sp500["prev_close"] = sp500["Close"].shift(1)
sp500['Volatility'] = sp500['daily_return'].rolling(window=10).std()

predictors.extend(["daily_return", "prev_volume", "prev_high", "prev_low",
                   "prev_open", "prev_close", 'Volatility'])


for h in horizons:
    roll_avs = sp500.rolling(h).mean()

    ratio_column = f"Close_Ratio_{h}"
    sp500[ratio_column] = sp500["Close"] / roll_avs["Close"]

    trend_column = f"Trend_{h}"
    sp500[trend_column] = sp500.shift(1).rolling(h).sum()["Target"]
    predictors.extend([ratio_column, trend_column])

for h in horizons:
    ratio = f"Close_Ratio_{h}"
    sp500[f"is_growing{h}"] = ((sp500[ratio] > sp500[ratio].shift(1))
                               & (sp500[ratio].shift(1) >
                                  sp500[ratio].shift(2))).astype(int)
    predictors.append(f"is_growing{h}")

sp500 = sp500.dropna()


model = Rf(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    comb = pd.concat([test["Target"], preds], axis=1)
    return comb


def backtest(data, model, predictors, start=2500, step=250):
    all_preds = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[0:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_preds.append(predictions)
    return pd.concat(all_preds)


predictions = backtest(sp500, model, predictors)
print(predictions["Predictions"].value_counts())
precise = precision_score(predictions["Target"], predictions["Predictions"])
print("precision: " + str(precise))
print(sp500[["Open", "Close"]].tail(7))
print(predictions["Predictions"].tail(7))
