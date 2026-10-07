# Sales Forecasting
# Codec Technologies - Data Analytics Internship

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "monthly_sales.csv",
)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("SALES FORECASTING")
print("=" * 70)

# -----------------------------
# Data cleaning
# -----------------------------
required = {"Date", "Sales"}
missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df = (
    df.dropna(subset=["Date", "Sales"])
    .drop_duplicates()
    .sort_values("Date")
)

# -----------------------------
# Monthly aggregation
# -----------------------------
df["Month"] = df["Date"].dt.to_period("M")

monthly = df.groupby("Month")["Sales"].sum().to_timestamp()

# Regular monthly frequency
monthly = monthly.asfreq("MS")

# Linear interpolation for missing monthly observations
monthly = monthly.interpolate(method="linear")

if len(monthly) < 24:
    print(
        "\nWarning: fewer than 24 monthly observations were supplied. "
        "Forecast stability may be limited."
    )

print("\nData range:", monthly.index.min(), "to", monthly.index.max())
print("Number of months:", len(monthly))

# -----------------------------
# Historical sales
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly.values)
plt.title("Historical Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# -----------------------------
# Rolling mean
# -----------------------------
rolling_mean = monthly.rolling(3).mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly.values, label="Actual")
plt.plot(
    rolling_mean.index,
    rolling_mean.values,
    label="3-Month Rolling Mean",
)
plt.title("Sales Trend and Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Chronological train/test split
# -----------------------------
split = int(len(monthly) * 0.80)

train = monthly.iloc[:split]
test = monthly.iloc[split:]

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))

# -----------------------------
# ARIMA baseline
# -----------------------------
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit()

forecast = fitted.forecast(steps=len(test))
forecast.index = test.index

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

safe_actual = test.replace(0, np.nan)
mape = (
    np.abs((test - forecast) / safe_actual)
    .dropna()
    .mean()
    * 100
)

print("\nEvaluation")
print("-" * 40)
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")

# -----------------------------
# Actual vs forecast
# -----------------------------
plt.figure(figsize=(11, 5))
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index, test.values, label="Actual Test")
plt.plot(forecast.index, forecast.values, label="Forecast")
plt.title("Actual vs Forecast Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Future forecast
# -----------------------------
future_steps = 6

final_model = ARIMA(monthly, order=(1, 1, 1)).fit()
future_forecast = final_model.forecast(steps=future_steps)

future_dates = pd.date_range(
    monthly.index[-1] + pd.offsets.MonthBegin(1),
    periods=future_steps,
    freq="MS",
)

future_forecast.index = future_dates

future_table = pd.DataFrame(
    {
        "Date": future_forecast.index,
        "ForecastSales": future_forecast.values,
    }
)

print("\nNext 6 months forecast:")
print(future_table.to_string(index=False))

future_table.to_csv(
    "future_sales_forecast.csv",
    index=False,
)

print("\nSaved: future_sales_forecast.csv")
