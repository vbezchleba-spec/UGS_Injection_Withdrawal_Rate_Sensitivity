import requests
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# ===================================================================
# 1. CONFIGURATION & BOUNDARIES
# ===================================================================
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv("GIE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found! Check if you have the line GIE_API_KEY=your_key in your .env file")

HEADERS = {"x-key": API_KEY}
COUNTRIES = ["AT", "SK"]
START_DATE_FILTER = "2025-11-01"

print("--- STEP 1: Fetching GIE AGSI+ Storage Data ---")
storage_data = []

# Paginate to fetch historical daily records past Nov 1, 2025
for country in COUNTRIES:
    for page in range(1, 3):
        url = f"https://agsi.gie.eu/api?country={country}&size=300&page={page}"
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            raw_json = response.json()
            data_list = raw_json.get("data", []) if isinstance(raw_json, dict) else raw_json
            for entry in data_list:
                raw_date = entry.get("gas_day") or entry.get("gasDayStart") or entry.get("gasDay")
                storage_data.append({
                    "Date": raw_date,
                    "Country": entry.get("code") or country,
                    "Net_Injection_GWh": float(entry.get("injection", 0) or 0) - float(entry.get("withdrawal", 0) or 0),
                    "Fill_Level_Pct": float(entry.get("full", 0) or 0)
                })

df_storage = pd.DataFrame(storage_data)
df_storage["Date"] = pd.to_datetime(df_storage["Date"], errors="coerce")
df_storage = df_storage.dropna(subset=["Date"])
df_storage = df_storage[df_storage["Date"] >= pd.to_datetime(START_DATE_FILTER)]

# Aggregate regional nodes (AT + SK) by date
df_regional = df_storage.groupby("Date").agg({
    "Net_Injection_GWh": "sum",
    "Fill_Level_Pct": "mean"
}).reset_index().sort_values("Date").reset_index(drop=True)

# ===================================================================
# 2. FETCH HISTORICAL WEATHER DATA (Open-Meteo)
# ===================================================================
print("--- STEP 2: Fetching Weather Data ---")
start_str = df_regional["Date"].min().strftime("%Y-%m-%d")
end_str = df_regional["Date"].max().strftime("%Y-%m-%d")

weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": 48.2082,  # Baumgarten / Vienna regional hub
    "longitude": 16.3738,
    "start_date": start_str,
    "end_date": end_str,
    "daily": "temperature_2m_mean",
    "timezone": "Europe/Berlin"
}
w_response = requests.get(weather_url, params=weather_params)
w_data = w_response.json()

df_weather = pd.DataFrame({
    "Date": pd.to_datetime(w_data["daily"]["time"]),
    "Temp_Mean": w_data["daily"]["temperature_2m_mean"]
})
df_weather["Temp_Anomaly"] = df_weather["Temp_Mean"] - df_weather["Temp_Mean"].mean()

# ===================================================================
# 3. FETCH FREE MARKET PRICES (Yahoo Finance TTF)
# ===================================================================
print("--- STEP 3: Fetching TTF Gas Prices ---")
ttf_data = yf.download("TTF=F", start=start_str, end=end_str)

if isinstance(ttf_data.columns, pd.MultiIndex):
    df_prices = ttf_data['Close'].reset_index()
else:
    df_prices = ttf_data[['Close']].reset_index()

df_prices.columns = ["Date", "Price_EUR_MWh"]
df_prices["Date"] = pd.to_datetime(df_prices["Date"])

# Forward-fill weekend market gaps to align with continuous 365-day storage data
full_dates = pd.date_range(start=start_str, end=end_str)
df_prices = df_prices.set_index("Date").reindex(full_dates).ffill().reset_index()
df_prices.rename(columns={"index": "Date"}, inplace=True)
df_prices["Price_Change_EUR"] = df_prices["Price_EUR_MWh"].diff().fillna(0)

# ===================================================================
# 4. MERGE DATASETS & SAVE CSV
# ===================================================================
print("--- STEP 4: Merging Datasets ---")
df_merged = pd.merge(df_regional, df_weather[["Date", "Temp_Anomaly"]], on="Date", how="inner")
df_final = pd.merge(df_merged, df_prices[["Date", "Price_EUR_MWh", "Price_Change_EUR"]], on="Date", how="inner")

output_csv = "final_regression_dataset.csv"
df_final.to_csv(output_csv, index=False)
print(f"Dataset compiled successfully into '{output_csv}'!")

# ===================================================================
# 5. ESTIMATE OLS REGRESSION (statsmodels with HAC Correction)
# ===================================================================
print("\n==================================================================")
print("             OLS REGRESSION RESULTS (HAC ADJUSTED)                ")
print("==================================================================")

# Dependent Variable (Y): Daily Net Storage Activity (GWh)
Y = df_final["Net_Injection_GWh"]

# Independent Variables (X): Fill %, Temp Anomaly, and Price Momentum
X = df_final[["Fill_Level_Pct", "Temp_Anomaly", "Price_Change_EUR"]]

# Add intercept (constant) term required by statsmodels
X = sm.add_constant(X)

# Fit OLS model using Newey-West HAC standard errors (7-day lag window)
model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 7})

# Print summary table
print(model.summary())