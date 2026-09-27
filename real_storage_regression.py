import os
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Configuration
GIE_API_KEY = os.getenv("GIE_API_KEY")
if not GIE_API_KEY:
    raise RuntimeError("Missing GIE_API_KEY in environment variables.")

GIE_HEADERS = {"x-key": GIE_API_KEY}
TARGET_COUNTRIES = ["AT", "SK"]
ANALYSIS_START_DATE = "2025-11-01"


def fetch_gie_storage_data(countries: list, start_date: str) -> pd.DataFrame:
    """Fetch and aggregate daily gas storage injection/withdrawal data from GIE AGSI+ API."""
    records = []

    for country in countries:
        for page in range(1, 3):
            url = f"https://agsi.gie.eu/api?country={country}&size=300&page={page}"
            res = requests.get(url, headers=GIE_HEADERS, timeout=10)
            if res.status_code != 200:
                continue

            payload = res.json()
            data = payload.get("data", []) if isinstance(payload, dict) else payload

            for entry in data:
                date_val = entry.get("gas_day") or entry.get("gasDayStart") or entry.get("gasDay")
                injection = float(entry.get("injection") or 0)
                withdrawal = float(entry.get("withdrawal") or 0)

                records.append({
                    "Date": date_val,
                    "Country": entry.get("code", country),
                    "Net_Injection_GWh": injection - withdrawal,
                    "Fill_Level_Pct": float(entry.get("full") or 0)
                })

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).query("Date >= @start_date")

    # Aggregate AT + SK clusters
    regional_df = (
        df.groupby("Date")
        .agg({"Net_Injection_GWh": "sum", "Fill_Level_Pct": "mean"})
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return regional_df


def fetch_openmeteo_weather(start_str: str, end_str: str) -> pd.DataFrame:
    """Pull historical daily mean temperature for the Baumgarten hub region (48.2082, 16.3738)."""
    params = {
        "latitude": 48.2082,
        "longitude": 16.3738,
        "start_date": start_str,
        "end_date": end_str,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin"
    }
    url = "https://archive-api.open-meteo.com/v1/archive"
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()

    daily = res.json().get("daily", {})
    df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "Temp_Mean": daily["temperature_2m_mean"]
    })
    df["Temp_Anomaly"] = df["Temp_Mean"] - df["Temp_Mean"].mean()
    return df


def fetch_ttf_prices(start_str: str, end_str: str) -> pd.DataFrame:
    """Fetch Dutch TTF Natural Gas futures from Yahoo Finance and forward-fill weekend gaps."""
    ttf = yf.download("TTF=F", start=start_str, end=end_str, progress=False)

    if isinstance(ttf.columns, pd.MultiIndex):
        df = ttf["Close"].reset_index()
    else:
        df = ttf[["Close"]].reset_index()

    df.columns = ["Date", "Price_EUR_MWh"]
    df["Date"] = pd.to_datetime(df["Date"])

    # Reindex to continuous calendar dates
    date_range = pd.date_range(start=start_str, end=end_str)
    df = df.set_index("Date").reindex(date_range).ffill().reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)
    df["Price_Change_EUR"] = df["Price_EUR_MWh"].diff().fillna(0)

    return df


def run_ols_pipeline():
    print("Pulling storage data...")
    df_storage = fetch_gie_storage_data(TARGET_COUNTRIES, ANALYSIS_START_DATE)

    start_date = df_storage["Date"].min().strftime("%Y-%m-%d")
    end_date = df_storage["Date"].max().strftime("%Y-%m-%d")

    print(f"Fetching weather and TTF market data ({start_date} to {end_date})...")
    df_weather = fetch_openmeteo_weather(start_date, end_date)
    df_prices = fetch_ttf_prices(start_date, end_date)

    # Merge inputs
    df = (
        df_storage
        .merge(df_weather[["Date", "Temp_Anomaly"]], on="Date", how="inner")
        .merge(df_prices[["Date", "Price_EUR_MWh", "Price_Change_EUR"]], on="Date", how="inner")
    )

    df.to_csv("final_regression_dataset.csv", index=False)
    print("Dataset exported to 'final_regression_dataset.csv'.")

    # Model estimation
    y = df["Net_Injection_GWh"]
    X = sm.add_constant(df[["Fill_Level_Pct", "Temp_Anomaly", "Price_Change_EUR"]])

    # 7-lag Newey-West HAC correction for serial correlation
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    print("\n" + "=" * 60)
    print(model.summary())


if __name__ == "__main__":
    run_ols_pipeline()