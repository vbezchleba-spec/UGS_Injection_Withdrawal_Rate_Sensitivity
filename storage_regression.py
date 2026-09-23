import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. GENERATE MOCK DATA (Simulating GIE AGSI+ & EEX Data)
# ---------------------------------------------------------
# We create 365 days of synthetic gas market data for demonstration.
# Later, you can replace this section with direct API requests.

np.random.seed(42)
dates = pd.date_range(start="2025-01-01", periods=365, freq="D")

# Independent Variables (X)
prompt_spread = np.random.normal(loc=1.5, scale=0.8, size=365)     # Front-Month minus Day-Ahead (€/MWh)
inventory_fill = np.linspace(20, 95, 365) + np.random.normal(0, 2, 365) # % Full Level
temp_anomaly = np.random.normal(loc=0.0, scale=3.5, size=365)       # Weather anomaly in °C

# True underlying relationship with noise (Simulating real market response)
# Net Injection = 100 + 15*(Spread) - 1.2*(Fill Level) - 8.5*(Temp Anomaly) + Error
noise = np.random.normal(0, 15, 365)
net_injection = 100 + (15 * prompt_spread) - (1.2 * inventory_fill) - (8.5 * temp_anomaly) + noise

# Combine into a pandas DataFrame (Gretl Dataset equivalent)
df = pd.DataFrame({
    "Net_Injection": net_injection,
    "Prompt_Spread": prompt_spread,
    "Inventory_Fill": inventory_fill,
    "Temp_Anomaly": temp_anomaly
}, index=dates)

# ---------------------------------------------------------
# 2. DEFINE AND ESTIMATE THE MULTIPLE REGRESSION MODEL
# ---------------------------------------------------------

# Define Dependent (Y) and Independent (X) variables
Y = df["Net_Injection"]
X = df[["Prompt_Spread", "Inventory_Fill", "Temp_Anomaly"]]

# In Gretl, the constant (intercept) is added automatically.
# In statsmodels/Python, we MUST manually add the constant term!
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(Y, X).fit()

# ---------------------------------------------------------
# 3. PRINT THE RESULTS SUMMARY
# ---------------------------------------------------------
print(model.summary())