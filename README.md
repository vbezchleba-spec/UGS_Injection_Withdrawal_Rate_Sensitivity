# 📈 Quantitative Econometric Analysis of European Natural Gas Storage (AT–SK Node)

An end-to-end Python pipeline analyzing daily operational dynamics, capacity limits, and weather sensitivity for the Austria–Slovakia (AT–SK) regional gas storage aggregation across 309 days of data.

## 📊 Model Specification & Results

$$\text{Net\_Injection}_t = \beta_0 + \beta_1 (\text{Fill\_Level}_t) + \beta_2 (\text{Temp\_Anomaly}_t) + \beta_3 (\text{Price\_Change}_t) + \varepsilon_t$$

* **Overall Fit:** Adjusted $R^2 = 0.787$ ($F\text{-stat} = 67.84$, $p < 0.001$)
* **Diagnostics:** Standard errors adjusted using a **7-lag Newey-West HAC covariance matrix** to resolve high residual autocorrelation ($\text{Durbin-Watson} = 0.383$).

### Key Findings
* **Weather Sensitivity ($H_1$ Confirmed):** $\beta_2 = +36.92\text{ GWh/°C}$ ($z = 12.96$, $p < 0.001$) — A $1^\circ\text{C}$ cold anomaly drives a $36.92\text{ GWh/day}$ inventory drawdown.
* **Capacity Constraints ($H_2$ Confirmed):** $\beta_1 = -3.86\text{ GWh/\%}$ ($z = -2.11$, $p = 0.035$) — Daily injection velocity slows as working capacity nears 100%.
* **Price Momentum ($H_3$ Rejected):** $\beta_3 = +0.56$ ($z = 0.12$, $p = 0.907$) — Single-day spot price moves do not drive physical storage dispatches.

## 🛠️ Quick Start

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/european-gas-storage-econometrics.git](https://github.com/your-username/european-gas-storage-econometrics.git)
   cd european-gas-storage-econometrics