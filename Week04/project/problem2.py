import pandas as pd
import numpy as np
from scipy.stats import norm, t
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller

# 加载数据
file_path = '/Users/xazhu/Fintech-545/Week04/DailyPrices.csv'
prices_df = pd.read_csv(file_path).set_index('Date')

# 定义计算回报率的函数
def calculate_returns(df, method="DISCRETE"):
    if method.upper() == "DISCRETE":
        returns = df.pct_change().dropna()
    elif method.upper() == "LOG":
        returns = np.log(df / df.shift(1)).dropna()
    else:
        raise ValueError("Method must be either 'DISCRETE' or 'LOG'")
    return returns

# 计算所有价格的回报率
returns_df = calculate_returns(prices_df, method="DISCRETE")

# 对META股票回报率进行平均值调整
meta_returns_adjusted = returns_df['META'] - returns_df['META'].mean()

# 定义计算VaR的函数
def calculate_var(returns, alpha=0.05):
    # Normal Distribution VaR
    var_normal = norm.ppf(alpha) * returns.std()
    
    # EWMA VaR
    ewma_variance = returns.ewm(alpha=0.06).var()  # λ=0.94 implies alpha=0.06 for the EWMA calculation
    var_ewma = norm.ppf(alpha) * np.sqrt(ewma_variance.iloc[-1])
    
    # T Distribution VaR
    params = t.fit(returns)
    var_t = t.ppf(alpha, *params)
    
    # AR(1) Model VaR
    if adfuller(returns.dropna())[1] < 0.05:  # Check for stationarity
        ar_model = AutoReg(returns.dropna(), lags=1).fit()
        forecast = ar_model.predict(start=len(returns), end=len(returns), dynamic=False)
        forecast_error = forecast.iloc[0] - returns.iloc[-1]
        error_std = np.std(returns - ar_model.fittedvalues)
        var_ar1 = norm.ppf(alpha) * error_std
    else:
        var_ar1 = np.nan  # Non-stationary, AR(1) model not suitable
    
    # Historical Simulation VaR
    var_historic = np.percentile(returns, alpha*100)
    
    return var_normal, var_ewma, var_t, var_ar1, var_historic

# 计算VaR
var_results = calculate_var(meta_returns_adjusted, alpha=0.05)

print(f"VaR (Normal Distribution): {var_results[0]}")
print(f"VaR (EWMA): {var_results[1]}")
print(f"VaR (T Distribution): {var_results[2]}")
print(f"VaR (AR(1) Model): {var_results[3]}")
print(f"VaR (Historic Simulation): {var_results[4]}")
