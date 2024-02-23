import pandas as pd
import numpy as np
from scipy.stats import norm

# 假设文件路径
portfolio_path = '/Users/xazhu/Fintech-545/Week04/project/portfolio.csv'
prices_path = '/Users/xazhu/Fintech-545/Week04/DailyPrices.csv'

# 加载数据
portfolio_df = pd.read_csv(portfolio_path)
prices_df = pd.read_csv(prices_path, parse_dates=['Date'], index_col='Date')

# 计算日收益率
daily_returns = prices_df.pct_change().dropna()

def monte_carlo_var(portfolio_name, prices_df, portfolio_df, days=1, iterations=10000, confidence_level=0.95):
    portfolio = portfolio_df[portfolio_df['Portfolio'] == portfolio_name]
    holdings = portfolio.set_index('Stock')['Holding']
    prices = prices_df[holdings.index]
    
    total_investment = np.dot(prices.iloc[-1], holdings)
    weights = holdings * prices.iloc[-1] / total_investment
    
    portfolio_returns = daily_returns[holdings.index].fillna(0).dot(weights)
    mean_return = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    simulated_end_values = np.zeros(iterations)
    for i in range(iterations):
        random_returns = np.random.normal(mean_return, sigma, days)
        simulated_end_values[i] = total_investment * (1 + random_returns[0])
    
    var = np.percentile(simulated_end_values, (1 - confidence_level) * 100)
    return total_investment - var

def calculate_total_var(prices_df, total_holdings, days=1, iterations=10000, confidence_level=0.95):
    total_investment = np.dot(prices_df.iloc[-1], total_holdings)
    
    weighted_returns = daily_returns.dot(total_holdings / total_investment)
    mean_return = weighted_returns.mean()
    sigma = weighted_returns.std()
    
    simulated_end_values = np.zeros(iterations)
    for i in range(iterations):
        random_returns = np.random.normal(mean_return, sigma, days)
        simulated_end_values[i] = total_investment * (1 + random_returns[0])
    
    var = np.percentile(simulated_end_values, (1 - confidence_level) * 100)
    return total_investment - var

# 计算每个投资组合的VaR
portfolio_names = portfolio_df['Portfolio'].unique()
vars_monte_carlo = {name: monte_carlo_var(name, prices_df, portfolio_df) for name in portfolio_names}

# 计算总投资组合的VaR
total_holdings = portfolio_df.groupby('Stock')['Holding'].sum().reindex(prices_df.columns, fill_value=0)
total_var_monte_carlo = calculate_total_var(prices_df, total_holdings)

# 打印结果
print("Monte Carlo VaR for each portfolio:")
for name, var in vars_monte_carlo.items():
    print(f"Portfolio {name} VaR: ${var:.2f}")
print(f"Total Portfolio VaR: ${total_var_monte_carlo:.2f}")
