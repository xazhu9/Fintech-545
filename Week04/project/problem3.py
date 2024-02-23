import pandas as pd
import numpy as np
from scipy.stats import norm

portfolio = pd.read_csv('/Users/xazhu/Fintech-545/Week04/project/portfolio.csv')
prices = pd.read_csv('/Users/xazhu/Fintech-545/Week04/DailyPrices.csv')

# Get the list of stocks in each portfolio
portfolio_A = portfolio[portfolio['Portfolio'] == 'A']['Stock'].tolist()
portfolio_B = portfolio[portfolio['Portfolio'] == 'B']['Stock'].tolist()
portfolio_C = portfolio[portfolio['Portfolio'] == 'C']['Stock'].tolist()

# Get the number of holdings for each stock in each portfolio
holdings_A = portfolio[portfolio['Portfolio'] == 'A']['Holding'].tolist()
holdings_B = portfolio[portfolio['Portfolio'] == 'B']['Holding'].tolist()
holdings_C = portfolio[portfolio['Portfolio'] == 'C']['Holding'].tolist()

# Get the daily prices for the stocks in each portfolio
portfolio_A_prices = prices[portfolio_A].values
portfolio_B_prices = prices[portfolio_B].values
portfolio_C_prices = prices[portfolio_C].values

# Calculate the daily returns for each portfolio
portfolio_A_returns = np.diff(np.log(portfolio_A_prices), axis=0)
portfolio_B_returns = np.diff(np.log(portfolio_B_prices), axis=0)
portfolio_C_returns = np.diff(np.log(portfolio_C_prices), axis=0)

# Calculate the covariance matrix for each portfolio using an exponentially weighted covariance with lambda = 0.94
cov_A = np.cov(portfolio_A_returns, rowvar=False, aweights=np.power(0.94, np.arange(len(portfolio_A_returns)-1, -1, -1)))
cov_B = np.cov(portfolio_B_returns, rowvar=False, aweights=np.power(0.94, np.arange(len(portfolio_B_returns)-1, -1, -1)))
cov_C = np.cov(portfolio_C_returns, rowvar=False, aweights=np.power(0.94, np.arange(len(portfolio_C_returns)-1, -1, -1)))

# Calculate the total covariance matrix using an exponentially weighted covariance with lambda = 0.94
total_returns = np.concatenate((portfolio_A_returns, portfolio_B_returns, portfolio_C_returns), axis=1)
total_cov = np.cov(total_returns, rowvar=False, aweights=np.power(0.94, np.arange(len(total_returns)-1, -1, -1)))

# Calculate the portfolio values for each portfolio
portfolio_A_values = portfolio_A_prices[-1,:] * holdings_A
portfolio_B_values = portfolio_B_prices[-1,:] * holdings_B
portfolio_C_values = portfolio_C_prices[-1,:] * holdings_C

# Calculate the total portfolio value
total_portfolio_value = np.sum(portfolio_A_values) + np.sum(portfolio_B_values) + np.sum(portfolio_C_values)

# Calculate the VaR for each portfolio and the total VaR
confidence_level = 0.95
z_score = norm.ppf(confidence_level)
portfolio_A_var = z_score * np.sqrt(np.dot(portfolio_A_values, np.dot(cov_A, portfolio_A_values)))
portfolio_B_var = z_score * np.sqrt(np.dot(portfolio_B_values, np.dot(cov_B, portfolio_B_values)))
portfolio_C_var = z_score * np.sqrt(np.dot(portfolio_C_values, np.dot(cov_C, portfolio_C_values)))
total_var = z_score * np.sqrt(np.dot(np.concatenate((portfolio_A_values, portfolio_B_values, portfolio_C_values)), np.dot(total_cov, np.concatenate((portfolio_A_values, portfolio_B_values, portfolio_C_values)))))

portfolio_A_var, portfolio_B_var, portfolio_C_var, total_var
 # Print the results
print(f"Portfolio A VaR: ${portfolio_A_var:.2f}")
print(f"Portfolio B VaR: ${portfolio_B_var:.2f}")
print(f"Portfolio C VaR: ${portfolio_C_var:.2f}")
print(f"Total VaR: ${total_var:.2f}")

# 使用简单收益率计算日收益率
portfolio_A_returns_simple = (portfolio_A_prices[1:] / portfolio_A_prices[:-1] - 1)
portfolio_B_returns_simple = (portfolio_B_prices[1:] / portfolio_B_prices[:-1] - 1)
portfolio_C_returns_simple = (portfolio_C_prices[1:] / portfolio_C_prices[:-1] - 1)

# 使用简单收益率重新计算每个投资组合的协方差矩阵
cov_A_simple = np.cov(portfolio_A_returns_simple, rowvar=False, aweights=np.power(0.94, np.arange(len(portfolio_A_returns_simple)-1, -1, -1)))
cov_B_simple = np.cov(portfolio_B_returns_simple, rowvar=False, aweights=np.power(0.94, np.arange(len(portfolio_B_returns_simple)-1, -1, -1)))
cov_C_simple = np.cov(portfolio_C_returns_simple, rowvar=False, aweights=np.power(0.94, np.arange(len(portfolio_C_returns_simple)-1, -1, -1)))

# 使用简单收益率重新计算整体协方差矩阵
total_simple_returns = np.concatenate((portfolio_A_returns_simple, portfolio_B_returns_simple, portfolio_C_returns_simple), axis=1)
total_cov_simple = np.cov(total_simple_returns, rowvar=False, aweights=np.power(0.94, np.arange(len(total_simple_returns)-1, -1, -1)))

# 使用简单收益率重新计算每个投资组合及整体的VaR
portfolio_A_var_simple = z_score * np.sqrt(np.dot(portfolio_A_values, np.dot(cov_A_simple, portfolio_A_values)))
portfolio_B_var_simple = z_score * np.sqrt(np.dot(portfolio_B_values, np.dot(cov_B_simple, portfolio_B_values)))
portfolio_C_var_simple = z_score * np.sqrt(np.dot(portfolio_C_values, np.dot(cov_C_simple, portfolio_C_values)))
total_var_simple = z_score * np.sqrt(np.dot(np.concatenate((portfolio_A_values, portfolio_B_values, portfolio_C_values)), np.dot(total_cov_simple, np.concatenate((portfolio_A_values, portfolio_B_values, portfolio_C_values)))))

# 打印结果
print(f"Portfolio A VaR (Simple Returns): ${portfolio_A_var_simple:.2f}")
print(f"Portfolio B VaR (Simple Returns): ${portfolio_B_var_simple:.2f}")
print(f"Portfolio C VaR (Simple Returns): ${portfolio_C_var_simple:.2f}")
print(f"Total VaR (Simple Returns): ${total_var_simple:.2f}")
