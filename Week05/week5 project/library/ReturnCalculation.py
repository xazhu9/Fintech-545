import numpy as np
import pandas as pd

'''
Calculate Return
'''

# Implement the function to calculate the return
import pandas as pd
import numpy as np

# 重新定义 return_calculate 函数
def return_calculate(prices, method='ARS', dateColumn='Date'):
    # Exclude the date column from the calculations
    tickers = [col for col in prices.columns if col != dateColumn]
    df = prices[tickers] # The dataframe is now with no date column.
    
    # Calculate the return using Classical Brownian Motion.
    if method == 'CBM':
        df = df.diff().dropna()
    
    # Calculate the return using Arithmetic Return System.
    elif method == 'ARS':
        df = (df - df.shift(1)) / df.shift(1)
        df = df.dropna()
        
    # Calculate the return using Geometric Brownian Motion.
    elif method == 'GBM':
        df = np.log(df).diff().dropna()
        
    else:
        raise ValueError(f"method: {method} must be in (\"CBM\",\"ARS\",\"GBM\")")
    
    return df