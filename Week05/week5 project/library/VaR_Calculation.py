from scipy.stats import norm, t
import numpy as np
import pandas as pd
import SimulationMethods
'''
Fit the Data with Normal Distribution
'''
def fit_normal(data):
    # Fit the normal distribution to the data
    mu, std = norm.fit(data)
    return mu, std


'''
VaR for Normal Distribution
'''

def var_normal(data, alpha=0.05):
    # Fit the data with normal distribution.
    mu, std = fit_normal(data)
    
    # Calculate the VaR
    VaR = -norm.ppf(alpha, mu, std)
    
    # Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu
    
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [VaR_diff]})

'''
VaR for t Distribution
''' 
def var_t(data, alpha=0.05):
    # Fit the data with t distribution.
    mu, sigma, nu = SimulationMethods.fit_general_t(data)
    
    # Calculate the VaR
    VaR = -t.ppf(alpha, nu, mu, sigma)

    # Calculate the relative difference from the mean expected.
    VaR_diff = VaR + mu
    
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [VaR_diff]})

'''
VaR for t Distribution simulation
''' 
def var_simulation(data, alpha=0.05, size=10000):
    # Fit the data with t distribution.
    mu, sigma, nu = SimulationMethods.fit_general_t(data)
    
    # Generate given size random numbers from a t-distribution
    random_numbers = t.rvs(df=nu, loc=mu, scale=sigma, size=size)
    
    return var_t(random_numbers, alpha)