import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Load the dataset for problem3
file_problem3 = '/Users/xazhu/Fintech-545/Week02/Project/problem3.csv'
data_problem3 = pd.read_csv(file_problem3)

data_problem3.head()

# Suppress warnings for model convergence issues
warnings.filterwarnings("ignore")

# Function to fit ARIMA model and return AIC
def fit_arima(data, ar_order, ma_order):
    """
    Fit an ARIMA model and return the AIC.
    """
    model = ARIMA(data, order=(ar_order, 0, ma_order))
    model_fit = model.fit()
    return model_fit.aic

# Dictionary to store AIC values for each model
aic_values = {}

# Fit AR(1) to AR(3) and MA(1) to MA(3)
for ar_order in range(1, 4):
    for ma_order in range(1, 4):
        aic_values[f'AR({ar_order})-MA({ma_order})'] = fit_arima(data_problem3['x'], ar_order, ma_order)

aic_values_sorted = sorted(aic_values.items(), key=lambda item: item[1])
aic_values_sorted
for model, aic in aic_values_sorted:
    print(f'{model}: AIC = {aic}')
