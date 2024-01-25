import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Load the datasets
file_x = 'Week02/Project/problem2_x.csv'
file_x1 = '/Users/xazhu/Fintech-545/Week02/Project/problem2_x1.csv'

data_x = pd.read_csv(file_x)
data_x1 = pd.read_csv(file_x1)

data_x.head(), data_x1.head()

# Calculate mean and covariance of the dataset
mean_x = data_x.mean()
cov_x = data_x.cov()

# Display the mean and covariance
mean_x, cov_x

# Extracting values from the covariance matrix
sigma11 = cov_x.iloc[0, 0]
sigma12 = cov_x.iloc[0, 1]
sigma22 = cov_x.iloc[1, 1]

# Function to calculate conditional mean and variance of X2 given X1
def conditional_params(x1, mean1, mean2, sigma11, sigma12, sigma22):
    """
    Calculate the conditional mean and variance of X2 given X1.
    """
    mean_x2_given_x1 = mean2 + sigma12 * (x1 - mean1) / sigma11
    var_x2_given_x1 = sigma22 - sigma12**2 / sigma11
    return mean_x2_given_x1, var_x2_given_x1

# Calculate conditional mean and variance for each X1
data_x1['mean_x2_given_x1'], data_x1['var_x2_given_x1'] = zip(*data_x1['x1'].apply(
    lambda x: conditional_params(x, mean_x['x1'], mean_x['x2'], sigma11, sigma12, sigma22)
))

# Calculating 95% confidence interval
z_95 = 1.96 # Approximately 1.96 for 95% confidence
data_x1['lower_95'] = data_x1['mean_x2_given_x1'] - z_95 * np.sqrt(data_x1['var_x2_given_x1'])
data_x1['upper_95'] = data_x1['mean_x2_given_x1'] + z_95 * np.sqrt(data_x1['var_x2_given_x1'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data_x1['x1'], data_x1['mean_x2_given_x1'], label='Expected Value of X2 given X1')
plt.fill_between(data_x1['x1'], data_x1['lower_95'], data_x1['upper_95'], color='gray', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Conditional Distribution of X2 given X1')
plt.legend()
plt.grid(True)
plt.show()