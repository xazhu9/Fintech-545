import pandas as pd
from scipy.stats import skew, kurtosis
file_path = "/Users/xazhu/Fintech-545/Week02/Project/problem1.csv"
data = pd.read_csv(file_path)

#calculate manually
n = len(data['x'])
mean_manual = sum(data['x']) / n
variance_manual = sum((x-mean_manual)**2 for x in data['x']) / (n-1)
skewness_manual = (sum((x-mean_manual)**3 for x in data['x']) / n) / (variance_manual ** (3/2))
kurtosis_manual = (sum((x - mean_manual) ** 4 for x in data['x']) / n) / (variance_manual ** 2) - 3

#calculate with package
mean_pandas = data['x'].mean()
variance_pandas = data['x'].var(ddof=1)
skewness_scipy = skew(data['x'])
kurtosis_scipy = kurtosis(data['x'])

#print manual result
print("Mean (manually calculated):", mean_manual)
print("Variance (manually calculated):", variance_manual)
print("Skewness (manually calculated):", skewness_manual)
print("Kurtosis (manually calculated):", kurtosis_manual)
print("Mean (package):", mean_pandas)
print("Variance (package):", variance_pandas)
print("Skewness (package):", skewness_scipy)
print("Kurtosis (package):", kurtosis_scipy)