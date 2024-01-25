import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize
from scipy.stats import t
file_path = "/Users/xazhu/Fintech-545/Week02/Project/problem2.csv"
data = pd.read_csv(file_path)

X = data.iloc[:,0]
Y = data.iloc[:,1]

# Add a constant to the predictor variable set to represent the intercept
X_with_const = sm.add_constant(X)

#fit OLS
ols_model = sm.OLS(Y, X_with_const)
ols_results = ols_model.fit()
#show result
ols_summary = ols_results.summary()
ols_params = ols_results.params
ols_std_err = ols_results.bse
ols_residual_std = ols_results.resid.std(ddof=X_with_const.shape[1]) 

#print out
print(ols_summary)
print("OLS Estimated coefficients:\n", ols_results.params)
print("Standard deviation of the OLS error:\n", ols_residual_std)

#MLE
model = sm.GLM(Y, X_with_const, family=sm.families.Gaussian())
mle_results = model.fit()
aic_value = mle_results.aic
mle_params = mle_results.params
mle_residual_std = np.sqrt(mle_results.scale)
#print MLE summary
print(mle_results.summary())
print("MLE Estimated coefficients:\n", mle_params)
print("Standard deviation of the MLE error:\n", mle_residual_std)
print("MLE AIC(error normal distribution):\n", aic_value)

#MLE when error under t distribution
def neg_log_likelihood(params):
    n = len(Y)
    X_betas = X_with_const.dot(params[:-2])
    sigma = params[-2]
    nu = params[-1]
    likelihoods = t.logpdf(Y, nu, loc=X_betas, scale=sigma)
    return -np.sum(likelihoods)
initial_guess = np.append(np.zeros(X_with_const.shape[1]), [1, 1])
result = optimize.minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B')
neg_log_likelihood_value = result.fun
max_log_likelihood_value = -neg_log_likelihood_value
k = len(result.x) 
AIC = 2 * k - 2 * max_log_likelihood_value
print("MLE(t-distribution)Estimated AIC:\n", AIC)
print("MLE(t-distribution)Estimated coefficients:\n", result.x)

