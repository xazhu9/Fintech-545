import numpy as np
import pandas as pd

'''
Function to calculate the covariance matrix for the dataframe that does not have the entire data.
When skipRow is true, use all the rows which have values. When it's false, use pairwise.
func can be np.cov (covariance) and np.corrcoef (correlation)
'''
def missing_cov_corr(df, skipRow=True, func=np.cov):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    m, n = df.shape
    missing_rows = df.isnull().any(axis=1).sum()

    # If there is no missing rows, simply calculate the covariance matrix.
    if not missing_rows:
        cov = func(df.T)
        print(cov)
    else:
        # skipRow is True, apply the method on the rows that have all the data.
        if skipRow:
            # Drop the rows that is not of whole data
            df = df.dropna(axis=0, how='any')
            cov = func(df.T)
        # skipRow is True, apply the pairwise method.
        else:
            out = np.empty((n, n))
            for i in range(n):
                for j in range(i + 1):
                    # Select only rows without missing values in either column i or j
                    valid_rows = df.iloc[:, [i, j]].dropna().index

                    if not valid_rows.empty:
                        cov_ij = func(df.iloc[valid_rows, [i, j]], rowvar=False)[0, 1]
                        out[i, j] = cov_ij
                        out[j, i] = cov_ij
                        cov = out
    return cov

'''
Function that calculates the EW covariance and correlation.
Func take the parameter of 'cov' and 'corr'
'''
def ew_cov_corr(df, lmbd, func='cov'):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        
    if func not in ['cov', 'corr']:
        raise ValueError(f'The func parameter must be "cov" or "corr", got {func} instead.')
    
    # Center the data - to calculate the covariance matrix.
    df -= df.mean(axis=0)
        
    m, n = df.shape
    wts = np.empty(m)
    
    # Setting weights for prior observation
    for i in range(m):
        wts[i] = (1 - lmbd) * lmbd ** (m - i - 1)
        
    # Normalizing the weights
    wts /= np.sum(wts)
    wts = wts.reshape(-1, 1)
    if func == 'cov':   
        res = (wts * df).T @ df
        
    elif func == 'corr':
        res = (wts * df).T @ df
        # Calculate the standard deviations (square root of variances along the diagonal)
        std_devs = np.sqrt(np.diag(res))

        # Convert the covariance matrix to a correlation matrix
        res /= np.outer(std_devs, std_devs)
        
    return res

'''
Near-PSD covariance and correlation.
'''
def near_psd(df, epsilon=0.0):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    invSD = None

    # If the matrix is the covariance matrix, convert it to correlation matrix first.
    if not np.allclose(np.diag(df), 1.0, rtol=1e-03):
        invSD = np.diag(1.0 / np.sqrt(np.diag(df)))
        df = invSD @ df @ invSD
        # # Calculate the standard deviations (square root of variances along the diagonal)
        # std_devs = np.sqrt(np.diag(df))
        # # Convert the covariance matrix to a correlation matrix
        # df /= np.outer(std_devs, std_devs)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(df)
    
    # Update the negative eigenvalues to 0.
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Construct the diagonal scaling matrix
    S = 1 / (eigenvectors * eigenvectors @ eigenvalues)
    S = np.diag(np.sqrt(S))
    #T = np.diag(1.0 / np.sqrt(np.sum(eigenvectors**2 * eigenvalues, axis=0)))
    l = np.diag(np.sqrt(eigenvalues))
    B = S @ eigenvectors @ l
    df = B @ B.T
    
    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        df = invSD @ df @ invSD

    return df

'''
Higham-PSD covariance and correlation.
'''
def higham_psd(df, W=None, epsilon=1e-9, maxIter=100, tol=1e-9):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    m, n = df.shape
    
    # Generate the identity matrix.
    if W is None:
        W = np.eye(m)
        
    deltaS = 0    
    invSD = None
    
    Yk = np.copy(df)

    # If the matrix is the covariance matrix, convert it to correlation matrix first.
    if not np.allclose(np.diag(Yk), 1.0, rtol=1e-03):
        invSD = np.diag(1.0 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
        
    Yo = np.copy(Yk)
    norml = np.finfo(np.float64).max
    i = 1
    
    while i <= maxIter:
        Rk = Yk - deltaS
        
        # Ps update
        Xk = getPS(Rk, W)
        deltaS = Xk - Rk
        # Pu update
        Yk = getPu(Xk)
        # Get Norm
        norm = wgtNorm(Yk-Yo, W)
        #Smallest Eigenvalue
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))
        
        if norm - norml < tol and minEigVal > -epsilon:
            break
        
        norml = norm
        i += 1

    if i < maxIter:
        print("Converged in {} iterations.".format(i))
    else:
        print("Convergence failed after {} iterations".format(i-1))
    
    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
        
    return Yk

'''
Helper functions
'''
def getAplus(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = np.diag(np.maximum(eigenvalues, 0))
    return eigenvectors @ eigenvalues @ eigenvectors.T

def getPS(A, W):
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return (iW @ getAplus(W05 @ A @ W05) @ iW)

def getPu(A):
    A = np.copy(A)  # Work on a copy to avoid modifying the original matrix
    np.fill_diagonal(A, 1)
    return A

def wgtNorm(A, W):
    W05 = np.sqrt(W)
    W05 = W05 @ A @ W05
    return np.sum(W05 * W05)

'''
Cholesky Factorization
'''
def chol_psd(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    m, n = df.shape
    root = np.zeros((m, n))  # Initialize the root matrix within the function
    
    for j in range(m):
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceeding row values.
        if j >= 0:
            s =  np.dot(root[j, :j], root[j, :j])
            
            # Diagonal element
            temp = df.iloc[j, j] - s
            if -1e-8 <= temp <= 0:
                temp = 0.0
            root[j, j] = np.sqrt(max(temp, 0))
            
            if root[j, j] == 0.0:
                root[j, (j+1):n] = 0.0
            else:
                ir = 1.0 / root[j, j]
                for i in range(j+1, m):
                    s = np.dot(root[i, :j], root[j, :j])
                    root[i, j] = (df.iloc[i, j] - s) * ir
                    
    return root