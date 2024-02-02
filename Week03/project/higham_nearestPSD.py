import numpy as np

def higham_nearestPSD(pc, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    W = np.diag(np.ones(n))

    deltaS = 0
    Yk = pc.copy()
    norml = np.finfo(np.float64).max
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS

        Xk = _getPS(Rk, W)  
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W) 

        norm = np.linalg.norm(Yk - pc, 'fro') 
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and minEigVal > -epsilon:
            break
        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print(f"Convergence failed after {i-1} iterations")

    return Yk

def _getPS(Rk, W):
    raise NotImplementedError("Function _getPS is not implemented.")

def _getPu(Xk, W):
    raise NotImplementedError("Function _getPu is not implemented.")
