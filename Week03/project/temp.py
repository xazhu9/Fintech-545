import numpy as np
from numpy import mean
from concurrent.futures import ThreadPoolExecutor

def colmean(x):
    return np.mean(x, axis=0)

def colmean2(x):
    m = x.shape[1]
    out = np.empty(m, dtype=float)
    for i in range(m):
        out[i] = np.mean(x[:, i])
    return out

def colmean3(x):
    m = x.shape[1]
    out = np.empty(m, dtype=float)
    
    def task(i):
        out[i] = np.mean(x[:, i])

    with ThreadPoolExecutor(max_workers=m) as executor:
        executor.map(task, range(m))
    
    return out

def colmean4(x):
    n, m = x.shape
    out = np.zeros(m, dtype=float)
    for i in range(n):
        for j in range(m):
            out[j] += x[i, j] / n
    return out

def colmean5(x):
    n, m = x.shape
    out = np.empty(m, dtype=float)
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += x[i, j]
        out[j] = s / n
    return out

def demean(x):
    n, m = x.shape
    xm = colmean5(x)
    for i in range(n):
        for j in range(m):
            x[i, j] -= xm[j]
    return x

# 测试示例
x = np.random.randn(10, 2)
print(colmean(x))
print(colmean2(x))
print(colmean3(x))
print(colmean4(x))
print(colmean5(x))
x2 = x.copy()
demean(x2)
print(colmean(x2))
