import numpy as np
from numpy.linalg import inv

from numba import jit
from tqdm import tqdm

def embed(x, dimension=1):
    if len(x.shape) == 2:
        n, m = x.shape
        if dimension < 1 or dimension > n:
            raise ValueError("Wrong embedding dimension")

        y = np.zeros((n - dimension + 1, dimension * m))
        for i in range(m):
            y[:, i::m] = embed(x[:, i], dimension)
        return y

    elif len(x.shape) == 1:
        n = len(x)
        if dimension < 1 or dimension > n:
            raise ValueError("Wrong embedding dimension")

        m = n - dimension + 1
        
        # data = np.array([x[i:i+dimension] for i in range(m)])
        data = x[np.arange(m)[:, None] + np.arange(dimension-1, -1, -1)]
        return data

    else:
        raise ValueError("'x' is not a vector or matrix")

@jit(nopython=True)
def embed_r_2d(x, dimension=1):
    n, m = x.shape

    if dimension < 1 or dimension > n:
        raise ValueError("Wrong embedding dimension")

    l = n - dimension + 1
    y = np.zeros((l, dimension * m))

    for row in range(l):
        for col in range(m):
            for d in range(dimension):
                y[row, d * m + col] = x[row + dimension - d - 1, col]
    
    return y

@jit(nopython=True)
def resample_columns(matrix):
    """Resample each column of the input matrix with replacement."""
    n, m = matrix.shape
    sampled_indices = np.random.choice(n, size=n, replace=True)
    return matrix[sampled_indices]


def loop_function(gam1, gam2, ECT, DeltaX, Y, M):
    # Threshold dummies
    ECTminus = np.where(ECT <= gam1, 1, 0) * ECT
    ECTplus = np.where(ECT > gam2, 1, 0) * ECT
    ThreshECT = np.column_stack((ECTplus, ECTminus))
    Z2 = np.vstack((ThreshECT.T, DeltaX))
    
    alpha2 = (inv(ThreshECT.T @ M @ ThreshECT) @ (ThreshECT.T @ M) @ Y.T).T
    res = Y - (Y @ Z2.T) @ inv(Z2 @ Z2.T) @ Z2
    
    T = len(Y)
    Sigma = (1/T) * (res @ res.T)
    detSigma = np.linalg.det(Sigma)

    # Wald Test
    Wald = alpha2.reshape(-1) @ inv(np.kron(inv(ThreshECT.T @ M @ ThreshECT), Sigma)) @ alpha2.reshape(-1)
    return {'Wald': Wald, 'detSigma': detSigma}


def bootstraploop(vec_beta, k, p, resSig, Bsig, gamma1Sigma, gamma2Sigma, y, T, ng, ninter, ngrid):
    
    Yb = np.zeros((y.shape[0] - 1, k))
    Xminus1 = np.zeros((y.shape[0], k))
    ECTtminus1 = np.zeros((y.shape[0], 1))

    # Boostrap the residuals
    resb = np.vstack([np.zeros((p, k)), resample_columns(resSig)])
    
    for i in range(p + 1, y.shape[0]-1):
        Xminus1[i] = Xminus1[i - 1] + Yb[i - 1]
        ECTtminus1[i] = Xminus1[i] @ vec_beta
        Yb[i] = np.sum(np.column_stack([Bsig[:,2], Bsig[:,3:] @ Yb[i - p:i].reshape(-1), resb[i]]), axis=1)

        if ECTtminus1[i] < gamma1Sigma:
            Yb[i] = np.sum(np.column_stack([Bsig[:,0] * ECTtminus1[i], Yb[i]]), axis=1)
        elif ECTtminus1[i] > gamma2Sigma:
            Yb[i] = np.sum(np.column_stack([Bsig[:,1] * ECTtminus1[i], Yb[i]]), axis=1)
    yboot = np.cumsum(np.vstack([y[0], Yb]), axis=0)

    # Regression on the new series
    ECTboot = (y @ vec_beta)[:-p-1]
    diff_y = np.diff(yboot, axis=0)
    DeltaYboot = diff_y[p:].T
    embedded = embed_r_2d(diff_y, dimension=p+1)
    DeltaXboot = np.vstack([np.ones(embedded.shape[0]), embedded[:,k:].T])
    Mboot = np.eye(T - p - 1) - DeltaXboot.T @ inv(DeltaXboot @ DeltaXboot.T) @ DeltaXboot
    
    gammasb = np.sort(np.unique(ECTboot))
    storeb = np.zeros((ng, ng))
    
    step_size = max(1,len(gammasb) // ngrid)
    for i in range(0, len(gammasb), step_size):
        gam1 = gammasb[i]
        for j in range(i+ninter+1, len(gammasb), step_size):
            gam2 = gammasb[j]
            try:
                result = loop_function(gam1, gam2, ECTboot, DeltaXboot, DeltaYboot, Mboot)
                storeb[i, j] = result['Wald']
            except:
                pass
    
    supWaldboot = storeb.max()
    return supWaldboot


def TVECM_SeoTest(data, lag, beta, nboot, trim=0.1, ngrid = 50):
    # global embedded, k, Y, M, DeltaX, ECT, ECTminussig, ECTplussig, Zsig, Bsig, resSig, Waldboots, PvalBoot, CriticalValBoot, resSig, store_wald, ninter
    y = data
    T, k = y.shape
    p = lag
    
    diff_y = np.diff(y, axis=0)
    Y = DeltaY = diff_y[p:].T
    
    embedded = embed_r_2d(diff_y, p + 1)
    DeltaX = np.vstack([np.ones(embedded.shape[0]), embedded[:,k:].T])

    M = np.eye(T - p - 1) - DeltaX.T @ inv(DeltaX @ DeltaX.T) @ DeltaX
    vec_beta = np.array([1, -beta])

    ECTfull = y @ vec_beta
    ECT = ECTfull[p:-1]

    # Grid calculations
    allgammas = np.sort(np.unique(ECT))
    ng = len(allgammas)
    ninter = round(trim * ng)  # minimal number of obs in each regime
    inf = int(np.ceil(trim * ng))
    sup = int(np.floor((1 - trim) * ng) - 1)
    gammas = allgammas[inf:sup+1]  # Python slicing is exclusive at the end

    store_wald = np.zeros((ng, ng))
    store_sig = np.zeros((ng, ng))

    # Loop for values of the grid
    step_size = max(1, len(gammas) // ngrid)
    for i in range(0,len(gammas),step_size):
        gam1 = gammas[i]
        for j in range(i+ninter+1, len(gammas), step_size):
            gam2 = gammas[j]
            result = loop_function(gam1, gam2, ECT, DeltaX, Y, M)
            try:
                store_wald[i, j] = result['Wald']
                store_sig[i, j] = result['detSigma']
            except Exception:
                pass

    # Sup Wald model
    # Getting the supWald and associated gammas
    supWald = store_wald.max()
    # position = np.unravel_index(np.argmax(store_wald, axis=None), store_wald.shape)
    row, col = np.unravel_index(np.argmax(store_wald, axis=None), store_wald.shape)
    gamma1 = gammas[row]
    gamma2 = gammas[col]

    # Bootstrap test
    # position2 = np.unravel_index(np.argmin(store_sig, axis=None), store_sig.shape)
    r2, c2 = np.unravel_index(np.argmin(store_sig, axis=None), store_sig.shape)
    gamma1Sigma = gammas[r2]
    gamma2Sigma = gammas[c2]

    ECTminussig = np.where(ECT <= gamma1Sigma, 1, 0) * ECT
    ECTplussig = np.where(ECT > gamma2Sigma, 1, 0) * ECT
    Zsig = np.vstack((ECTminussig, ECTplussig, DeltaX))

    Bsig = Y @ Zsig.T @ inv(Zsig @ Zsig.T)
    resSig = (Y - Bsig @ Zsig).T

    Waldboots = []
    for _ in tqdm(range(nboot)):
        res = bootstraploop(vec_beta, k, lag, resSig, Bsig, gamma1Sigma, gamma2Sigma, y, T, ng, ninter, ngrid)
        Waldboots.append(res)

    PvalBoot = np.mean(np.where(np.array(Waldboots) > supWald, 1, 0))
    CriticalValBoot = np.quantile(Waldboots, q=[0.9, 0.95, 0.975, 0.99])
    
    return {'supWald': supWald, 'gamma1': gamma1, 'gamma2': gamma2, 'PvalBoot': PvalBoot, 'CriticalValBoot': CriticalValBoot}

