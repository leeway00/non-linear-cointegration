import statsmodels.nonparametric.bandwidths as bw
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np

def normal_pdf(x, scale = 2.5, const = 1):
    additive_const = (1- 0.990199)/2
    return np.exp(-0.5 * (x*scale)**2) / const + additive_const

def gaussian_kernel(x, ker_x):
    h = bw.bw_scott(ker_x)
    kernel_val = normal_pdf((x - ker_x)/h) / h
    return kernel_val

def kernel_regresssion(z, x_train, y_train, z_train):
    """kernel_regresssion Execute kernel regression

    z: I(0) covariate process on which beta depends on. 
        z can be differ from z_train when we want to predict beta on new z
    x_train, y_train: cointegrated time series, following y = beta(z) * x + u
    z_train: I(0) covariate process
    """
    z = np.asarray(z)
    num_prep = x_train * y_train
    den_prep = x_train **2
    def kernel(z):
        w = gaussian_kernel(z, z_train)
        return np.sum(num_prep * w) / np.sum(den_prep * w)
    kernel = np.vectorize(kernel)
    return kernel(z)

def get_max_chi_pval(p, k, m, nboot = int(1e4)):
    chi_cval = np.zeros(nboot)
    chi_cval.sort()
    for i in range(nboot):
        chi_cval[i] = np.random.chisquare(k, m).max()
    
    p1_ind, p5_ind, p10_ind = int(1e4 * 0.01), int(1e4 * 0.05), int(1e4 * 0.1)
    cval = [chi_cval[p1_ind], chi_cval[p5_ind], chi_cval[p10_ind]]
    return (chi_cval > p).mean(), cval


u_range = np.arange(-1, 1, 1e-5)
def stability_test(x_train, y_train, z_train):
    """stability_test Test for stability of Functional-Coefficient Cointegration

    x_train, y_train: cointegrated time series
    z_train: I(0) covariate process
    return: t-stat, p-value, critical values
    """
    beta_const = sm.OLS(y_train, x_train).fit().params[0]
    betas = kernel_regresssion(z_train, x_train, y_train, z_train)
    u = y_train - betas * x_train
    n = len(x_train)
    h = bw.bw_scott(z_train)
    
    sig_u = u.var()
    xt_2 = x_train **2
    deno_prep = np.vectorize(lambda z: 1/n**2 * np.sum(gaussian_kernel(z, z_train) * xt_2))
    deno = deno_prep(z_train)
    nu0 = np.sum(normal_pdf(u_range) ** 2) * 1e-5
    omega = sig_u * nu0/deno
    
    T = n**2 * np.sqrt(h) * (betas - beta_const) / np.sqrt(omega) # Is m * 1 vector where m = len(z_train)
    Tm = abs(T).max()
    pval, cval = get_max_chi_pval(Tm, 1, n)
    return Tm, pval, cval


def coint_test(x_train, y_train, z_train):
    """coint_test Test for cointegration between x and y given z as covariate

    x_train, y_train: cointegrated time series
    z_train: I(0) covariate process
    return: t-stat, p-value, critical values
    """
    
    beta_const = sm.OLS(y_train, x_train).fit().params[0]
    betas = kernel_regresssion(z_train, x_train, y_train, z_train)
    u = y_train - betas * x_train
    n = len(u)
    time = np.arange(n)
    mod = sm.OLS(u**2, sm.add_constant(time)).fit()
    b_hat = mod.params[1]
    u2 = u**2
    u2_mean = u2.mean()
    
    u2_dem = u2 - u2_mean
    M = int(np.sqrt(n))
    def khM_ch(h):
        if h != 0:
            return normal_pdf(h/M) * np.sum((u2.iloc[:-h] - u2_mean).values * (u2.iloc[h:] - u2_mean).values)/n
        else: 
            return normal_pdf(h/M) * np.sum((u2 - u2_mean).values * (u2 - u2_mean).values)/n
    omega_val = np.vectorize(khM_ch)
    M_range = np.arange(-M, M)
    omega2 = np.sum(omega_val(M_range))
    t_sum = time.var() * n
    s_b = np.sqrt(omega2/t_sum)
    t = b_hat/s_b
    pval = 1-stats.norm(0,1).cdf(abs(t))
    cval = stats.norm(0,1).ppf([0.01, 0.05, 0.1])
    return t, pval, cval


