import jax
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from functools import partial

import scipy.io
from collections import namedtuple
import statsmodels.api as sm # Load data from .mat files
import matplotlib.pyplot as plt
def load_data():
    debt_data = scipy.io.loadmat('../CleanData/debt.mat')
    macro_data = scipy.io.loadmat('../CleanData/macro.mat')
    convyield_data = scipy.io.loadmat('../CleanData/convyield.mat')
    # Create date array
    date = np.arange(1971.5, 2020.75 + 0.25, 0.25)
    # Quarterly interest rates
    m = debt_data['m'] / 4
    # Extracting yield data
    ynom1q = m[:, 0]
    ynom1y = m[:, 1]
    ynom2y = m[:, 2]
    ynom3y = m[:, 3]
    ynom4y = m[:, 4]
    ynom5y = m[:, 5]
    ynom7y = m[:, 7]
    ynom10y = m[:, 10]
    ynom15y = m[:, 15]
    ynom20y = m[:, 20]
    ynom30y = m[:, 30]

    # Combine yield data
    yielddata = np.column_stack([ynom1q, ynom1y, ynom2y, ynom3y, ynom4y, ynom5y, ynom7y, ynom10y])
    yieldmaturity = [1, 4, 8, 12, 16, 20, 28, 40]

    # Macro data
    m = macro_data['m']
    infl = m[:, 0]  # growth in GDP price deflator
    x = m[:, 1]     # real gdp growth

    # Convyield data
    m = convyield_data['m'] / 4
    cy_diff = m[:, 0] - ynom1q  # conv yield from 3m CD minus 3m Tbill
    #moment function

    data = namedtuple('data',['Psi', 'Sig', 'I_pi', 'I_gdp', 'I_y1', 'I_yspr', 'I_cy',
                           'inflpos', 'gdppos', 'y1pos', 'ysprpos', 'pi0', 'x0', 'ynom1q0', 
                           'yspr0', 'cy0', 'X2', 'yielddata', 'yieldmaturity', 'eps2'])
    return data, date, yielddata, yieldmaturity, infl, x, cy_diff, ynom15y, ynom20y, ynom30y

def setup_model(tmpyields, yieldmaturity, cy, numsearch, param_start, infl, x, optionssimplex, optionsderiv):
    # Assembly
    inflpos = 1
    gdppos = 2
    y1pos = 3
    ysprpos = 4
    cypos = 5

    ynom1q = tmpyields[:, 0] + cy
    yspr = tmpyields[:, 5] - ynom1q  # spread between 5-yr yield and 3-month
    pi0 = jnp.mean(infl)
    x0 = jnp.mean(x)
    ynom1q0 = jnp.mean(ynom1q)
    yspr0 = jnp.mean(yspr)
    cy0 = jnp.mean(cy)

    X2 = jnp.column_stack([infl, x, ynom1q, yspr, cy])
    X2 = X2 - jnp.mean(X2, axis=0)

    T, N = X2.shape
    I = jnp.eye(N)
    I_pi = I[:, inflpos - 1]
    I_gdp = I[:, gdppos - 1]
    I_y1 = I[:, y1pos - 1]
    I_yspr = I[:, ysprpos - 1]
    I_cy = I[:, cypos - 1]

    Y = X2[1:, :]
    X = X2[:-1, :]

    # Estimate Psi and Sig
    Psi = jnp.zeros((N, N))
    Sig = jnp.zeros((N, N))
    eps = jnp.zeros(Y.shape)

    # OLS regression for each variable
    #for i in [inflpos, gdppos, y1pos, ysprpos]:
    #    model = sm.OLS(Y[:, i - 1], sm.add_constant(X[:, :4]))
    #    results = model.fit()
    #    Psi[i - 1, :4] = results.params[1:]
    #    eps[:, i - 1] = results.resid
    def jax_ols(Y, X):
        X_with_const = jnp.column_stack([jnp.ones(X.shape[0]), X])
        XTX = jnp.dot(X_with_const.T, X_with_const)
        XTY = jnp.dot(X_with_const.T, Y)
        beta= jnp.linalg.solve(XTX, XTY)
        resid= Y - jnp.dot(X_with_const, beta)
        return beta, resid
    for i in [inflpos, gdppos, y1pos, ysprpos]:
        beta, resid = jax_ols(Y[:, i - 1], X[:, :4])
        Psi = Psi.at[i - 1, :4].set(beta[1:])  # update Psi
        eps = eps.at[:, i - 1].set(resid)  # update eps
        
    if jnp.std(Y[:, cypos - 1]) == 0:
        Y = Y.at[:, cypos - 1].set(0)
        Psi = Psi.at[cypos - 1, :].set(0)
        eps = eps.at[:, cypos - 1].set(0)
        Sigma = jnp.cov(eps[:, :4], rowvar=False)
        Sig = Sig.at[:4, :4].set(jnp.linalg.cholesky(Sigma))
    else:
        # manually conduct OLS using Jax
        beta, resid = jax_ols(Y[:, cypos - 1], X)
        Psi = Psi.at[cypos - 1, :].set(beta[1:])
        eps = eps.at[:, cypos - 1].set(resid)
        Sigma = jnp.cov(eps, rowvar=False)
        Sig = Sig.at[:].set(jnp.linalg.cholesky(Sigma))
    eps2 = eps

    return N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_cy, inflpos, gdppos, y1pos, ysprpos, pi0, x0, ynom1q0, yspr0, cy0, X2, tmpyields, yieldmaturity, eps2


@partial(jax.jit, static_argnums=(1,2))
def mmt(x, N,T,data):
    striphorizon = 100

    L0 = jnp.zeros((N, 1))
    L1 = jnp.zeros((N, N))

    L0 = L0.at[:4, 0].set(x[:4])

    tmp = jnp.zeros((4, 4))
    tmp = tmp.at[:].set(x[4:4 + 4**2].reshape((4, 4)))
    tmp = tmp.T
    L1 = L1.at[:4, :4].set(tmp / jnp.std(data.X2[:, 0:4], axis=0, ddof=1))
    FF = []
    
    Lt = L0.reshape(-1, 1) + L1 @ data.X2.T
    
    mean_sr = jnp.mean(jnp.sqrt(jnp.diag(Lt.T @ Lt)))

    # use sigmoid function to penalize the mean_sr > 0.35
    soft_penalty_param = 1e4
   
    iota = 0.001
    FF.append(soft_penalty_param * jax.nn.sigmoid((mean_sr - 0.36) / iota))

    Bpibar = jnp.linalg.inv(jnp.eye(N) - (data.Psi - data.Sig @ L1)).T @ -data.I_y1
    dApibar = -data.ynom1q0 + 0.5 * Bpibar @ (data.Sig @ data.Sig.T) @ Bpibar - Bpibar @ (data.Sig @ L0)

    iota = 0.0001
    FF.append(soft_penalty_param * jax.nn.sigmoid((0.005 - 4 * (-dApibar[0])) / iota))
    
    
    Api = jnp.zeros(striphorizon + 1)
    Bpi = jnp.zeros((N, striphorizon + 1))

    Api = Api.at[0].set(-data.ynom1q0 + data.cy0)
    Bpi = Bpi.at[:, 0].set(-data.I_y1.T + data.I_cy.T)
    
    for j in range(striphorizon):
        new_Api_value = -data.ynom1q0 + Api[j] + 0.5 * Bpi[:, j] @ (data.Sig @ data.Sig.T) @ Bpi[:, j] - Bpi[:, j] @ (data.Sig @ L0)
        new_Bpi_value = (Bpi[:, j].T @ data.Psi - data.I_y1.T - Bpi[:, j].T @ (data.Sig @ L1)).T

        new_Api_value = new_Api_value[0]

        Api = Api.at[j + 1].set(new_Api_value)
        Bpi = Bpi.at[:, j + 1].set(new_Bpi_value)

    # penalize long end yield reverting to large negative numbers
    iota = 0.001
    FF.append(soft_penalty_param * jax.nn.sigmoid((Api[-1] - 0) / iota))
    
    # Insist on matching the 5-year yield more closely
    penalty = 1e8
    FF_newa = (100 * (-(Api[19] / 20) - (data.ynom1q0 + data.yspr0)))**2 * penalty
    FF.append(FF_newa) 
    FFtmp = (10 * (-(Bpi[:, 19].T / 20) - (data.I_y1.T + data.I_yspr.T)))**2 * penalty

    FF += list(FFtmp)
    
    # Match Yield Curve
    yieldmaturity_index = jnp.array(data.yieldmaturity) - 1
    yieldmaturity_array = jnp.array(data.yieldmaturity)  

    predicted_yield = jnp.tile(-Api[yieldmaturity_index] / yieldmaturity_array, (T, 1)) - ((Bpi[:, yieldmaturity_index] / jnp.tile(yieldmaturity_array, (N, 1))).T @ data.X2.T).T
    Nom_error = 100 * (predicted_yield - data.yielddata)

    penalty = 1e4
    FF_new = penalty * jnp.mean(Nom_error**2, axis=0)

    FF += list(FF_new)
    
    return jnp.sum(jnp.array(FF))

def plotyield(x, N, T, data):
    striphorizon = 100

    L0 = jnp.zeros((N, 1))
    L1 = jnp.zeros((N, N))

    L0 = L0.at[:4, 0].set(x[:4])

    tmp = jnp.zeros((4, 4))
    tmp = tmp.at[:].set(x[4:4 + 4**2].reshape((4, 4)))
    tmp = tmp.T
    L1 = L1.at[:4, :4].set(tmp / jnp.std(data.X2[:, 0:4], axis=0, ddof=1))
    FF = []

    Lt = L0.reshape(-1, 1) + L1 @ data.X2.T
    
    mean_sr = jnp.mean(jnp.sqrt(jnp.diag(Lt.T @ Lt)))

    Bpibar = jnp.linalg.inv(jnp.eye(N) - (data.Psi - data.Sig @ L1)).T @ -data.I_y1
    
    dApibar = -data.ynom1q0 + 0.5 * Bpibar @ (data.Sig @ data.Sig.T) @ Bpibar - Bpibar @ (data.Sig @ L0)
    Api = jnp.zeros(striphorizon + 1)
    Bpi = jnp.zeros((N, striphorizon + 1))

    Api = Api.at[0].set(-data.ynom1q0 + data.cy0)
    Bpi = Bpi.at[:, 0].set(-data.I_y1.T + data.I_cy.T)
    
    for j in range(striphorizon):
        new_Api_value = -data.ynom1q0 + Api[j] + 0.5 * Bpi[:, j] @ (data.Sig @ data.Sig.T) @ Bpi[:, j] - Bpi[:, j] @ (data.Sig @ L0)
        new_Bpi_value = (Bpi[:, j].T @ data.Psi - data.I_y1.T - Bpi[:, j].T @ (data.Sig @ L1)).T

        new_Api_value = new_Api_value[0]

        Api = Api.at[j + 1].set(new_Api_value)
        Bpi = Bpi.at[:, j + 1].set(new_Bpi_value)

    # Match Yield Curve
    yieldmaturity_index = jnp.array(data.yieldmaturity) - 1
    yieldmaturity_array = jnp.array(data.yieldmaturity)  
    predicted_yield = jnp.tile(-Api[yieldmaturity_index] / yieldmaturity_array, (T, 1)) - ((Bpi[:, yieldmaturity_index] / jnp.tile(yieldmaturity_array, (N, 1))).T @ data.X2.T).T

    actual_yields = data.yielddata
    maturities_in_quarters = np.array([1, 4, 8, 20, 28, 40])
    # Find indices in the data that match the desired maturities
    maturities_indices = [np.where(data.yieldmaturity == mat)[0][0] for mat in maturities_in_quarters]
    start_year = 1971
    start_quarter = 3
    end_year = 2020
    end_quarter = 4
    dates = np.arange(start_year + start_quarter / 4, end_year + (end_quarter + 1) / 4, 0.25)
    titles = ['1-quarter', '1-year', '2-year', '5-year', '7-year', '10-year']
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for i, ax in enumerate(axs.flat):
        maturity_index = maturities_indices[i]
        model_yield = predicted_yield[:, maturity_index]
        actual_yield = actual_yields[:, maturity_index]

        ax.plot(dates, model_yield, label=f'Model Yield ({titles[i]})')
        ax.plot(dates, actual_yield, label=f'Actual Yield ({titles[i]})', alpha=0.7)
        ax.set_title(f'Yield Comparison on {titles[i]} Bond')
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield')
        ax.legend()
        ax.grid(True)

    
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))  
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}.{int((x - int(x)) * 4) + 1}Q'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()