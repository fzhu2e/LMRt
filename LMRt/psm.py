import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa import stattools as st
import numpy as np
import pandas as pd
# import fbm
from .utils import (
    clean_ts,
)

def clean_df(df, mask=None):
    pd.options.mode.chained_assignment = None
    if mask is not None:
        df_cleaned = df.loc[mask]
    else:
        df_cleaned = df

    for col in df.columns:
        df_cleaned.dropna(subset=[col], inplace=True)

    return df_cleaned


class Linear:
    def __init__(self, proxy_time, proxy_value, obs_tas_time, obs_tas_value, prior_tas_time=None, prior_tas_value=None):
        self.proxy_time = proxy_time
        self.proxy_value = proxy_value
        self.obs_tas_time = obs_tas_time
        self.obs_tas_value = obs_tas_value
        self.seasons_tas = list(self.obs_tas_value.keys())
        self.prior_tas_time = prior_tas_time
        self.prior_tas_value = prior_tas_value
    
    def calibrate(self, calib_period=None, nobs_lb=25, fit_args=None, metric='fitR2adj'):
        score_list = []
        mdl_list = []
        df_list = []
        sn_list = []
        for sn in self.seasons_tas:
            df_proxy = pd.DataFrame({'time': self.proxy_time, 'proxy': self.proxy_value})
            df_tas = pd.DataFrame({'time': self.obs_tas_time[sn], 'tas': self.obs_tas_value[sn]})
            df = df_proxy.dropna().merge(df_tas.dropna(), how='inner', on='time')
            df.set_index('time', drop=True, inplace=True)
            df.sort_index(inplace=True)
            df.astype(np.float)
            if calib_period is not None:
                mask = (df.index>=calib_period[0]) & (df.index<=calib_period[1])
                df = clean_df(df, mask=mask)

            formula_spell = 'proxy ~ tas'

            nobs = len(df)
            if nobs < nobs_lb:
                print(f'The number of overlapped data points is {nobs} < {nobs_lb}. Skipping ...')
            else:
                fit_args = {} if fit_args is None else fit_args.copy()
                mdl = smf.ols(formula=formula_spell, data=df).fit(**fit_args)
                fitR2adj =  mdl.rsquared_adj,
                mse = np.mean(mdl.resid**2),
                score = {
                    'fitR2adj': fitR2adj,
                    'mse': mse,
                }
                score_list.append(score[metric])
                mdl_list.append(mdl)
                df_list.append(df)
                sn_list.append(sn)

        if len(score_list) > 0:
            opt_idx_dict = {
                'fitR2adj': np.argmax(score_list),
                'mse': np.argmin(score_list),
            }

            opt_idx = opt_idx_dict[metric]
            opt_mdl = mdl_list[opt_idx]
            opt_sn = sn_list[opt_idx]

            calib_details = {
                'df': df_list[opt_idx],
                'nobs': opt_mdl.nobs,
                'fitR2adj': opt_mdl.rsquared_adj,
                'PSMresid': opt_mdl.resid,
                'PSMmse': np.mean(opt_mdl.resid**2),
                'SNR': np.std(opt_mdl.predict()) / np.std(opt_mdl.resid),
                'seasonality': opt_sn,
            }

            self.calib_details = calib_details
            self.model = opt_mdl
        else:
            self.calib_details = None
            self.model = None

    def forward(self):
        sn_tas = self.calib_details['seasonality']
        clean_t, clean_v = clean_ts(self.prior_tas_time[sn_tas], self.prior_tas_value[sn_tas])
        exog_dict = {
            'tas': clean_v,
        }
        self.ye_value = np.array(self.model.predict(exog=exog_dict).values)
        self.ye_time = np.array(clean_t)

class Bilinear:
    def __init__(self, proxy_time, proxy_value, obs_tas_time, obs_tas_value, obs_pr_time, obs_pr_value,
        prior_tas_time=None, prior_tas_value=None, prior_pr_time=None, prior_pr_value=None, ):
        self.proxy_time = proxy_time
        self.proxy_value = proxy_value
        self.obs_tas_time = obs_tas_time
        self.obs_tas_value = obs_tas_value
        self.obs_pr_time = obs_pr_time
        self.obs_pr_value = obs_pr_value
        self.seasons_tas = list(self.obs_tas_value.keys())
        self.seasons_pr = list(self.obs_pr_value.keys())
        self.prior_tas_time = prior_tas_time
        self.prior_tas_value = prior_tas_value
        self.prior_pr_time = prior_pr_time
        self.prior_pr_value = prior_pr_value
    
    def calibrate(self, calib_period=None, nobs_lb=25, fit_args=None, metric='fitR2adj'):
        score_list = []
        mdl_list = []
        df_list = []
        sn_list = []
        for sn_tas in self.seasons_tas:
            if all(np.isnan(self.obs_tas_value[sn_tas])):
                continue
            df_proxy = pd.DataFrame({'time': self.proxy_time, 'proxy': self.proxy_value})
            df_tas = pd.DataFrame({'time': self.obs_tas_time[sn_tas], 'tas': self.obs_tas_value[sn_tas]})
            df = df_proxy.dropna().merge(df_tas.dropna(), how='inner', on='time')
            df_copy = df.copy()
            for sn_pr in self.seasons_pr:
                if all(np.isnan(self.obs_pr_value[sn_pr])):
                    continue
                df_pr = pd.DataFrame({'time': self.obs_pr_time[sn_pr], 'pr': self.obs_pr_value[sn_pr]})
                df = df.merge(df_pr.dropna(), how='inner', on='time')
                df.set_index('time', drop=True, inplace=True)
                df.sort_index(inplace=True)
                df.astype(np.float)
                if calib_period is not None:
                    mask = (df.index>=calib_period[0]) & (df.index<=calib_period[1])
                    df = clean_df(df, mask=mask)

                formula_spell = 'proxy ~ tas + pr'

                nobs = len(df)
                if nobs < nobs_lb:
                    print(f'The number of overlapped data points is {nobs} < {nobs_lb}. Skipping ...')
                else:
                    fit_args = {} if fit_args is None else fit_args.copy()
                    mdl = smf.ols(formula=formula_spell, data=df).fit(**fit_args)
                    score = {
                        'fitR2adj': mdl.rsquared_adj,
                        'mse': np.mean(mdl.resid**2),
                    }
                    score_list.append(score[metric])
                    mdl_list.append(mdl)
                    df_list.append(df)
                    sn_list.append((sn_tas, sn_pr))

                df = df_copy.copy()

        if len(score_list) > 0:
            opt_idx_dict = {
                'fitR2adj': np.argmax(score_list),
                'mse': np.argmin(score_list),
            }

            opt_idx = opt_idx_dict[metric]
            opt_mdl = mdl_list[opt_idx]
            opt_sn = sn_list[opt_idx]

            calib_details = {
                'df': df_list[opt_idx],
                'nobs': opt_mdl.nobs,
                'fitR2adj': opt_mdl.rsquared_adj,
                'PSMresid': opt_mdl.resid,
                'PSMmse': np.mean(opt_mdl.resid**2),
                'SNR': np.std(opt_mdl.predict()) / np.std(opt_mdl.resid),
                'seasonality': opt_sn,
            }

            self.calib_details = calib_details
            self.model = opt_mdl
        else:
            self.calib_details = None
            self.model = None

    def forward(self):
        sn_tas, sn_pr = self.calib_details['seasonality']
        df_tas = pd.DataFrame({'time': self.prior_tas_time[sn_tas], 'tas': self.prior_tas_value[sn_tas]})
        df_pr = pd.DataFrame({'time': self.prior_pr_time[sn_pr], 'pr': self.prior_pr_value[sn_pr]})
        df = df_tas.dropna().merge(df_pr.dropna(), how='inner', on='time')
        df.set_index('time', drop=True, inplace=True)
        df.sort_index(inplace=True)
        df.astype(np.float)

        exog_dict = {
            'tas': df['tas'].values,
            'pr': df['pr'].values,
        }
        self.ye_value = np.array(self.model.predict(exog=exog_dict).values)
        self.ye_time = np.array(df.index.values)


# def gammify(X, shape=1.5, mean=1, jitter=False, seed=0):
#     ''' Transform each **row** of data matrix X to a gaussian distribution using the inverse Rosenblatt transform
#     '''
#     X = np.matrix(X)
#     n = np.shape(X)[0]  # number of rows
#     p = np.shape(X)[1]  # number of columns

#     random.seed(seed)
#     if jitter:
#         # add tiny random numbers to aviod ties
#         X += np.random.normal(0, np.std(X)/1e6, n*p).reshape(n, p)

#     Xn = np.matrix(np.zeros(n*p).reshape(n, p))
#     for j in range(n):
#         # sort the data in ascending order and retain permutation indices
#         R = ss.rankdata(X[j, :])
#         # the cumulative distribution function
#         CDF = R/p - 1/(2*p)
#         # apply the inverse Rosenblatt transformation
#         rate = shape/mean
#         Xn[j, :] = ss.gamma.ppf(CDF, shape, scale=1/rate)  # Xn is now gamma distributed

#     return Xn

# def moving_avg(ys, ts, start=0, end=11):
#     ys_out = []
#     ts = np.asarray(list(set(np.floor(ts))))
#     ts_out = ts[:-1]
#     nt = np.size(ts_out)
#     for i in range(nt):
#         ys_out.append(np.mean(ys[12*i+start:12*i+end]))

#     ys_out = np.asarray(ys_out)
#     return ys_out, ts_out


# def simpleVarveModel(signal, H, shape=1.5, mean=1, SNR=.25, seed=0):
#     ''' The python version of the simple varve model (in R lang) by Nick McKay
#         Adapted by Feng Zhu

#     Args:
#         signal (array): the signal matrix, each row is a time series
#         H (float): Hurst index, should be in (0, 1)
#     '''

#     # first, gammify the input
#     gamSig = gammify(signal, shape=shape, mean=mean, seed=seed)

#     # create gammafied autocorrelated fractal brownian motion series
#     #     wa = Spectral.WaveletAnalysis()
#     #     fBm_ts = wa.fBMsim(N=np.size(signal), H=H)
#     fBm_ts = fbm.fgn(np.size(signal), H)
#     gamNoise = gammify(fBm_ts, shape=shape, mean=mean, seed=seed)

#     # combine the signal with the noise, based on the SNR
#     varves = (gamSig*SNR + gamNoise*(1/SNR)) / (SNR + 1/SNR)

#     res = {
#         'signal': signal,
#         'gamSig': gamSig,
#         'fBm_ts': fBm_ts,
#         'gamNoise': gamNoise,
#         'varves': varves,
#         'H': H,
#         'shape': shape,
#         'mean': mean,
#         'SNR': SNR,
#         'seed': seed,
#     }

#     return res

# class Lake_VarveThickness:
#     def __init__(self, proxy_time, proxy_value, obs_tas_time, obs_tas_value, prior_tas_time=None, prior_tas_value=None):
#         self.proxy_time = proxy_time
#         self.proxy_value = proxy_value
#         self.obs_tas_time = obs_tas_time
#         self.obs_tas_value = obs_tas_value
#         self.season_tas = self.obs_tas_value.keys()  # should be only one season
#         self.prior_tas_time = prior_tas_time
#         self.prior_tas_value = prior_tas_value

#     def calibrate(self, calib_period=None):
#         pass

#     def forward(self):
#         pass

# class Coral_d18O:
#     ''' The PSM is based on the forward model published by [Thompson, 2011]:
#        <Thompson, D. M., T. R. Ault, M. N. Evans, J. E. Cole, and J. Emile-Geay (2011),
#        Comparison of observed and simulated tropical climate trends using a forward
#        model of coral \u03b418O, Geophys.Res.Lett., 38, L14706, doi:10.1029/2011GL048224.>
#        Returns a numpy array that is the same size and shape as the input vectors for SST, SSS.
#     '''
#     def __init__(self, proxy_time, proxy_value,
#             obs_sst_time, obs_sst_value,
#             prior_sst_time=None, prior_sst_value=None,
#             obs_sss_time=None, obs_sss_value=None,
#             prior_sss_time=None, prior_sss_value=None,
#             obs_d18O_time=None, obs_d18O_value=None,
#             prior_d18O_time=None, prior_d18O_value=None,
#         ):
#         self.proxy_time = proxy_time
#         self.proxy_value = proxy_value
#         self.obs_sst_time = obs_sst_time
#         self.obs_sst_value = obs_sst_value
#         self.obs_sss_time = obs_sss_time
#         self.obs_sss_value = obs_sss_value
#         self.obs_d18O_time = obs_d18O_time
#         self.obs_d18O_value = obs_d18O_value
#         self.season_tas = self.obs_tas_value.keys()  # should be only one season
#         self.prior_sst_time = prior_sst_time
#         self.prior_sst_value = prior_sst_value
#         self.prior_sss_time = prior_sss_time
#         self.prior_sss_value = prior_sss_value
#         self.prior_d18O_time = prior_d18O_time
#         self.prior_d18O_value = prior_d18O_value

#     def calibrate(self, calib_period=None):
#         pass

#     def forward(self):
#         pass
