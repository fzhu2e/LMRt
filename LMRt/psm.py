import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa import stattools as st
import scipy.stats as ss
import numpy as np
import pandas as pd
import random
import fbm
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import integrate, signal
# import fbm
from .utils import (
    clean_ts,
    seasonal_var,
    annualize_var,
)
from tqdm import tqdm

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
    ''' The univariate linear PSM
    '''
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
    ''' The bivariate linear PSM
    '''
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


class Lake_VarveThickness():
    ''' The varve thickness model

    It takes summer temperature as input (JJA for NH and DJF for SH).
    '''
    def __init__(self, proxy_time, proxy_value, obs_tas_time, obs_tas_value, prior_tas_time=None, prior_tas_value=None):
        self.proxy_time = proxy_time
        self.proxy_value = proxy_value
        self.obs_tas_time = obs_tas_time
        self.obs_tas_value = obs_tas_value
        self.prior_tas_time = prior_tas_time
        self.prior_tas_value = prior_tas_value

        # params for the model
        self.H = None
        self.shape = None
        self.mean = None
        self.SNR = None
        self.seed = None

    def forward(self):

        def simpleVarveModel(signal, H=0.75, shape=1.5, mean=1, SNR=.25, seed=0):
            ''' The python version of the simple varve model (in R lang) by Nick McKay
                Adapted by Feng Zhu

            Args:
                signal (array): the signal matrix, each row is a time series
                H (float): Hurst index, should be in (0, 1)
            '''
            # first, gammify the input
            gamSig = gammify(signal, shape=shape, mean=mean, seed=seed)

            # create gammafied autocorrelated fractal brownian motion series
            #     wa = Spectral.WaveletAnalysis()
            #     fBm_ts = wa.fBMsim(N=np.size(signal), H=H)
            fBm_ts = fbm.fgn(np.size(signal), H)
            gamNoise = gammify(fBm_ts, shape=shape, mean=mean, seed=seed)

            # combine the signal with the noise, based on the SNR
            varves = (gamSig*SNR + gamNoise*(1/SNR)) / (SNR + 1/SNR)

            res = {
                'signal': signal,
                'gamSig': gamSig,
                'fBm_ts': fBm_ts,
                'gamNoise': gamNoise,
                'varves': varves,
                'H': H,
                'shape': shape,
                'mean': mean,
                'SNR': SNR,
                'seed': seed,
            }

            return res

        def gammify(X, shape=1.5, mean=1, jitter=False, seed=0):
            ''' Transform each **row** of data matrix X to a gaussian distribution using the inverse Rosenblatt transform
            '''
            X = np.matrix(X)
            n = np.shape(X)[0]  # number of rows
            p = np.shape(X)[1]  # number of columns

            random.seed(seed)
            if jitter:
                # add tiny random numbers to aviod ties
                X += np.random.normal(0, np.std(X)/1e6, n*p).reshape(n, p)

            Xn = np.matrix(np.zeros(n*p).reshape(n, p))
            for j in range(n):
                # sort the data in ascending order and retain permutation indices
                R = ss.rankdata(X[j, :])
                # the cumulative distribution function
                CDF = R/p - 1/(2*p)
                # apply the inverse Rosenblatt transformation
                rate = shape/mean
                Xn[j, :] = ss.gamma.ppf(CDF, shape, scale=1/rate)  # Xn is now gamma distributed

            return Xn
        ###########################

        # update the params
        kwargs = {
            'H': 0.75,
            'shape': 1.5,
            'mean': 1,
            'SNR': 0.25,
            'seed': 0,
        }

        input_args = {
            'H': self.H,
            'shape': self.shape,
            'mean': self.mean,
            'SNR': self.SNR,
            'seed': self.seed,
        }

        for k, v in kwargs.items():
            if input_args[k] is not None:
                kwargs[k] = input_args[k]

        # run the model
        sn = list(self.prior_tas_value.keys())[0]
        varve_res = simpleVarveModel(self.prior_tas_value[sn], **kwargs)
        self.ye_time = self.prior_tas_time[sn]
        self.ye_value = np.array(varve_res['varves'])[0]


class Ice_d18O():
    ''' The ice d18O model

    It takes montly tas, pr, and psl as input.
    '''
    def __init__(self, proxy_time, proxy_value,
        prior_tas_time=None, prior_tas_value=None, prior_pr_time=None, prior_pr_value=None,
        prior_d18Opr_time=None, prior_d18Opr_value=None, prior_psl_time=None, prior_psl_value=None):
        self.proxy_time = proxy_time
        self.proxy_value = proxy_value
        self.prior_tas_time = prior_tas_time
        self.prior_tas_value = prior_tas_value
        self.prior_pr_time = prior_pr_time
        self.prior_pr_value = prior_pr_value
        self.prior_d18Opr_time = prior_d18Opr_time
        self.prior_d18Opr_value =prior_d18Opr_value
        self.prior_psl_time = prior_psl_time
        self.prior_psl_value = prior_psl_value

    def forward(self, nproc=8):
        def ice_sensor(year, d18Op, pr, alt_diff=0.):
            ''' Icecore sensor model

            The ice core sensor model calculates precipitation-weighted del18OP (i.e. isotope ratio is weighted by
            the amount of precipitation that accumulates) and corrects for temperature and altitude bias between model
            and site ([Yurtsever, 1975], 0.3/100m [Vogel et al., 1975]).

            Args:
                year (1d array: time): time axis [year in float]
                d18Op (2d array: location, time): d18O of precipitation [permil]
                pr (2d array: location, time): precipitation rate [kg m-2 s-1]
                alt_diff 12d array: location): actual Altitude-Model Altitude [meters]

            Returns:
                d18Oice (2d array: location, year in int): annualizd d18O of ice [permil]

            References:
                Yurtsever, Y., Worldwide survey of stable isotopes in precipitation., Rep. Isotope Hydrology Section, IAEA, 1975.

            '''
            # Altitude Effect: cooling and precipitation of heavy isotopes.
            # O18 ~0.15 to 0.30 permil per 100m.

            alt_eff = -0.25
            alt_corr = (alt_diff/100.)*alt_eff

            year_ann, d18Op_weighted = annualize_var(year, d18Op, weights=pr)
            d18O_ice = d18Op_weighted + alt_corr

            return d18O_ice


        def diffusivity(rho, T=250, P=0.9, rho_d=822, b=1.3):
            '''
            DOCSTRING: Function 'diffusivity'
            Description: Calculates diffusivity (in m^2/s) as a function of density.

            Inputs:
            P: Ambient Pressure in Atm
            T: Temperature in K
            rho: density profile (kg/m^3)
            rho_d: 822 kg/m^2 [default], density at which ice becomes impermeable to diffusion

            Defaults are available for all but rho, so only one argument need be entered.

            Note values for diffusivity in air:

            D16 = 2.1e-5*(T/273.15)^1.94*1/P
            D18 = D16/1.0285
            D2 = D16/1.0251
            D17 = D16/((D16/D18)^0.518)

            Reference: Johnsen et al. (2000): Diffusion of Stable isotopes in polar firn and ice:
            the isotope effect in firn diffusion

            '''

            # Set Constants

            R = 8.314478                                                # Gas constant
            m = 18.02e-3                                                # molar weight of water (in kg)
            alpha18 = np.exp(11.839/T-28.224e-3)                    # ice-vapor fractionation for oxygen 18
            p = np.exp(9.5504+3.53*np.log(T)-5723.265/T-0.0073*T)     # saturation vapor pressure
            Po = 1.                                                 # reference pressure, atmospheres
            rho_i = 920.  # kg/m^3, density of solid ice

            # Set diffusivity in air (units of m^2/s)

            Da = 2.1e-5*np.power((T/273.15), 1.94)*(Po/P)
            Dai = Da/1.0285

            # Calculate Tortuosity

            invtau = np.zeros(len(rho))
            #  for i in range(len(rho)):
            #      if rho[i] <= rho_i/np.sqrt(b):
            #          # invtau[i]=1.-1.3*np.power((rho[i]/rho_d),2)
            #          invtau[i] = 1.-1.3*np.power((rho[i]/rho_i), 2)
            #      else:
            #          invtau[i] = 0.

            selector =rho <= rho_i/np.sqrt(b)
            invtau[selector] = 1.-1.3*(rho[selector]/rho_i)**2

            D = m*p*invtau*Dai*(1/rho-1/rho_d)/(R*T*alpha18)

            return D


        def densification(Tavg, bdot, rhos, z):  # ,model='hljohnsen'):
            ''' Calculates steady state snow/firn depth density profiles using Herron-Langway type models.



            Args:
                Tavg: 10m temperature in celcius ## CELCIUS!  # fzhu: should be in K now
                bdot: accumulation rate in mwe/yr or (kg/m2/yr)
                rhos: surface density in kg/m3
                z: depth in true_metres

                model can be: {'HLJohnsen' 'HerronLangway' 'LiZwally' 'Helsen' 'NabarroHerring'}
                default is herronlangway. (The other models are tuned for non-stationary modelling (Read Arthern et al.2010 before applying in steady state).

            Returns:
                rho: density (kg/m3) for all z-values.
                zieq: ice equivalent depth for all z-values.
                t: age for all z-values (only taking densification into account.)

                Example usage:
                z=0:300
                [rho,zieq,t]=densitymodel(-31.5,177,340,z,'HerronLangway')
                plot(z,rho)

            References:
                Herron-Langway type models. (Arthern et al. 2010 formulation).
                Aslak Grinsted, University of Copenhagen 2010
                Adapted by Sylvia Dee, Brown University, 2017
                Optimized by Feng Zhu, University of Southern California, 2017

            '''
            rhoi = 920.
            rhoc = 550.
            rhow = 1000.
            rhos = 340.
            R = 8.314

            # Tavg=248.
            # bdot=0.1
            # Herron-Langway with Johnsen et al 2000 corrections.
            # Small corrections to HL model which are not in Arthern et al. 2010

            c0 = 0.85*11*(bdot/rhow)*np.exp(-10160./(R*Tavg))
            c1 = 1.15*575*np.sqrt(bdot/rhow)*np.exp(-21400./(R*Tavg))

            k0 = c0/bdot  # ~g4
            k1 = c1/bdot

            # critical depth at which rho=rhoc
            zc = (np.log(rhoc/(rhoi-rhoc))-np.log(rhos/(rhoi-rhos)))/(k0*rhoi)  # g6

            ix = z <= zc  # find the z's above and below zc
            upix = np.where(ix)  # indices above zc
            dnix = np.where(~ix)  # indices below zc

            q = np.zeros((z.shape))  # pre-allocate some space for q, rho
            rho = np.zeros((z.shape))

            # test to ensure that this will not blow up numerically if you have a very very long core.
            # manually set all super deep layers to solid ice (rhoi=920)
            NUM = k1*rhoi*(z-zc)+np.log(rhoc/(rhoi-rhoc))

            numerical = np.where(NUM <= 100.0)
            blowup = np.where(NUM > 100.0)

            q[dnix] = np.exp(k1*rhoi*(z[dnix]-zc)+np.log(rhoc/(rhoi-rhoc)))  # g7
            q[upix] = np.exp(k0*rhoi*z[upix]+np.log(rhos/(rhoi-rhos)))  # g7

            rho[numerical] = q[numerical]*rhoi/(1+q[numerical])  # [g8] modified by fzhu to fix inconsistency of array size
            rho[blowup] = rhoi

            # only calculate this if you want zieq
            tc = (np.log(rhoi-rhos)-np.log(rhoi-rhoc))/c0  # age at rho=rhoc [g17]
            t = np.zeros((z.shape))  # pre allocate a vector for age as a function of z
            t[upix] = (np.log(rhoi-rhos)-np.log(rhoi-rho[upix]))/c0  # [g16] above zc
            t[dnix] = (np.log(rhoi-rhoc)-np.log(rhoi+0.0001-rho[dnix]))/c1 + tc  # [g16] below zc
            tdiff = np.diff(t)

            # make sure time keeps increasing even after we reach the critical depth.
            if np.any(tdiff == 0.00):
                inflection = np.where(tdiff == 0.0)
                lineardepth_change = t[inflection][0]

                for i in range(len(t)):
                    if t[i] > lineardepth_change:
                        t[i] = t[i-1] + 1e-5

            zieq = t*bdot/rhoi  # [g15]

            return rho, zieq, t


        def ice_archive(d18Oice, pr_ann, tas_ann, psl_ann, nproc=8):
            ''' Accounts for diffusion and compaction in the firn.

            Args:
                d18Oice (1d array: year in int): annualizd d18O of ice [permil]
                pr_ann (1d array: year in int): precipitation rate [kg m-2 s-1]
                tas_ann (1d array: year in int): annualizd atomspheric temerature [K]
                psl_ann (1d array: year in int): annualizd sea level pressure [Pa]
                nproc (int): the number of processes for multiprocessing

            Returns:
                ice_diffused (1d array: year in int): archived ice d18O [permil]

            '''
            # ======================================================================
            # A.0: Initialization
            # ======================================================================
            # accumulation rate [m/yr]
            # note that the unit of pr_ann is [kg m-2 s-1], so need to divide by density [kg m-3] and convert the time
            yr2sec_factor = 3600*24*365.25
            accum = pr_ann/1000*yr2sec_factor

            # depth horizons (accumulation per year corresponding to depth moving down-core)
            bdown = accum[::-1]
            bmean = np.mean(bdown)
            depth = np.sum(bdown)
            depth_horizons = np.cumsum(bdown)
            dz = np.min(depth_horizons)/10.  # step in depth [m]

            Tmean = np.mean(tas_ann)  # unit in [K]
            Pmean = np.mean(psl_ann)*9.8692e-6  # unit in [Atm]

            # contants
            rho_s = 300.  # kg/m^3, surface density
            rho_d = 822.  # kg/m^2, density at which ice becomes impermeable to diffusion
            rho_i = 920.  # kg/m^3, density of solid ice

            # ======================================================================
            # A.1: Compaction Model
            # ======================================================================
            z = np.arange(0, depth, dz) + dz  # linear depth scale

            # set density profile by calling densification function
            rho, zieq, t = densification(Tmean, bmean, rho_s, z)

            rho = rho[:len(z)]  # cutoff the end
            time_d = np.cumsum(dz/bmean*rho/rho_i)
            ts = time_d*yr2sec_factor  # convert time in years to ts in seconds

            # integrate diffusivity along the density gradient to obtain diffusion length
            D = diffusivity(rho, Tmean, Pmean, rho_d, bmean)

            D = D[:-1]
            rho = rho[:-1]
            diffs = np.diff(z)/np.diff(time_d)
            diffs = diffs[:-1]

            # Integration using the trapezoidal method

            # IMPORTANT: once the ice reaches crtiical density (solid ice), there will no longer
            # be any diffusion. There is also numerical instability at that point. Set Sigma=1E-13 for all
            # points below that threshold.

            # Set to 915 to be safe.
            solidice = np.where(rho >= rho_d-5.0)
            diffusion = np.where(rho < rho_d-5.0)

            dt = np.diff(ts)
            sigma_sqrd_dummy = 2*np.power(rho, 2)*dt*D
            sigma_sqrd = integrate.cumtrapz(sigma_sqrd_dummy)
            diffusion_array = diffusion[0]
            diffusion_array = diffusion_array[diffusion_array < len(sigma_sqrd)]  # fzhu: to avoid the boundary index error
            diffusion = np.array(diffusion_array)

            #  rho=rho[0:-1] # modified by fzhu to fix inconsistency of array size
            #  sigma=np.zeros((len(rho)+1)) # modified by fzhu to fix inconsistency of array size
            sigma = np.zeros((len(rho)))
            sigma[diffusion] = np.sqrt(1/np.power(rho[diffusion],2)*sigma_sqrd[diffusion]) # modified by fzhu to fix inconsistency of array size
            #sigma[solidice]=np.nanmax(sigma) #max diffusion length in base of core // set in a better way. max(sigma)
            sigma[solidice] = sigma[diffusion][-1]
            sigma = sigma[:-1]

            # ======================================================================
            # A.2. Diffusion Profile
            # ======================================================================
            # Load water isotope series
            del18 = np.flipud(d18Oice)  # NOTE YOU MIGHT NOT NEED FLIP UD here. Our data goes forward in time.

            # interpolate over depths to get an array of dz values corresponding to isotope values for convolution/diffusion
            iso_interp = np.interp(z, depth_horizons, del18)

            # Return a warning if the kernel length is approaching 1/2 that of the timeseries.
            # This will result in spurious numerical effects.

            zp = np.arange(-100, 100, dz)
            if (len(zp) >= 0.5*len(z)):
                print("Warning: convolution kernel length (zp) is approaching that of half the length of timeseries. Kernel being clipped.")
                bound = 0.20*len(z)*dz
                zp = np.arange(-bound, bound, dz)

            #  print('start for loop ...')
            #  start_time = time.time()

            rm = np.nanmean(iso_interp)
            cdel = iso_interp-rm

            diffused_final = np.zeros(len(iso_interp))
            if nproc == 1:
                for i in tqdm(range(len(sigma))):
                    sig = sigma[i]
                    part1 = 1./(sig*np.sqrt(2.*np.pi))
                    part2 = np.exp(-zp**2/(2*sig**2))
                    G = part1*part2
                    #  diffused = np.convolve(G, cdel, mode='same')*dz  # fzhu: this is way too slow
                    diffused = signal.fftconvolve(cdel, G, mode='same')*dz  # put cdel in the front to keep the same length as before
                    diffused += rm  # remove mean and then put back
                    diffused_final[i] = diffused[i]

            else:
                #  print('Multiprocessing: nproc = {}'.format(nproc))

                def conv(sig, i):
                    part1 = 1./(sig*np.sqrt(2.*np.pi))
                    part2 = np.exp(-zp**2/(2*sig**2))
                    G = part1*part2
                    diffused = signal.fftconvolve(cdel, G, mode='same')*dz
                    diffused += rm  # remove mean and then put back

                    return diffused[i]

                res = Pool(nproc).map(conv, sigma, range(len(sigma)))
                diffused_final[:len(res)] = np.array(res)

            #  print('for loop: {:0.2f} s'.format(time.time()-start_time))

            # take off the first few and last few points used in convolution
            diffused_timeseries = diffused_final[0:-3]

            # Now we need to pack our data back into single year data units based on the depths and year interpolated data
            final_iso = np.interp(depth_horizons, z[0:-3], diffused_timeseries)
            ice_diffused = final_iso

            return ice_diffused

        ################
        # run the model

        # annualize the data
        year_ann, tas_ann = annualize_var(self.prior_tas_time, self.prior_tas_value)
        year_ann, psl_ann = annualize_var(self.prior_psl_time, self.prior_psl_value)
        year_ann, pr_ann = annualize_var(self.prior_pr_time, self.prior_pr_value)

        # sensor model
        d18O_ice = ice_sensor(self.prior_pr_time, self.prior_d18Opr_value, self.prior_pr_value)

        # diffuse model
        ice_diffused = ice_archive(d18O_ice, pr_ann, tas_ann, psl_ann, nproc=nproc)

        self.ye_time = year_ann
        self.ye_value = ice_diffused[::-1]


class Coral_SrCa:
    ''' The coral Sr/Ca model
    '''
    def __init__(self, proxy_time, proxy_value,
            obs_tos_time, obs_tos_value,
            prior_tos_time=None, prior_tos_value=None,
        ):
        self.proxy_time = proxy_time
        self.proxy_value = proxy_value
        self.obs_tos_time = obs_tos_time
        self.obs_tos_value = obs_tos_value
        self.prior_tos_time = prior_tos_time
        self.prior_tos_value = prior_tos_value

    def forward(self, b=10.553, a=None, seed=0):
        ''' Sensor model for Coral Sr/Ca = a * tos + b

        Args:
            tos (1-D array): sea surface temperature in [degC]
        '''
        if a is None:
            mu = -0.06
            std = 0.01
            a = ss.norm.rvs(loc=mu, scale=std, random_state=seed)

        SrCa = a*self.prior_tos_value + b

        if type(self.prior_tos_value) is dict:
            sn = list(self.prior_tos_value.keys())[0]
            self.ye_time = self.prior_tos_time[sn]
        else:
            self.ye_time = self.prior_tos_time
        self.ye_value = SrCa


class Coral_d18O:
    ''' The PSM is based on the forward model published by [Thompson, 2011]:
       <Thompson, D. M., T. R. Ault, M. N. Evans, J. E. Cole, and J. Emile-Geay (2011),
       Comparison of observed and simulated tropical climate trends using a forward
       model of coral \u03b418O, Geophys.Res.Lett., 38, L14706, doi:10.1029/2011GL048224.>
       Returns a numpy array that is the same size and shape as the input vectors for SST, SSS.
    '''
    def __init__(self, proxy_time, proxy_value,
            obs_sst_time, obs_sst_value,
            prior_sst_time=None, prior_sst_value=None,
            obs_sss_time=None, obs_sss_value=None,
            prior_sss_time=None, prior_sss_value=None,
            obs_d18O_time=None, obs_d18O_value=None,
            prior_d18O_time=None, prior_d18O_value=None,
        ):
        self.proxy_time = proxy_time
        self.proxy_value = proxy_value
        self.obs_sst_time = obs_sst_time
        self.obs_sst_value = obs_sst_value
        self.obs_sss_time = obs_sss_time
        self.obs_sss_value = obs_sss_value
        self.obs_d18O_time = obs_d18O_time
        self.obs_d18O_value = obs_d18O_value
        self.season_tas = self.obs_tas_value.keys()  # should be only one season
        self.prior_sst_time = prior_sst_time
        self.prior_sst_value = prior_sst_value
        self.prior_sss_time = prior_sss_time
        self.prior_sss_value = prior_sss_value
        self.prior_d18O_time = prior_d18O_time
        self.prior_d18O_value = prior_d18O_value

    def forward(self):
        pass
