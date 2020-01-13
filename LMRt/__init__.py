''' The main structure
'''
__author__ = 'Feng Zhu'
__email__ = 'fengzhu@usc.edu'
__version__ = '0.6.3'

import yaml
import os
from dotmap import DotMap
from collections import namedtuple
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import xarray as xr
import random
from scipy import stats
import statsmodels.api as sm

from . import utils
from . import visual
from . import nn

ProxyManager = namedtuple(
    'ProxyManager',
    ['all_proxies', 'ind_assim', 'ind_eval', 'sites_assim_proxy_objs', 'sites_eval_proxy_objs', 'proxy_type_list']
)

Prior = namedtuple('Prior', ['prior_dict', 'ens', 'prior_sample_indices', 'coords', 'full_state_info', 'trunc_state_info'])

Y = namedtuple('Y', ['Ye_assim', 'Ye_assim_coords', 'Ye_eval', 'Ye_eval_coords'])

Results = namedtuple('Results', ['field_ens'])


class ReconJob:
    ''' A reconstruction job
    '''

    def __init__(self):
        pwd = os.path.dirname(__file__)
        with open(os.path.join(pwd, './cfg/cfg_template.yml'), 'r') as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            self.cfg = DotMap(cfg_dict)
            self.cfg = utils.setup_cfg(self.cfg)
            print(f'pid={os.getpid()} >>> job.cfg created')

    def load_cfg(self, cfg_filepath):
        with open(cfg_filepath, 'r') as f:
            cfg_new = yaml.load(f, Loader=yaml.FullLoader)
            self.cfg = DotMap(cfg_new)
            self.cfg = utils.setup_cfg(self.cfg)
            print(f'pid={os.getpid()} >>> job.cfg updated')

    def load_proxies(self, proxies_df_filepath, metadata_df_filepath, precalib_filesdict=None,
                     select_box_lf=None, select_box_ur=None, exclude_list=None, NH_only=False, SH_only=False,
                     seed=0, verbose=False, print_assim_proxy_count=False, print_proxy_type_list=False,
                     detrend_proxy=False, detrend_method=None, detrend_kws={}):

        all_proxy_ids, all_proxies = utils.get_proxy(
            self.cfg, proxies_df_filepath, metadata_df_filepath,
            select_box_lf=select_box_lf, select_box_ur=select_box_ur,
            NH_only=NH_only, SH_only=SH_only,
            precalib_filesdict=precalib_filesdict,
            exclude_list=exclude_list,
            detrend_proxy=detrend_proxy, detrend_method=detrend_method, detrend_kws=detrend_kws,
            verbose=verbose)

        ind_assim, ind_eval = utils.generate_proxy_ind(self.cfg, len(all_proxy_ids), seed=seed)

        sites_assim_proxy_objs = []
        for i in ind_assim:
            sites_assim_proxy_objs.append(all_proxies[i])

        sites_eval_proxy_objs = []
        for i in ind_eval:
            sites_eval_proxy_objs.append(all_proxies[i])

        ptypes = []
        ptype_count = {}
        for pobj in all_proxies:
            ptype = pobj.type
            ptypes.append(ptype)
            if ptype not in ptype_count.keys():
                ptype_count[ptype] = 1
            else:
                ptype_count[ptype] += 1

        ptypes = sorted(list(set(ptypes)))
        proxy_type_list = []
        for ptype in ptypes:
            proxy_type_list.append((ptype, ptype_count[ptype]))

        self.proxy_manager = ProxyManager(
            all_proxies,
            ind_assim, ind_eval,
            sites_assim_proxy_objs, sites_eval_proxy_objs,
            proxy_type_list
        )
        print(f'pid={os.getpid()} >>> job.proxy_manager created')

        if print_assim_proxy_count:
            assim_sites_types = {}
            for pobj in self.proxy_manager.sites_assim_proxy_objs:
                if pobj.type not in assim_sites_types:
                    assim_sites_types[pobj.type] = 1
                else:
                    assim_sites_types[pobj.type] += 1

            assim_proxy_count = 0
            for pkey, pnum in sorted(assim_sites_types.items()):
                print(f'{pkey:>45s}:{pnum:5d}')
                assim_proxy_count += pnum

            print(f'{"TOTAL":>45s}:{assim_proxy_count:5d}')

        if print_proxy_type_list:
            print('\nProxy types')
            print('--------------')
            for ptype, count in self.proxy_manager.proxy_type_list:
                print(f'{ptype:>45s}:{count:5d}')

            print(f'{"TOTAL":>45s}:{len(self.proxy_manager.all_proxies):5d}')

    def load_prior(self, prior_filepath, prior_datatype, anom_reference_period=(1951, 1980), seed=0,
                   avgInterval=None, verbose=False):
        ''' Load prior variables

        Args:
            prior_filepath(str): the full path of the prior file; only one variable is required,
                and other variables under the same folder will be also loaded if specified in the configuration
            prior_datatype (str): available options: 'CMIP5'
            anom_reference_period (tuple): the period used for the calculation of climatology/anomaly
            seed (int): random number seed
        '''
        prior_dict = utils.get_prior(
            prior_filepath, prior_datatype, self.cfg,
            anom_reference_period=anom_reference_period,
            avgInterval=avgInterval,
            verbose=verbose
        )

        ens, prior_sample_indices, coords, full_state_info = utils.populate_ensemble(
            prior_dict, self.cfg, seed=seed, verbose=verbose)

        self.prior = Prior(prior_dict, ens, prior_sample_indices, coords, full_state_info, full_state_info)
        print(f'pid={os.getpid()} >>> job.prior created')
        if self.cfg.prior.regrid_method:
            ens, coords, trunc_state_info = utils.regrid_prior(self.cfg, self.prior, verbose=verbose)
            self.prior = Prior(prior_dict, ens, prior_sample_indices, coords, full_state_info, trunc_state_info)
            print(f'pid={os.getpid()} >>> job.prior regridded')

    def calibrate_OLSp(self, seasonal_inst_filesdict, precalib_savepath, seasonality=None,
                       calib_period=[1850, 2015], auto_choose_p=True, p=0, p_max=4,
                       p_dict=None, seasonality_dict=None,
                       metric='fitR2adj', search_distance=3,
                       verbose=False, pacf_threshold_dict=None, fit_args={}, print_OLSp_args=True,
                       ye_savepath=None, seasonal_GCM_filesdict=None, nobs_lb=25,
                       optimal_reg_savepath=None, normalize_proxy=False, detrend_proxy=False):
        ''' Calibrate the OLSp PSM and generate precalib & Ye files

        Args:
            seasonal_inst_filesdict (dict): instrumental environmentals, expecting 'tas' and 'pr' (optional)
            seasonal_GCM_filesdict (dict): GCM simulated envrionmentals, expecting 'tas' and 'pr' (optional)
            seasonality (list): e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] for annual, [7, 8, 9] for NH summer
        '''
        # PSM calibration
        seasonal_avg = {}
        year_ann = {}
        lat = {}
        lon = {}
        season_tags = {}
        for varname, filepath in seasonal_inst_filesdict.items():
            if varname not in ['tas', 'pr']:
                raise ValueError(f'Wrong variable name: {varname}; should be "tas" or "pr" ')

            seasonal_avg[varname], year_ann[varname], lat[varname], lon[varname] = pd.read_pickle(filepath)
            season_tags[varname] = seasonal_avg[varname].keys()

        if seasonality is not None:
            season_tag = '_'.join(str(m) for m in seasonality)
            for varname in season_tags.keys():
                season_tags[varname] = [season_tag]

        #  print(season_tags)
        OLSp_args = {
            'calib_period': calib_period,
            'auto_choose_p': auto_choose_p,
            'p_max': p_max,
            'p': p,
            'fit_args': fit_args,
            'verbose': verbose,
            'nobs_lb': nobs_lb,
        }

        if print_OLSp_args:
            print('OLSp settings:')
            print(OLSp_args)

        precalib_dict = {}
        optimal_reg_dict = {}
        for pobj in tqdm(self.proxy_manager.all_proxies, desc='Calibrating OLSp'):
            proxy_value = pobj.values.values
            if normalize_proxy:
                proxy_value = nn.norm(proxy_value)

            if detrend_proxy:
                proxy_value = utils.detrend(proxy_value)

            proxy_time = pobj.time
            lat_obs = pobj.lat
            lon_obs = pobj.lon
            lat_ind = {}
            lon_ind = {}
            for varname in lat.keys():
                lat_ind[varname], lon_ind[varname] = utils.find_closest_loc(lat[varname], lon[varname], lat_obs, lon_obs)

            if auto_choose_p:
                OLSp_args['pacf_threshold'] = pacf_threshold_dict[pobj.type]

            if p_dict is not None:
                OLSp_args['p'] = p_dict[pobj.id]

            if seasonality_dict is not None:
                optimal_season_tag_tas, optimal_season_tag_pr = seasonality_dict[pobj.id]
            else:
                optimal_season_tag_tas, optimal_season_tag_pr = None, None

            res_dict_list = []
            season_tag_list = []
            R2_adj_list = []
            mse_list = []

            for season_tag_tas in season_tags['tas']:
                if optimal_season_tag_tas is not None and season_tag_tas != optimal_season_tag_tas:
                    continue

                tas_sub = seasonal_avg['tas'][season_tag_tas][:, lat_ind['tas'], lon_ind['tas']]
                if np.all(np.isnan(tas_sub)):
                    tas_sub, _, _ = utils.search_nearest_not_nan(
                        seasonal_avg['tas'][season_tag_tas], lat_ind['tas'], lon_ind['tas'],
                        distance=search_distance,
                    )
                if 'pr' in season_tags.keys():
                    # bivariate linear regression
                    for season_tag_pr in season_tags['pr']:
                        if optimal_season_tag_pr is not None and season_tag_pr != optimal_season_tag_pr:
                            continue
                        pr_sub = seasonal_avg['pr'][season_tag_pr][:, lat_ind['pr'], lon_ind['pr']]
                        if np.all(np.isnan(pr_sub)):
                            pr_sub, _, _ = utils.search_nearest_not_nan(
                                seasonal_avg['pr'][season_tag_tas], lat_ind['pr'], lon_ind['pr'],
                                distance=search_distance,
                            )

                        res_dict = nn.OLSp(proxy_time, proxy_value, year_ann['tas'], tas_sub,
                                           time_exog2=year_ann['pr'], exog2=pr_sub, **OLSp_args)
                        if res_dict['mdl'] is not None:
                            mdl_resid = res_dict['mdl'].resid
                            mse = nn.mean_squared(mdl_resid)
                            mse_list.append(mse)
                            R2_adj = res_dict['mdl'].rsquared_adj
                            R2_adj_list.append(R2_adj)
                            res_dict_list.append(res_dict)
                            season_tag_list.append((season_tag_tas, season_tag_pr))
                        else:
                            print(f'Skipping {pobj.id} due to nobs < {nobs_lb}')

                else:
                    # univariate linear regression
                    res_dict = nn.OLSp(proxy_time, proxy_value, year_ann['tas'], tas_sub, **OLSp_args)
                    if res_dict['mdl'] is not None:
                        mdl_resid = res_dict['mdl'].resid
                        mse = nn.mean_squared(mdl_resid)
                        mse_list.append(mse)
                        R2_adj = res_dict['mdl'].rsquared_adj
                        R2_adj_list.append(R2_adj)
                        res_dict_list.append(res_dict)
                        season_tag_list.append((season_tag_tas, None))
                    else:
                        print(f'Skipping {pobj.id} due to nobs < {nobs_lb}')

            if len(res_dict_list) > 0:
                if metric == 'fitR2adj':
                    opt_idx = np.argmax(R2_adj_list)
                elif metric == 'mse':
                    opt_idx = np.argmin(mse_list)
                else:
                    raise ValueError(f'Wrong metric: {metric}; should be "fitR2adj" or "mse"')

                optimal_reg = res_dict_list[opt_idx]
                optimal_reg_resid = optimal_reg['mdl'].resid
                optimal_nobs = optimal_reg['mdl'].nobs
                optimal_mse = mse_list[opt_idx]
                optimal_R2_adj = R2_adj_list[opt_idx]
                optimal_seasonality_tag = season_tag_list[opt_idx]

                SNR = np.std(optimal_reg['mdl'].predict()) / np.std(optimal_reg_resid)

                precalib_dict[(pobj.type, pobj.id)] = {
                    'lat': pobj.lat,
                    'lon': pobj.lon,
                    'elev': pobj.elev,
                    'ptype': pobj.type,
                    'Seasonality': optimal_seasonality_tag,
                    'PSMresid': optimal_reg['mdl'].resid,
                    'PSMmse': optimal_mse,
                    'fitR2adj': optimal_R2_adj,
                    'SNR': SNR,
                    'p': optimal_reg['p'],
                }
                optimal_reg_dict[(pobj.type, pobj.id)] = optimal_reg
            else:
                print(f'optimal_reg is None for {pobj.id} due to nobs < {nobs_lb}')
                #  precalib_dict[(pobj.type, pobj.id)] = None
                #  optimal_reg_dict[pobj.id] = None

        with open(precalib_savepath, 'wb') as f:
            pickle.dump(precalib_dict, f)

        print(f'\npid={os.getpid()} >>> Saving calibration results to {precalib_savepath}')

        if optimal_reg_savepath is not None:
            with open(optimal_reg_savepath, 'wb') as f:
                pickle.dump(optimal_reg_dict, f)

            print(f'\npid={os.getpid()} >>> Saving optimal_reg_dict to {optimal_reg_savepath}')

        # Calculate Ye
        if ye_savepath is not None:
            seasonal_avg = {}
            year_ann = {}
            lat = {}
            lon = {}
            for varname, filepath in seasonal_GCM_filesdict.items():
                if varname not in ['tas', 'pr']:
                    raise ValueError(f'Wrong variable name: {varname}; should be "tas" or "pr" ')

                seasonal_avg[varname], year_ann[varname], lat[varname], lon[varname] = pd.read_pickle(filepath)

            pid_map = {}
            nproxy = len(precalib_dict)
            ye_out = np.ndarray((nproxy, np.size(year_ann['tas'])))
            i = 0
            for k, v in tqdm(precalib_dict.items(), desc='Calculating Ye', total=nproxy):
                ptype, pid = k
                pid_map[pid] = i
                optimal_reg = optimal_reg_dict[k]
                p = optimal_reg['p']
                season_tag_tas, season_tag_pr = v['Seasonality']

                lat_obs = v['lat']
                lon_obs = v['lon']
                lat_ind = {}
                lon_ind = {}
                for varname in lat.keys():
                    lat_ind[varname], lon_ind[varname] = utils.find_closest_loc(lat[varname], lon[varname], lat_obs, lon_obs)

                exog_dict = {}
                tas_sub = seasonal_avg['tas'][season_tag_tas][:, lat_ind['tas'], lon_ind['tas']]
                exog_dict['exog'] = tas_sub
                if p > 0:
                    for k in range(1, p+1):
                        exog_dict[f'exog_lag{k}'] = np.pad(tas_sub[:-k], (k, 0), 'edge')

                if season_tag_pr is not None:
                    pr_sub = seasonal_avg['pr'][season_tag_pr][:, lat_ind['pr'], lon_ind['pr']]
                    exog_dict['exog2'] = pr_sub

                ye_out[i] = optimal_reg['mdl'].predict(exog=exog_dict)
                i += 1

            np.savez(ye_savepath, pid_index_map=pid_map, ye_vals=ye_out)
            print(f'\npid={os.getpid()} >>> Saving Ye to {ye_savepath}')

    def build_precalib_files(
        self, ptypes, psm_name, precalib_savepath,
        calib_refsdict=None,
        ref_period=[1951, 1980], calib_period=[1850, 2015],
        precalc_avg_pathdict=None, make_yr_mm_nan=True,
        seasonality = {
            'Tree Rings_WidthBreit': {
                'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
            },
            'Tree Rings_WidthPages2': {
                'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
            },
            'Tree Rings_WoodDensity': {
                'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]]},
            'Tree Rings_Isotopes': {
                'seasons_T': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
                'seasons_M': [[1,2,3,4,5,6,7,8,9,10,11,12],[6,7,8],[3,4,5,6,7,8],[6,7,8,9,10,11],[-12,1,2],[-9,-10,-11,-12,1,2],[-12,1,2,3,4,5]],
            },
        }, nproc=4, nobs_lb=25, verbose=False, output_optimal_reg=False):
        ''' Build precalibration files for linear/bilinear PSMs

        Args:
            calib_refsdict (dict):
                e.g., {'T': ('GISTEMP', GISTEMP_filepath), 'M': ('GPCC', GPCC_filepath)}
            precalc_avg_pathdict (dict):
                e.g., {'T': T_seasonal_avg_filepath), 'M': M_seasonal_avg_filepath)}

            NOTE: one of the above two dicts must be provided
        '''
        if precalc_avg_pathdict is None:
            refsdict = {}
            avgMonths_setdict = {}
            for var_name, v in calib_refsdict.items():
                dataset_name, dataset_path = v
                refsdict[dataset_name] = dataset_path

                seasons_list = []
                season_tag = f'seasons_{var_name}'
                for k, season_dict in seasonality.items():
                    seasons_list.append(season_dict[season_tag])

                for pobj in self.proxy_manager.all_proxies:
                    seasons_list.append(list(pobj.seasonality))

                seasons_set = set(map(tuple, seasons_list))
                avgMonths_setdict[var_name] = list(map(list, seasons_set))

            inst_field, inst_time, inst_lat, inst_lon = utils.load_inst_analyses(
                refsdict, var='field', verif_yrs=None,
                ref_period=ref_period, outfreq='monthly',
            )

            precalc_avg = {}
            for var_name, v in calib_refsdict.items():
                dataset_name, dataset_path = v
                print(f'>>> Calculating seasonal-average on: {dataset_name} ...')
                #  print(avgMonths_setdict[varname])
                var_ann_dict, year_ann = utils.calc_seasonal_avg(
                    inst_field[dataset_name], inst_time[dataset_name],
                    lat=inst_lat[dataset_name], lon=inst_lon[dataset_name],
                    seasonality=avgMonths_setdict[var_name], make_yr_mm_nan=make_yr_mm_nan,
                    verbose=True,
                )

                precalc_avg[var_name] = {
                    'var_ann_dict': var_ann_dict,
                    'year_ann': year_ann,
                    'lat': inst_lat[dataset_name],
                    'lon': inst_lon[dataset_name],
                }

        else:
            precalc_avg = {}
            for var_name, data_path in precalc_avg_pathdict.items():
                with open(data_path, 'rb') as f:
                    var_ann_dict, year_ann, lat, lon =  pickle.load(f)

                precalc_avg[var_name] = {
                    'var_ann_dict': var_ann_dict,
                    'year_ann': year_ann,
                    'lat': lat,
                    'lon': lon,
                }

        precalib_dict = utils.calibrate_psm(self.proxy_manager, ptypes, psm_name,
                                            precalc_avg, calib_period=calib_period,
                                            seasonality=seasonality, nproc=nproc,
                                            verbose=verbose, nobs_lb=nobs_lb, output_optimal_reg=output_optimal_reg)

        with open(precalib_savepath, 'wb') as f:
            pickle.dump(precalib_dict, f)

        print(f'\npid={os.getpid()} >>> Saving calibration results to {precalib_savepath}')

    def build_ye_files(self, ptypes, psm_name, prior_filesdict, ye_savepath,
                       rename_vars={'tmp': 'tas', 'pre': 'pr', 'd18O': 'd18Opr', 'tos': 'sst', 'sos': 'sss'},
                       precalib_filesdict=None, verbose=False, useLib='netCDF4', nproc=1, elev_model=None,
                       lat_str='lat', lon_str='lon',
                       repeat_frac_threashold=0.5,
                       match_std=True, match_mean=True, tas_bias=None, pr_factor=None,
                       calc_anomaly=None, ref_period=(1951, 1980), precalc_avg_pathdict=None, **psm_params):
        ''' Build precalculated Ye files from priors

        Args:
            ptype (str): the target proxy type
            psm_name (str): the name of the PSM used to forward prior variables
            prior_filesdict (dict): e.g. {'tas': tas_filepath, 'pr': pr_filepath}
            ye_savepath (str): the filepath to save precalculated Ye
            rename_vars (dict): a map used to rename the variable names,
                e.g., {'d18O': 'd18Opr', 'tos': 'sst', 'sos': 'sss'}
            psm_params (kwargs): the specific parameters for certain PSMs
            precalc_avg_pathdict (dict):
                e.g., {'T': T_seasonal_avg_filepath), 'M': M_seasonal_avg_filepath)}

        '''

        if type(ptypes) is not list:
            ptypes = [ptypes]

        if psm_name in ['linear', 'bilinear']:
            precalib_filepath = precalib_filesdict[psm_name]
            precalib_data = pd.read_pickle(precalib_filepath)

            if precalc_avg_pathdict is None:
                # load environmental variables
                if calc_anomaly is None:
                    calc_anomaly = True

                lat_model, lon_model, time_model, prior_vars = utils.get_env_vars(
                    prior_filesdict, rename_vars=rename_vars,
                    useLib=useLib, calc_anomaly=calc_anomaly, ref_period=ref_period,
                    lat_str=lat_str, lon_str=lon_str,
                    tas_bias=tas_bias, pr_factor=pr_factor,
                    verbose=verbose
                )

                seasons_list = {}
                avgMonths_setdict = {}
                if psm_name == 'linear':
                    seasons_list['T'] = []
                    for k, v in precalib_data.items():
                        seasons_list['T'].append(v['Seasonality'])

                elif psm_name == 'bilinear':
                    seasons_list['T'] = []
                    seasons_list['M'] = []
                    for k, v in precalib_data.items():
                        seasons_list['T'].append(v['Seasonality'][0])
                        seasons_list['M'].append(v['Seasonality'][1])

                for var_name, s_list in seasons_list.items():
                    seasons_set = set(map(tuple, s_list))
                    avgMonths_setdict[var_name] = list(map(list, seasons_set))

                precalc_avg = {}
                for var_name, s_set in avgMonths_setdict.items():
                    var_rename = {
                        'T': 'tas',
                        'M': 'pr',
                    }

                    print(f'>>> Calculating seasonal-average on: {var_rename[var_name]} ...')
                    var_ann_dict, year_ann = utils.calc_seasonal_avg(
                        prior_vars[var_rename[var_name]], time_model,
                        lat=lat_model, lon=lon_model,
                        seasonality=avgMonths_setdict[var_name],
                        verbose=True, make_yr_mm_nan=False,  # keep the 1st yr even the avgMonths requires previous yr
                    )

                    precalc_avg[var_name] = {
                        'var_ann_dict': var_ann_dict,
                        'year_ann': year_ann,
                        'lat': lat_model,
                        'lon': lon_model,
                    }

            else:
                precalc_avg = {}
                for var_name, data_path in precalc_avg_pathdict.items():
                    with open(data_path, 'rb') as f:
                        var_ann_dict, year_ann, lat, lon =  pickle.load(f)

                    precalc_avg[var_name] = {
                        'var_ann_dict': var_ann_dict,
                        'year_ann': year_ann,
                        'lat': lat,
                        'lon': lon,
                    }

            pid_map, ye_out = utils.calc_ye_linearPSM(
                self.proxy_manager, ptypes, psm_name,
                precalib_data, precalc_avg,
                nproc=nproc, verbose=verbose
            )
        else:
            # load environmental variables
            if calc_anomaly is None:
                calc_anomaly = False

            lat_model, lon_model, time_model, prior_vars = utils.get_env_vars(
                prior_filesdict, rename_vars=rename_vars,
                useLib=useLib, calc_anomaly=calc_anomaly, ref_period=ref_period,
                lat_str=lat_str, lon_str=lon_str, verbose=verbose
            )

            pid_map, ye_out = utils.calc_ye(
                self.proxy_manager, ptypes, psm_name,
                lat_model, lon_model, time_model, prior_vars,
                elev_model=elev_model,
                match_std=match_std, match_mean=match_mean,
                verbose=verbose, **psm_params
            )

        np.savez(ye_savepath, pid_index_map=pid_map, ye_vals=ye_out)
        print(f'\npid={os.getpid()} >>> Saving Ye to {ye_savepath}')

    def build_pseudoproxies_from_ye(self, db_proxies_filepath, db_metadata_filepath,
                                    precalib_filepath, ye_filepath,
                                    proxies_savepath=None, metadata_savepath=None,
                                    timespan=[850, 2005], add_noise=True,
                                    SNR=None, noise_type='white', g=None):
        df_proxies = pd.read_pickle(db_proxies_filepath)
        df_metadata = pd.read_pickle(db_metadata_filepath)
        pid_selected = []
        for pobj in self.proxy_manager.all_proxies:
            pid = pobj.id
            pid_selected.append(pid)

        with open(precalib_filepath, 'rb') as f:
            precalib_dict = pickle.load(f)

        seasonality_dict = {}
        SNR_dict = {}
        resid_dict = {}
        for k, v in precalib_dict.items():
            ptype, pid = k
            seasonality_dict[pid] = v['Seasonality']
            SNR_dict[pid] = v['SNR']
            resid_dict[pid] = v['PSMresid']

        df_metadata_out =  df_metadata.loc[df_metadata['Proxy ID'].isin(pid_selected)]
        df_proxies_out =  pd.DataFrame({'Year C.E.': np.arange(timespan[0], timespan[1]+1)})
        df_proxies_out.set_index('Year C.E.', drop=True, inplace=True)

        for pid in pid_selected:
            df_proxies_out[pid] = np.nan

        for idx, row in df_metadata_out.iterrows():
            df_metadata_out.at[idx, 'Seasonality'] = seasonality_dict[row['Proxy ID']]

        ye_data = np.load(ye_filepath, allow_pickle=True)
        pid_idx_map = ye_data['pid_index_map'][()]
        ye_vals = ye_data['ye_vals']

        for i, pobj in tqdm(enumerate(self.proxy_manager.all_proxies), total=len(self.proxy_manager.all_proxies)):
            pid = pobj.id
            lat_obs, lon_obs = pobj.lat, pobj.lon
            oldest_yr, youngest_yr = timespan
            df_metadata_out.at[df_metadata_out['Proxy ID']==pid, 'Oldest (C.E.)'] = oldest_yr
            df_metadata_out.at[df_metadata_out['Proxy ID']==pid, 'Youngest (C.E.)'] = youngest_yr

            idx = pid_idx_map[pid]
            signal = ye_vals[idx]  # no noise
            if SNR is None:
                SNR = SNR_dict[pid]

            if noise_type=='white':
                noise = np.random.normal(0, 1, np.size(signal))
            elif noise_type=='AR1':
                if g is None:
                    ar1_mod = sm.tsa.AR(resid_dict[pid].values, missing='drop').fit(maxlag=1)
                    g = ar1_mod.params[1]

                ar = np.r_[1, -g]
                ma = np.r_[1, 0.0]
                noise = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=np.size(signal), burnin=50)
            else:
                raise ValueError(f'Wrong noise type {noise_type}; should be "white" or "AR1"')

            noise /= np.std(noise)
            signal_std = np.std(signal)
            noise_std = signal_std / SNR
            noise *= noise_std
            pseudo_value = signal + noise

            df_proxies_out[pid] = pseudo_value

        if metadata_savepath:
            df_metadata_out.to_pickle(metadata_savepath)
            print(f'\npid={os.getpid()} >>> Saving metadata to {metadata_savepath}')

        if proxies_savepath:
            df_proxies_out.to_pickle(proxies_savepath)
            print(f'\npid={os.getpid()} >>> Saving proxies to {proxies_savepath}')

        return df_metadata_out, df_proxies_out

    def build_proxies_from_df(self, df, exclude_list=None, time_year=np.arange(1, 2020),
                              time_col='year', value_col='paleoData_values',
                              metadata_savepath=None, proxies_savepath=None):
        ''' Build proxies database from a DataFrame

        Args:
            df (DataFrame): the DataFrame of the proxies
        '''
        df_metadata = pd.DataFrame(columns=[
            'Proxy ID',
            'Archive type',
            'Lat (N)',
            'Lon (E)',
            'Elev',
            'Proxy measurement',
            'Databases',
            'Resolution (yr)',
            'Seasonality',
        ])
        df_metadata['Seasonality'] = np.nan
        df_metadata['Seasonality'] = df_metadata['Seasonality'].astype(object)

        df_proxies = pd.DataFrame(index=time_year)

        archive_dict = {
            'coral': 'Corals and Sclerosponges',
            'sclerosponge': 'Corals and Sclerosponges',
            'glacier ice': 'Ice Cores',
            'lake sediment': 'Lake Cores',
            'marine sediment': 'Marine Cores',
            'speleothem': 'Speleothems',
            'tree': 'Tree Rings',
            'bivalve': 'Bivalve',
            'borehole': 'Borehole',
            'documents': 'Documents',
            'hybrid': 'Hybrid',
        }

        for i, row in tqdm(df.iterrows(), total=len(df)):
            p2k_id = row['paleoData_pages2kID']
            p2k_dsn = row['dataSetName']
            p2k_archive = row['archiveType']
            p2k_lat = row['geo_meanLat']
            p2k_lon = row['geo_meanLon']
            p2k_elev = row['geo_meanElev']
            p2k_vn = row['paleoData_variableName']
            p2k_season = row['seasonality']

            if p2k_archive == 'glacier ice' and p2k_vn == 'd18O1':
                p2k_vn = 'd18O'

            if p2k_archive == 'tree' and p2k_vn == 'temperature1':
                p2k_vn = 'temperature'

            if p2k_archive == 'lake sediment' and (p2k_vn == 'temperature1' or p2k_vn == 'temperature3'):
                p2k_vn = 'temperature'

            pid = f'PAGES2kv2_{p2k_dsn}_{p2k_id}:{p2k_vn}'
            df_metadata.loc[i, 'Proxy ID'] = pid
            df_metadata.loc[i, 'Archive type'] = archive_dict[p2k_archive]
            df_metadata.loc[i, 'Lat (N)'] = p2k_lat
            df_metadata.loc[i, 'Lon (E)'] = np.mod(p2k_lon, 360)
            df_metadata.loc[i, 'Elev'] = p2k_elev
            df_metadata.loc[i, 'Proxy measurement'] = p2k_vn
            df_metadata.loc[i, 'Databases'] = ['PAGES2kv2']
            df_metadata.at[i, 'Seasonality'] = p2k_season

            p2k_time = np.array(row[time_col], dtype=np.float)
            p2k_value = np.array(row[value_col], dtype=np.float)
            p2k_time, p2k_value = utils.clean_ts(p2k_time, p2k_value)
            time_annual, data_annual, proxy_resolution = utils.compute_annual_means(p2k_time, p2k_value, 0.5, 'calendar year')
            data_annual = np.squeeze(data_annual)
            df_metadata.loc[i, 'Resolution (yr)'] = proxy_resolution
            df_metadata.loc[i, 'Oldest (C.E.)'] = np.min(time_annual)
            df_metadata.loc[i, 'Youngest (C.E.)'] = np.max(time_annual)

            series = pd.Series(data_annual, index=time_annual)
            df_proxies[pid] = series

        if metadata_savepath:
            df_metadata.to_pickle(metadata_savepath)

        if proxies_savepath:
            df_proxies.to_pickle(proxies_savepath)

        return df_metadata, df_proxies

    def load_ye_files(self, ye_filesdict, verbose=False):
        ''' Load precalculated Ye files

        Args:
            ye_filesdict (dict): e.g. {'linear': linear_filepath, 'blinear': bilinear_filepath}
            proxy_set (str): 'assim' or 'eval'
        '''
        Ye_assim, Ye_assim_coords = utils.get_ye(self.proxy_manager,
                                             self.prior.prior_sample_indices,
                                             ye_filesdict=ye_filesdict,
                                             proxy_set='assim',
                                             verbose=verbose)

        Ye_eval, Ye_eval_coords = utils.get_ye(self.proxy_manager,
                                             self.prior.prior_sample_indices,
                                             ye_filesdict=ye_filesdict,
                                             proxy_set='eval',
                                             verbose=verbose)

        self.ye = Y(Ye_assim, Ye_assim_coords, Ye_eval, Ye_eval_coords)
        print(f'pid={os.getpid()} >>> job.ye created')

    def run_da(self, recon_years=None, proxy_inds=None, verbose=False, mode='normal'):
        cfg = self.cfg
        prior = self.prior
        proxy_manager = self.proxy_manager

        Ye_assim = self.ye.Ye_assim
        Ye_assim_coords = self.ye.Ye_assim_coords
        assim_proxy_count = np.shape(Ye_assim)[0]

        Ye_eval = self.ye.Ye_eval
        Ye_eval_coords = self.ye.Ye_eval_coords
        eval_proxy_count = np.shape(Ye_eval)[0]

        if recon_years is None:
            yr_start = cfg.core.recon_period[0]
            yr_end = cfg.core.recon_period[-1]
            recon_years = list(range(yr_start, yr_end+1))
        else:
            yr_start, yr_end = recon_years[0], recon_years[-1]

        nyr = len(recon_years)
        print(f'\npid={os.getpid()} >>> Recon. period: [{yr_start}, {yr_end}]; {nyr} years')

        Xb_one = prior.ens
        Xb_one_aug = np.append(Xb_one, Ye_assim, axis=0)
        Xb_one_aug = np.append(Xb_one_aug, Ye_eval, axis=0)
        Xb_one_coords = np.append(prior.coords, Ye_assim_coords, axis=0)
        Xb_one_coords = np.append(Xb_one_coords, Ye_eval_coords, axis=0)

        grid = utils.make_grid(prior)

        var_names = prior.trunc_state_info.keys()
        ibeg = {}
        iend = {}
        field_ens = {}
        for name in var_names:
            ibeg[name] = prior.trunc_state_info[name]['pos'][0]
            iend[name] = prior.trunc_state_info[name]['pos'][1]
            field_ens[name] = np.zeros((nyr, grid.nens, grid.nlat, grid.nlon))

        update_func = {
            'normal': utils.update_year,
            'optimal': utils.update_year_optimal,
        }

        for yr_idx, target_year in enumerate(tqdm(recon_years, desc=f'KF updating (pid={os.getpid()})')):
            res = update_func[mode](
                yr_idx, target_year,
                cfg, Xb_one_aug, Xb_one_coords, prior, proxy_manager.sites_assim_proxy_objs,
                assim_proxy_count, eval_proxy_count, grid,
                ibeg, iend, verbose=verbose
            )
            for name in var_names:
                field_ens[name][yr_idx] = res[name]

        self.res = Results(field_ens)
        print(f'\npid={os.getpid()} >>> job.res created')

    def save_results(self, save_dirpath, seed=0, recon_years=None, dtype=np.float32,
                     output_geo_mean=False, target_lats=[], target_lons=[],
                     compress_dict={'zlib': True, 'least_significant_digit': 1, 'complevel': 9},
                     output_full_ens=False):
        if recon_years is None:
            yr_start = self.cfg.core.recon_period[0]
            yr_end = self.cfg.core.recon_period[-1]
            recon_years = list(range(yr_start, yr_end+1))

        utils.save_to_netcdf(
                self.prior,
                self.res.field_ens,
                recon_years,
                seed,
                save_dirpath,
                dtype=dtype,
                output_geo_mean=output_geo_mean,
                target_lats=target_lats,
                target_lons=target_lons,
                compress_dict=compress_dict,
                output_full_ens=output_full_ens,
            )

        save_path = os.path.join(save_dirpath, f'job_r{seed:02d}.nc')
        print(f'\npid={os.getpid()} >>> Saving results to {save_path}')
        print('-----------------------------------------------------')
        print('')

    def run(self, prior_filepath, prior_datatype, db_proxies_filepath, db_metadata_filepath,
            recon_years=None, seed=0, precalib_filesdict=None, ye_filesdict=None,
            anom_reference_period=(1951, 1980), avgInterval=None,
            select_box_lf=None, select_box_ur=None, exclude_list=None,
            output_geo_mean=False, target_lats=[], target_lons=[],
            compress_dict={'zlib': True, 'least_significant_digit': 1, 'complevel': 9},
            print_proxy_type_list=False, print_assim_proxy_count=False,
            output_full_ens=False, verbose=False, save_dirpath=None, mode='normal'):

        self.load_prior(prior_filepath, prior_datatype,
                        anom_reference_period=anom_reference_period,
                        avgInterval=avgInterval,
                        verbose=verbose, seed=seed)

        self.load_proxies(db_proxies_filepath, db_metadata_filepath, precalib_filesdict=precalib_filesdict,
                          select_box_lf=select_box_lf, select_box_ur=select_box_ur,
                          print_proxy_type_list=print_proxy_type_list,
                          print_assim_proxy_count=print_assim_proxy_count,
                          exclude_list=exclude_list, verbose=verbose, seed=seed)

        self.load_ye_files(ye_filesdict=ye_filesdict, verbose=verbose)

        self.run_da(recon_years=recon_years, mode=mode, verbose=verbose)

        if save_dirpath:
            self.save_results(save_dirpath, seed=seed, recon_years=recon_years,
                              output_geo_mean=output_geo_mean, target_lats=target_lats, target_lons=target_lons,
                              output_full_ens=output_full_ens, compress_dict=compress_dict)
