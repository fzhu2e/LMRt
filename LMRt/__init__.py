''' The main structure
'''
__author__ = 'Feng Zhu'
__email__ = 'fengzhu@usc.edu'
__version__ = '0.5.1'

import yaml
import os
from dotmap import DotMap
from collections import namedtuple
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import xarray as xr

from . import utils

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
                     seed=0, verbose=False, print_assim_proxy_count=False, print_proxy_type_list=False):

        all_proxy_ids, all_proxies = utils.get_proxy(self.cfg, proxies_df_filepath, metadata_df_filepath,
                                                     precalib_filesdict=precalib_filesdict, verbose=verbose)

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

    def load_prior(self, prior_filepath, prior_datatype, anom_reference_period=(1951, 1980), seed=0, verbose=False):
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
            anom_reference_period=anom_reference_period, verbose=verbose
        )

        ens, prior_sample_indices, coords, full_state_info = utils.populate_ensemble(
            prior_dict, self.cfg, seed=seed, verbose=verbose)

        self.prior = Prior(prior_dict, ens, prior_sample_indices, coords, full_state_info, full_state_info)
        print(f'pid={os.getpid()} >>> job.prior created')
        if self.cfg.prior.regrid_method:
            ens, coords, trunc_state_info = utils.regrid_prior(self.cfg, self.prior, verbose=verbose)
            self.prior = Prior(prior_dict, ens, prior_sample_indices, coords, full_state_info, trunc_state_info)
            print(f'pid={os.getpid()} >>> job.prior regridded')

    def build_precalib_files(
        self, ptypes, psm_name, calib_refsdict, precalib_savepath,
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
        }, nproc=4, verbose=False):
        ''' Build precalibration files for linear/bilinear PSMs

        Args:
            calib_refsdict (dict):
                e.g., {'T': ('GISTEMP', GISTEMP_filepath), 'M': ('GPCC', GPCC_filepath)}
            precalc_avg_pathdict (dict):
                e.g., {'T': T_seasonal_avg_filepath), 'M': M_seasonal_avg_filepath)}
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
                    seasons_list += season_dict[season_tag]

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
                                            verbose=verbose)

        with open(precalib_savepath, 'wb') as f:
            pickle.dump(precalib_dict, f)

        print(f'\npid={os.getpid()} >>> Saving calibration results to {precalib_savepath}')

    def build_ye_files(self, ptypes, psm_name, prior_filesdict, ye_savepath,
                       rename_vars={'tmp': 'tas', 'd18O': 'd18Opr', 'tos': 'sst', 'sos': 'sss'},
                       precalib_filesdict=None, verbose=False, useLib='netCDF4', nproc=4,
                       lat_str='lat', lon_str='lon', match_std=True, match_mean=True,
                       calc_anomaly=True, ref_period=(1951, 1980), precalc_avg_pathdict=None, **psm_params):
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
                lat_model, lon_model, time_model, prior_vars = utils.get_env_vars(
                    prior_filesdict, rename_vars=rename_vars,
                    useLib=useLib, calc_anomaly=calc_anomaly, ref_period=ref_period,
                    lat_str=lat_str, lon_str=lon_str, verbose=verbose
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
            lat_model, lon_model, time_model, prior_vars = utils.get_env_vars(
                prior_filesdict, rename_vars=rename_vars,
                useLib=useLib, calc_anomaly=calc_anomaly, ref_period=ref_period,
                lat_str=lat_str, lon_str=lon_str, verbose=verbose
            )

            pid_map, ye_out = utils.calc_ye(
                self.proxy_manager, ptypes, psm_name,
                lat_model, lon_model, time_model, prior_vars,
                match_std=match_std, match_mean=match_mean,
                verbose=verbose, **psm_params
            )

        np.savez(ye_savepath, pid_index_map=pid_map, ye_vals=ye_out)
        print(f'\npid={os.getpid()} >>> Saving Ye to {ye_savepath}')

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

    def save_results(self, save_dirpath, seed=0, recon_years=None):
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
            )

        save_path = os.path.join(save_dirpath, f'job_r{seed:02d}.nc')
        print(f'\npid={os.getpid()} >>> Saving results to {save_path}')
        print('-----------------------------------------------------')
        print('')

    def run(self, prior_filepath, prior_datatype, db_proxies_filepath, db_metadata_filepath,
            recon_years=None, seed=0, precalib_filesdict=None, ye_filesdict=None,
            verbose=False, print_proxy_type_list=False, print_assim_proxy_count=False, save_dirpath=None, mode='normal'):

        self.load_prior(prior_filepath, prior_datatype, verbose=verbose, seed=seed)
        self.load_proxies(db_proxies_filepath, db_metadata_filepath, precalib_filesdict=precalib_filesdict,
                          print_proxy_type_list=print_proxy_type_list,
                          print_assim_proxy_count=print_assim_proxy_count,
                          verbose=verbose, seed=seed)
        self.load_ye_files(ye_filesdict=ye_filesdict, verbose=verbose)

        self.run_da(recon_years=recon_years, mode=mode, verbose=verbose)

        if save_dirpath:
            self.save_results(save_dirpath, seed=seed, recon_years=recon_years)
