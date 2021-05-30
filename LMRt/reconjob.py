import pickle
import numpy as np
import random
import os
import pandas as pd
import yaml
import copy
from tqdm import tqdm
from . import utils
from . import visual
import xarray as xr

from .proxy import ProxyDatabase
from .gridded import Dataset
from .utils import (
    pp,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
    cfg_abspath,
    cwd_abspath,
    geo_mean,
    nino_indices,
    calc_tpi,
    global_hemispheric_means,
)
from .da import (
    enkf_update_array,
    cov_localization,
)


class ReconJob:
    ''' Reconstruction Job

    General rule of loading parameters: load from the YAML first if available, then update with the parameters in the function calling,
      so the latter has a higher priority
    '''
    def __init__(self, configs=None, proxydb=None, prior=None, obs=None):
        self.configs = configs
        self.proxydb = proxydb
        self.prior = prior
        self.obs = obs

    def copy(self):
        return copy.deepcopy(self)

    def load_configs(self, cfg_path=None, job_dirpath=None, verbose=False):
        ''' Load the configuration YAML file

        self.configs will be updated

        Parameters
        ----------

        cfg_path : str
            the path of a configuration YAML file

        '''
        pwd = os.path.dirname(__file__)
        if cfg_path is None:
            cfg_path = os.path.abspath(os.path.join(pwd, './cfg/cfg_template.yml'))

        self.cfg_path = cfg_path
        if verbose: p_header(f'LMRt: job.load_configs() >>> loading reconstruction configurations from: {cfg_path}')

        self.configs = yaml.safe_load(open(cfg_path, 'r'))
        if verbose: p_success(f'LMRt: job.load_configs() >>> job.configs created')

        if job_dirpath is None:
            if os.path.isabs(self.configs['job_dirpath']):
                job_dirpath = self.configs['job_dirpath']
            else:
                job_dirpath = cfg_abspath(self.cfg_path, self.configs['job_dirpath'])
        else:
            job_dirpath = cwd_abspath(job_dirpath)

        self.configs['job_dirpath'] = job_dirpath
        os.makedirs(job_dirpath, exist_ok=True)
        if verbose:
            p_header(f'LMRt: job.load_configs() >>> job.configs["job_dirpath"] = {job_dirpath}')
            p_success(f'LMRt: job.load_configs() >>> {job_dirpath} created')
            pp.pprint(self.configs)

    def load_proxydb(self, path=None, verbose=False, load_df_kws=None):
        ''' Load the proxy database

        self.proxydb will be updated

        Parameters
        ----------
        
        proxydb_path : str
            if given, should point to a pickle file with a Pandas DataFrame underlying

        '''
        # update self.configs with not None parameters in the function calling
        if path is None:
            if os.path.isabs(self.configs['proxydb_path']):
                path = self.configs['proxydb_path']
            else:
                path = cfg_abspath(self.cfg_path, self.configs['proxydb_path'])
        else:
            path = cwd_abspath(path)

        self.configs['proxydb_path'] = path
        if verbose: p_header(f'LMRt: job.load_proxydb() >>> job.configs["proxydb_path"] = {path}')

        # load proxy database
        proxydb = ProxyDatabase()
        proxydb_df = pd.read_pickle(self.configs['proxydb_path'])
        load_df_kws = {} if load_df_kws is None else load_df_kws.copy()
        proxydb.load_df(proxydb_df, ptype_psm=self.configs['ptype_psm'],
            ptype_season=self.configs['ptype_season'], verbose=verbose, **load_df_kws)
        if verbose: p_success(f'LMRt: job.load_proxydb() >>> {proxydb.nrec} records loaded')

        proxydb.source = self.configs['proxydb_path']
        self.proxydb = proxydb
        if verbose: p_success(f'LMRt: job.load_proxydb() >>> job.proxydb created')

    def filter_proxydb(self, ptype_psm=None, dt=1, pids=None, verbose=False):
        if ptype_psm is None:
            ptype_psm = self.configs['ptype_psm']
        else:
            self.configs['ptype_psm'] = ptype_psm
            if verbose: p_header(f'LMRt: job.filter_proxydb() >>> job.configs["ptype_psm"] = {ptype_psm}')

        proxydb = self.proxydb.copy()
        if self.configs['ptype_psm'] is not None:
            ptype_list = list(self.configs['ptype_psm'].keys())
            if verbose: p_header(f'LMRt: job.filter_proxydb() >>> filtering proxy records according to: {ptype_list}')
            proxydb.filter_ptype(ptype_list, inplace=True)

        proxydb.filter_dt(dt, inplace=True)

        if pids is not None:
            self.configs['assim_pids'] = pids
            if verbose: p_header(f'LMRt: job.filter_proxydb() >>> job.configs["assim_pids"] = {pids}')

        if 'assim_pids' in self.configs and self.configs['assim_pids'] is not None:
            proxydb.filter_pids(self.configs['assim_pids'], inplace=True)

        if verbose: p_success(f'LMRt: job.filter_proxydb() >>> {proxydb.nrec} records remaining')

        self.proxydb = proxydb

    def seasonalize_proxydb(self, ptype_season=None, verbose=False):
        if ptype_season is None:
            ptype_season = self.configs['ptype_season']
        else:
            self.configs['ptype_season'] = ptype_season
            if verbose: p_header(f'LMRt: job.seasonalize_proxydb() >>> job.configs["ptype_season"] = {ptype_season}')

        proxydb = self.proxydb.copy()
        if self.configs['ptype_season'] is not None:
            if verbose: p_header(f'LMRt: job.seasonalize_proxydb() >>> seasonalizing proxy records according to: {self.configs["ptype_season"]}')
            proxydb.seasonalize(self.configs['ptype_season'], inplace=True)
            if verbose: p_success(f'LMRt: job.seasonalize_proxydb() >>> {proxydb.nrec} records remaining')

        self.proxydb = proxydb
        if verbose: p_success(f'LMRt: job.seasonalize_proxydb() >>> job.proxydb updated')

    def load_prior(self, path_dict=None, varname_dict=None, verbose=False, anom_period=None):
        ''' Load model prior fields

        Parameters
        ----------

        path_dict: dict
            a dict of environmental variables

        varname_dict: dict
            a dict to map variable names, e.g. {'tas': 'sst'} means 'tas' is named 'sst' in the input NetCDF file
        
        '''
        # update self.configs with not None parameters in the function calling
        if path_dict is None:
            path_dict = cfg_abspath(self.cfg_path, self.configs['prior_path'])
            self.configs['prior_path'] = path_dict
        else:
            self.configs['prior_path'] = cwd_abspath(path_dict)
            if verbose: p_header(f'LMRt: job.load_prior() >>> job.configs["prior_path"] = {path_dict}')

        if anom_period is None:
            anom_period = self.configs['anom_period']
        else:
            self.configs['anom_period'] = anom_period
            if verbose: p_header(f'LMRt: job.load_prior() >>> job.configs["anom_period"] = {anom_period}')

        vn_dict = {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon',
        }
        if 'prior_varname' in self.configs:
            vn_dict.update(self.configs['prior_varname'])
        if varname_dict is not None:
            vn_dict.update(varname_dict)
        self.configs['prior_varname'] = vn_dict

        # load data
        if verbose: p_header(f'LMRt: job.load_prior() >>> loading model prior fields from: {self.configs["prior_path"]}')
        ds = Dataset()
        ds.load_nc(self.configs['prior_path'], varname_dict=self.configs['prior_varname'], anom_period=anom_period, inplace=True)

        if verbose:
            p_hint('LMRt: job.load_prior() >>> raw prior')
            print(ds)

        self.prior = ds
        if verbose: p_success(f'LMRt: job.load_prior() >>> job.prior created')

    def seasonalize_ds_for_psm(self, ds_type=None, seasonalized_ds_path=None, save_path=None, ptype_season=None, verbose=False):

        if seasonalized_ds_path is not None and os.path.exists(seasonalized_ds_path):
            with open(seasonalized_ds_path, 'rb') as f:
                if ds_type == 'prior':
                    self.seasonalized_prior = pickle.load(f)
                elif ds_type == 'obs':
                    self.seasonalized_obs = pickle.load(f)
                else:
                    raise ValueError('Wrong ds_type')
        else:
            if ptype_season is None:
                ptype_season = self.configs['ptype_season']
            else:
                self.configs['ptype_season'] = ptype_season
                if verbose: p_header(f'LMRt: job.seasonalize_ds_for_psm() >>> job.configs["ptype_season"] = {ptype_season}')

            all_seasons = []
            for ptype, season in ptype_season.items():
                if isinstance(season[0], list):
                    # when ptype_season[pobj.ptype] contains multiple seasonality possibilities
                    for sn in season:
                        if sn not in all_seasons:
                            all_seasons.append(sn)
                else:
                    # when ptype_season[pobj.ptype] contains only one seasonality possibility
                    if season not in all_seasons:
                        all_seasons.append(season)

            # print(all_seasons)
            if ds_type == 'prior':
                ds = self.prior.copy()
            elif ds_type == 'obs':
                ds = self.obs.copy()
            else:
                raise ValueError('Wrong ds_type')

            seasonalized_ds = {}
            for season in all_seasons:
                if verbose: p_header(f'LMRt: job.seasonalize_ds_for_psm() >>> Seasonalizing variables from {ds_type} with season: {season}')
                season_tag = '_'.join(str(s) for s in season)
                seasonalized_ds[season_tag] = ds.seasonalize(season, inplace=False)

            if ds_type == 'prior':
                self.seasonalized_prior = seasonalized_ds
            elif ds_type == 'obs':
                self.seasonalized_obs = seasonalized_ds
            else:
                raise ValueError('Wrong ds_type')

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(seasonalized_ds, f)

        if verbose: p_success(f'LMRt: job.seasonalize_ds_for_psm() >>> job.seasonalized_{ds_type} created')


    def seasonalize_prior(self, season=None, verbose=False):
        if season is None:
            season = self.configs['prior_season']
        else:
            self.configs['prior_season'] = season
            if verbose: p_header(f'LMRt: job.seasonalize_prior() >>> job.configs["prior_season"] = {season}')

        ds = self.prior.copy()
        ds.seasonalize(self.configs['prior_season'], inplace=True)
        if verbose:
            p_hint(f'LMRt: job.seasonalize_prior() >>> seasonalized prior w/ season {season}')
            print(ds)

        self.prior = ds
        if verbose: p_success(f'LMRt: job.seasonalize_prior() >>> job.prior updated')

    def regrid_prior(self, ntrunc=None, verbose=False):
        if ntrunc is None:
            ntrunc = self.configs['prior_regrid_ntrunc']
        self.configs['prior_regrid_ntrunc'] = ntrunc

        ds = self.prior.copy()
        ds.regrid(self.configs['prior_regrid_ntrunc'], inplace=True)
        if verbose:
            p_hint('LMRt: job.regrid_prior() >>> regridded prior')
            print(ds)

        self.prior = ds
        if verbose: p_success(f'LMRt: job.regrid_prior() >>> job.prior updated')

    def load_obs(self, path_dict=None, varname_dict=None, verbose=False, anom_period=None):
        ''' Load instrumental observations fields

        Parameters
        ----------

        path_dict: dict
            a dict of environmental variables

        varname_dict: dict
            a dict to map variable names, e.g. {'tas': 'sst'} means 'tas' is named 'sst' in the input NetCDF file
        
        '''
        if path_dict is None:
            obs_path = cfg_abspath(self.cfg_path, self.configs['obs_path'])
        else:
            obs_path = cwd_abspath(path_dict)
        self.configs['obs_path'] = obs_path

        if anom_period is None:
            anom_period = self.configs['anom_period']
        else:
            self.configs['obs_anom_period'] = anom_period
            if verbose: p_header(f'LMRt: job.load_obs() >>> job.configs["anom_period"] = {anom_period}')

        vn_dict = {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon',
        }
        if 'obs_varname' in self.configs:
            vn_dict.update(self.configs['obs_varname'])
        if varname_dict is not None:
            vn_dict.update(varname_dict)
        self.configs['obs_varname'] = vn_dict

        if verbose: p_header(f'LMRt: job.load_obs() >>> loading instrumental observation fields from: {self.configs["obs_path"]}')

        ds = Dataset()
        ds.load_nc(self.configs['obs_path'], varname_dict=vn_dict, anom_period=anom_period, inplace=True)
        self.obs = ds
        if verbose: p_success(f'LMRt: job.load_obs() >>> job.obs created')


    def calibrate_psm(self, ptype_season=None,
                      seasonalized_prior_path=None, prior_loc_path=None,
                      seasonalized_obs_path=None, obs_loc_path=None,
                      calibed_psm_path=None, calib_period=None, verbose=False):

        if ptype_season is None:
            ptype_season = self.configs['ptype_season']
        else:
            self.configs['ptype_season'] = ptype_season
            if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["ptype_season"] = {ptype_season}')

        ptype_season = {k:self.configs['ptype_season'][k] for k in self.configs['ptype_psm'].keys()}

        # set paths for precalculated data
        if 'prepcalc' not in self.configs:
            self.configs['precalc'] = {}

        if seasonalized_prior_path is None:
            seasonalized_prior_path = os.path.abspath(os.path.join(self.configs['job_dirpath'], 'seasonalized_prior.pkl'))

        self.configs['precalc']['seasonalized_prior_path'] = seasonalized_prior_path
        if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["precalc"]["seasonalized_prior_path"] = {seasonalized_prior_path}')

        if seasonalized_obs_path is None:
            seasonalized_obs_path = os.path.abspath(os.path.join(self.configs['job_dirpath'], 'seasonalized_obs.pkl'))

        self.configs['precalc']['seasonalized_obs_path'] = seasonalized_obs_path
        if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["precalc"]["seasonalized_obs_path"] = {seasonalized_obs_path}')

        if prior_loc_path is None:
            prior_loc_path = os.path.abspath(os.path.join(self.configs['job_dirpath'], 'prior_loc.pkl'))

        self.configs['precalc']['prior_loc_path'] = prior_loc_path
        if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["precalc"]["prior_loc_path"] = {prior_loc_path}')

        if obs_loc_path is None:
            obs_loc_path = os.path.abspath(os.path.join(self.configs['job_dirpath'], 'obs_loc.pkl'))

        self.configs['precalc']['obs_loc_path'] = obs_loc_path
        if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["precalc"]["obs_loc_path"] = {obs_loc_path}')

        if calibed_psm_path is None:
            calibed_psm_path = os.path.abspath(os.path.join(self.configs['job_dirpath'], 'calibed_psm.pkl'))

        self.configs['precalc']['calibed_psm_path'] = calibed_psm_path
        if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["precalc"]["calibed_psm_path"] = {calibed_psm_path}')


        for ds_type, seasonalized_path, loc_path in zip(
            ['prior', 'obs'], [seasonalized_prior_path, seasonalized_obs_path], [prior_loc_path, obs_loc_path]):

            # seasonalize ds for PSM calibration
            self.seasonalize_ds_for_psm(ds_type=ds_type, ptype_season=ptype_season,
                seasonalized_ds_path=seasonalized_path, save_path=seasonalized_path, verbose=verbose)

            if ds_type == 'prior':
                ds = self.prior
                seasonalized_ds = self.seasonalized_prior
            elif ds_type == 'obs':
                ds = self.obs
                seasonalized_ds = self.seasonalized_obs

            # get modeled environmental variables at proxy locales from prior
            psm_types = set([v for k, v in self.configs['ptype_psm'].items()])
            if 'bilinear' in psm_types:
                var_names = ['tas', 'pr']
            else:
                var_names = ['tas']

            self.proxydb.find_nearest_loc(var_names, ds=ds, ds_type=ds_type, ds_loc_path=loc_path, save_path=loc_path, verbose=verbose)
            self.proxydb.get_var_from_ds(seasonalized_ds, ptype_season, ds_type=ds_type, verbose=verbose)


        # initialize PSM
        self.proxydb.init_psm(verbose=verbose)

        # calibrate PSM
        if calib_period is None:
            calib_period = self.configs['psm_calib_period']
        else:
            self.configs['psm_calib_period'] = calib_period
            if verbose: p_header(f'LMRt: job.calibrate_psm() >>> job.configs["psm_calib_period"] = {calib_period}')

        if verbose: p_header(f'LMRt: job.calibrate_psm() >>> PSM calibration period: {calib_period}')

        if calibed_psm_path is not None and os.path.exists(calibed_psm_path):
            self.proxydb.calib_psm(calib_period=calib_period, calibed_psm_path=calibed_psm_path, verbose=verbose)
        else:
            self.proxydb.calib_psm(calib_period=calib_period, save_path=calibed_psm_path, verbose=verbose)

    def forward_psm(self, verbose=False):
        self.proxydb.forward_psm(verbose=verbose)


    def gen_Xb(self, recon_vars=None, verbose=False):
        ''' Generate Xb
        '''
        if not hasattr(self, 'prior_sample_years'):
            raise ValueError('job.prior_sample_years not existing, please run job.gen_Ye() first!')

        if recon_vars is None:
            recon_vars = self.configs['recon_vars']
        else:
            self.configs['recon_vars'] = recon_vars
            if verbose: p_header(f'LMRt: job.gen_Xb() >>> job.configs["recon_vars"] = {recon_vars}')

        if type(recon_vars) is str:
            # contains only one variable
            recon_vars = [recon_vars]

        vn_1st = recon_vars[0]
        self.prior_sample_idx = [list(self.prior.fields[vn_1st].time).index(yr) for yr in self.prior_sample_years]
        if verbose: p_success(f'LMRt: job.gen_Xb() >>> job.prior_sample_idx created')
        nens = np.size(self.prior_sample_years)

        Xb_var_irow = {}  # index of rows in Xb to store the specific var
        loc = 0
        for vn in recon_vars:
            nt, nlat, nlon = np.shape(self.prior.fields[vn].value)
            lats, lons = self.prior.fields[vn].lat, self.prior.fields[vn].lon
            lon2d, lat2d = np.meshgrid(lons, lats)
            fd_coords = np.ndarray((nlat*nlon, 2))
            fd_coords[:, 0] = lat2d.flatten()
            fd_coords[:, 1] = lon2d.flatten()
            fd = self.prior.fields[vn].value[self.prior_sample_idx]
            fd = np.moveaxis(fd, 0, -1)
            fd_flat = fd.reshape((nlat*nlon, nens))
            if vn == vn_1st:
                Xb = fd_flat
                Xb_coords = fd_coords
            else:
                Xb = np.concatenate((Xb, fd_flat), axis=0)
                Xb_coords = np.concatenate((Xb_coords, fd_coords), axis=0)
            Xb_var_irow[vn] = [loc, loc+nlat*nlon-1]
            loc += nlat*nlon

        self.Xb = Xb
        self.Xb_coords = Xb_coords
        self.Xb_var_irow = Xb_var_irow
        if verbose:
            p_success(f'LMRt: job.gen_Xb() >>> job.Xb created')
            p_success(f'LMRt: job.gen_Xb() >>> job.Xb_coords created')
            p_success(f'LMRt: job.gen_Xb() >>> job.Xb_var_irow created')


    def gen_Ye(self, proxy_frac=None, nens=None, verbose=False, seed=0):
        ''' Generate Ye
        '''

        if proxy_frac is None:
            proxy_frac = self.configs['proxy_frac']
        else:
            self.configs['proxy_frac'] = proxy_frac
            if verbose: p_header(f'LMRt: job.gen_Ye() >>> job.configs["proxy_frac"] = {proxy_frac}')

        if nens is None:
            nens = self.configs['recon_nens']
        else:
            self.configs['recon_nens'] = nens
            if verbose: p_header(f'LMRt: job.gen_Xb() >>> job.configs["recon_nens"] = {nens}')

        self.proxydb.split(proxy_frac, verbose=verbose, seed=seed)

        vn_1st = list(self.prior.fields.keys())[0]
        time = self.prior.fields[vn_1st].time
        Ye_assim_df = pd.DataFrame(index=time)
        Ye_eval_df = pd.DataFrame(index=time)

        Ye_assim_lat = []
        Ye_assim_lon = []
        Ye_eval_lat = []
        Ye_eval_lon = []
        Ye_assim_coords = np.ndarray((self.proxydb.assim.nrec, 2))
        Ye_eval_coords = np.ndarray((self.proxydb.eval.nrec, 2))

        for pid, pobj in self.proxydb.assim.records.items():
            series = pd.Series(index=pobj.ye_time, data=pobj.ye_value)
            Ye_assim_df[pid] = series
            Ye_assim_lat.append(pobj.lat)
            Ye_assim_lon.append(pobj.lon)

        Ye_assim_df.dropna(inplace=True)
        Ye_assim_coords[:, 0] = Ye_assim_lat
        Ye_assim_coords[:, 1] = Ye_assim_lon

        for pid, pobj in self.proxydb.eval.records.items():
            series = pd.Series(index=pobj.ye_time, data=pobj.ye_value)
            Ye_eval_df[pid] = series
            Ye_eval_lat.append(pobj.lat)
            Ye_eval_lon.append(pobj.lon)

        Ye_eval_df.dropna(inplace=True)
        Ye_eval_coords[:, 0] = Ye_eval_lat
        Ye_eval_coords[:, 1] = Ye_eval_lon

        Ye_df = pd.concat([Ye_assim_df, Ye_eval_df], axis=1).dropna()
        self.Ye_df = Ye_df
        nt = len(Ye_df)

        self.Ye_assim_df = Ye_assim_df
        self.Ye_eval_df = Ye_eval_df

        random.seed(seed)
        sample_idx = random.sample(list(range(nt)), nens)
        self.prior_sample_years = Ye_df.index[sample_idx].values
        if verbose:
            p_success(f'LMRt: job.gen_Ye() >>> job.prior_sample_years created')

        # use self.prior_sample_idx for sampling
        self.Ye_assim = np.array(Ye_assim_df)[sample_idx].T
        self.Ye_eval = np.array(Ye_eval_df)[sample_idx].T
        self.Ye_assim_coords = Ye_assim_coords
        self.Ye_eval_coords = Ye_eval_coords

        if verbose:
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_df created')
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_assim_df created')
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_eval_df created')
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_assim created')
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_eval created')
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_assim_coords created')
            p_success(f'LMRt: job.gen_Ye() >>> job.Ye_eval_coords created')

    def update_yr(self, target_yr, Xb_aug, Xb_aug_coords, recon_loc_rad, recon_timescale=1, verbose=False, debug=False):
        start_yr = target_yr - recon_timescale/2
        end_yr = target_yr + recon_timescale/2
        Xb = np.copy(Xb_aug)

        i = 0
        for pid, pobj in self.proxydb.assim.records.items():
            mask = (pobj.time >= start_yr) & (pobj.time <= end_yr)
            nYobs = np.sum(mask)
            if nYobs == 0:
                i += 1
                continue  # skip to next proxy record
            
            Yobs = pobj.value[mask].mean()
            loc = cov_localization(recon_loc_rad, pobj, Xb_aug_coords)
            Ye = Xb[i - (self.proxydb.assim.nrec+self.proxydb.eval.nrec)]
            ob_err = pobj.R / nYobs
            Xa = enkf_update_array(Xb, Yobs, Ye, ob_err, loc=loc, debug=debug)

            if debug:
                Xb_mean = Xb[:-(self.proxydb.assim.nrec+self.proxydb.eval.nrec)].mean()
                Xa_mean = Xa[:-(self.proxydb.assim.nrec+self.proxydb.eval.nrec)].mean()
                innov = Yobs - Ye.mean()
                if np.abs(innov / Yobs) > 1:
                    print(pid, i - (self.proxydb.assim.nrec+self.proxydb.eval.nrec))
                    print(f'\tXb_mean: {Xb_mean:.2f}, Xa_mean: {Xa_mean:.2f}')
                    print(f'\tInnovation: {innov:.2f}, ob_err: {ob_err:.2f}, Yobs: {Yobs:.2f}, Ye_mean: {Ye.mean():.2f}')
            Xbvar = Xb.var(axis=1, ddof=1)
            Xavar = Xa.var(axis=1, ddof=1)
            vardiff = Xavar - Xbvar
            if (not np.isfinite(np.min(vardiff))) or (not np.isfinite(np.max(vardiff))):
                raise ValueError('Reconstruction has blown-up. Exiting!')

            if debug: print('min/max change in variance: ('+str(np.min(vardiff))+','+str(np.max(vardiff))+')')
            i += 1

            Xb = Xa

        return Xb

    def run_da(self, recon_period=None, recon_loc_rad=None, recon_timescale=None, verbose=False, debug=False):
        if recon_period is None:
            recon_period = self.configs['recon_period']
        else:
            self.configs['recon_period'] = recon_period
            if verbose: p_header(f'LMRt: job.run_da() >>> job.configs["recon_period"] = {recon_period}')

        if recon_timescale is None:
            recon_timescale = self.configs['recon_timescale']
        else:
            self.configs['recon_timescale'] = recon_timescale
            if verbose: p_header(f'LMRt: job.run_da() >>> job.configs["recon_timescale"] = {recon_timescale}')

        if recon_loc_rad is None:
            recon_loc_rad = self.configs['recon_loc_rad']
        else:
            self.configs['recon_loc_rad'] = recon_loc_rad
            if verbose: p_header(f'LMRt: job.run_da() >>> job.configs["recon_loc_rad"] = {recon_loc_rad}')

        recon_yrs = np.arange(recon_period[0], recon_period[-1]+1)
        Xb_aug = np.append(self.Xb, self.Ye_assim, axis=0)
        Xb_aug = np.append(Xb_aug, self.Ye_eval, axis=0)
        Xb_aug_coords = np.append(self.Xb_coords, self.Ye_assim_coords, axis=0)
        Xb_aug_coords = np.append(Xb_aug_coords, self.Ye_eval_coords, axis=0)

        nt = np.size(recon_yrs)
        nrow, nens = np.shape(Xb_aug)

        Xa = np.ndarray((nt, nrow, nens))
        for yr_idx, target_yr in enumerate(tqdm(recon_yrs, desc='KF updating')):
            Xa[yr_idx] = self.update_yr(target_yr, Xb_aug, Xb_aug_coords, recon_loc_rad, recon_timescale, verbose=verbose, debug=debug)

        recon_fields = {}
        for vn, irow in self.Xb_var_irow.items():
            _, nlat, nlon = np.shape(self.prior.fields[vn].value)
            recon_fields[vn] = Xa[:, irow[0]:irow[-1]+1, :].reshape((nt, nlat, nlon, nens))
            recon_fields[vn] = np.moveaxis(recon_fields[vn], -1, 1)

        self.recon_fields = recon_fields
        if verbose: p_success(f'LMRt: job.run_da() >>> job.recon_fields created')

    def save_recon(self, save_path, compress_dict={'zlib': True, 'least_significant_digit': 1}, verbose=False,
                   output_geo_mean=False, target_lats=[], target_lons=[], output_full_ens=False, dtype=np.float32):
        output_dict = {}
        for vn, fd in self.recon_fields.items():
            nyr, nens, nlat, nlon = np.shape(fd)
            if output_full_ens:
                output_var = np.array(fd, dtype=dtype)
                output_dict[vn] = (('year', 'ens', 'lat', 'lon'), output_var)
            else:
                output_var = np.array(fd.mean(axis=1), dtype=dtype)
                output_dict[vn] = (('year', 'lat', 'lon'), output_var)

            lats, lons = self.prior.fields[vn].lat, self.prior.fields[vn].lon
            gm_ens = np.ndarray((nyr, nens), dtype=dtype)
            nhm_ens = np.ndarray((nyr, nens), dtype=dtype)
            shm_ens = np.ndarray((nyr, nens), dtype=dtype)
            for k in range(nens):
                gm_ens[:,k], nhm_ens[:,k], shm_ens[:,k] = global_hemispheric_means(fd[:,k,:,:], lats)

            output_dict[f'{vn}_gm_ens'] = (('year', 'ens'), gm_ens)
            output_dict[f'{vn}_nhm_ens'] = (('year', 'ens'), nhm_ens)
            output_dict[f'{vn}_shm_ens'] = (('year', 'ens'), shm_ens)

            if vn == 'tas':
                nino_ind = nino_indices(fd, lats, lons)
                nino12 = nino_ind['nino1+2']
                nino3 = nino_ind['nino3']
                nino34 = nino_ind['nino3.4']
                nino4 = nino_ind['nino4']
                wpi = nino_ind['wpi']

                nino12 = np.array(nino12, dtype=dtype)
                nino3 = np.array(nino3, dtype=dtype)
                nino34 = np.array(nino34, dtype=dtype)
                nino4 = np.array(nino4, dtype=dtype)

                output_dict['nino1+2'] = (('year', 'ens'), nino12)
                output_dict['nino3'] = (('year', 'ens'), nino3)
                output_dict['nino3.4'] = (('year', 'ens'), nino34)
                output_dict['nino4'] = (('year', 'ens'), nino4)
                output_dict['wpi'] = (('year', 'ens'), wpi)

                # calculate tripole index (TPI)
                tpi = calc_tpi(fd, lats, lons)
                tpi = np.array(tpi, dtype=dtype)
                output_dict['tpi'] = (('year', 'ens'), tpi)

            if output_geo_mean:
                geo_mean_ts = geo_mean(fd, lats, lons, target_lats, target_lons)
                output_dict['geo_mean'] = (('year', 'ens'), geo_mean_ts)

        ds = xr.Dataset(
            data_vars=output_dict,
            coords={
                'year': np.arange(self.configs['recon_period'][0], self.configs['recon_period'][1]+1),
                'ens': np.arange(nens),
                'lat': lats,
                'lon': lons,
            })

        if compress_dict is not None:
            encoding_dict = {}
            for k in output_dict.keys():
                encoding_dict[k] = compress_dict

            ds.to_netcdf(save_path, encoding=encoding_dict)
        else:
            ds.to_netcdf(save_path)

        if verbose: p_header(f'LMRt: job.save_recon() >>> Reconstructed fields saved to: {save_path}')

    def prepare(self, job_dirpath=None, proxydb_path=None, ptype_psm=None, ptype_season=None, verbose=False,
                prior_path=None, prior_varname_dict=None, prior_season=None, prior_regrid_ntrunc=None,
                obs_path=None, obs_varname_dict=None, anom_period=None,
                calib_period=None, seasonalized_prior_path=None, seasonalized_obs_path=None,
                prior_loc_path=None, obs_loc_path=None, calibed_psm_path=None, prep_savepath=None):

        if job_dirpath is None:
            job_dirpath = self.configs['job_dirpath']
        else:
            self.configs['job_dirpath'] = job_dirpath
            if verbose: p_header(f'LMRt: job.prepare() >>> job.configs["job_dirpath"] = {job_dirpath}')

        os.makedirs(job_dirpath, exist_ok=True)
        if prep_savepath is None:
            prep_savepath = os.path.join(job_dirpath, f'job.pkl')
        else:
            if 'precalc' not in self.configs:
                self.configs['precalc'] = {}

            self.configs['precalc']['prep_savepath'] = prep_savepath
            if verbose: p_header(f'LMRt: job.prepare() >>> job.configs["precalc"]["prep_savepath"] = {prep_savepath}')

        if os.path.exists(prep_savepath):
            job_prep = pd.read_pickle(prep_savepath)
            if verbose: p_header(f'LMRt: job.prepare() >>> Prepration data loaded from: {prep_savepath}')
            self.proxydb = job_prep.proxydb
            self.prior = job_prep.prior
            self.obs = job_prep.obs
            del(job_prep)
        else:
            # load & process proxy database
            self.load_proxydb(path=proxydb_path, verbose=verbose)
            self.filter_proxydb(ptype_psm=ptype_psm, verbose=verbose)
            self.seasonalize_proxydb(ptype_season=ptype_season, verbose=verbose)

            # load prior & obs
            self.load_prior(path_dict=prior_path, varname_dict=prior_varname_dict, anom_period=anom_period, verbose=verbose)
            self.load_obs(path_dict=obs_path, varname_dict=obs_varname_dict, anom_period=anom_period, verbose=verbose)

            # calibrate & forward PSM
            self.calibrate_psm(
                seasonalized_prior_path=seasonalized_prior_path,
                seasonalized_obs_path=seasonalized_obs_path,
                prior_loc_path=prior_loc_path,
                obs_loc_path=obs_loc_path,
                calibed_psm_path=calibed_psm_path,
                calib_period=calib_period,
                verbose=verbose,
            )
            self.forward_psm(verbose=verbose)

            # seasonalize & regrid prior
            del(self.seasonalized_prior)
            del(self.seasonalized_obs)
            self.seasonalize_prior(season=prior_season, verbose=verbose)
            self.regrid_prior(ntrunc=prior_regrid_ntrunc, verbose=verbose)

            # save result
            pd.to_pickle(self, prep_savepath)
            self.configs['precalc']['prep_savepath'] = prep_savepath
            if verbose:
                p_header(f'LMRt: job.prepare() >>> Prepration data saved to: {prep_savepath}')
                p_header(f'LMRt: job.prepare() >>> job.configs["precalc"]["prep_savepath"] = {prep_savepath}')

    def save(self, prep_savepath=None, verbose=False):
        if hasattr(self, 'seasonalized_prior'):
            del(self.seasonalized_prior)
        if hasattr(self, 'seasonalized_obs'):
            del(self.seasonalized_obs)

        if prep_savepath is None:
            prep_savepath = os.path.join(self.configs['job_dirpath'], f'job.pkl')

        if 'prepcalc' not in self.configs:
            self.configs['precalc'] = {}

        pd.to_pickle(self, prep_savepath)
        self.configs['precalc']['prep_savepath'] = prep_savepath
        if verbose:
            p_header(f'LMRt: job.save_job() >>> Prepration data saved to: {prep_savepath}')
            p_header(f'LMRt: job.save_job() >>> job.configs["precalc"]["prep_savepath"] = {prep_savepath}')

        

    def run(self, recon_seeds=None, recon_vars=None, recon_period=None, recon_timescale=None, recon_loc_rad=None,
            nens=None, proxy_frac=None, verbose=False, save_configs=True,
            compress_dict={'zlib': True, 'least_significant_digit': 1},
            output_geo_mean=False, target_lats=[], target_lons=[],
            output_full_ens=False, dtype=np.float32):

        job_dirpath = self.configs["job_dirpath"]

        if recon_seeds is None:
            recon_seeds = self.configs['recon_seeds']
        else:
            self.configs['recon_seeds'] = np.array(recon_seeds).tolist()
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_seeds"] = {recon_seeds}')

        if recon_vars is None:
            recon_vars = self.configs['recon_vars']
        else:
            self.configs['recon_vars'] = recon_vars
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_vars"] = {recon_vars}')

        if type(recon_vars) is str:
            # contains only one variable
            recon_vars = [recon_vars]

        if nens is None:
            nens = self.configs['recon_nens']
        else:
            self.configs['recon_nens'] = nens
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_nens"] = {nens}')

        if proxy_frac is None:
            proxy_frac = self.configs['proxy_frac']
        else:
            self.configs['proxy_frac'] = proxy_frac
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["proxy_frac"] = {proxy_frac}')

        if recon_period is None:
            recon_period = self.configs['recon_period']
        else:
            self.configs['recon_period'] = recon_period
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_period"] = {recon_period}')

        if recon_timescale is None:
            recon_timescale = self.configs['recon_timescale']
        else:
            self.configs['recon_timescale'] = recon_timescale
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_timescale"] = {recon_timescale}')

        if recon_loc_rad is None:
            recon_loc_rad = self.configs['recon_loc_rad']
        else:
            self.configs['recon_loc_rad'] = recon_loc_rad
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_loc_rad"] = {recon_loc_rad}')

        # add settings for data saving to configs
        self.configs['save_settings'] = {}
        self.configs['save_settings']['compress_dict'] = compress_dict
        self.configs['save_settings']['output_geo_mean'] = output_geo_mean
        self.configs['save_settings']['target_lats'] = target_lats
        self.configs['save_settings']['target_lons'] = target_lons
        self.configs['save_settings']['output_full_ens'] = output_full_ens
        if dtype is np.float32:
            self.configs['save_settings']['dtype'] = 32
        elif dtype is np.float64:
            self.configs['save_settings']['dtype'] = 64
        else:
            raise ValueError('Wrong dtype!')

        if verbose: p_header(f'LMRt: job.run() >>> job.configs["save_settings"] = {self.configs["save_settings"]}')

        os.makedirs(job_dirpath, exist_ok=True)
        if save_configs:
            cfg_savepath = os.path.join(job_dirpath, f'job_configs.yml')
            with open(cfg_savepath, 'w') as f:
                yaml.dump(self.configs, f)
                if verbose: p_header(f'LMRt: job.run() >>> job.configs saved to: {cfg_savepath}')

        for seed in recon_seeds:
            p_header(f'LMRt: job.run() >>> seed: {seed} | max: {recon_seeds[-1]}')
            recon_savepath = os.path.join(job_dirpath, f'job_r{seed:02d}_recon.nc')
            if os.path.exists(recon_savepath):
                p_header(f'LMRt: job.run() >>> reconstruction existed at: {recon_savepath}')
                continue
            else:
                self.gen_Ye(proxy_frac=proxy_frac, nens=nens, seed=seed)
                self.gen_Xb(recon_vars=recon_vars)
                idx_savepath = os.path.join(job_dirpath, f'job_r{seed:02d}_idx.pkl')
                pd.to_pickle([self.prior_sample_idx, self.proxydb.calibed_idx_assim, self.proxydb.calibed_idx_eval], idx_savepath)
                if verbose: p_header(f'LMRt: job.run() >>> randomized indices for prior and proxies saved to: {idx_savepath}')

                print(self.proxydb.assim)
                self.run_da(recon_period=recon_period, recon_timescale=recon_timescale, recon_loc_rad=recon_loc_rad)

                self.save_recon(recon_savepath, compress_dict=compress_dict, output_geo_mean=output_geo_mean, verbose=verbose,
                    target_lats=target_lats, target_lons=target_lons, output_full_ens=output_full_ens, dtype=dtype)

        p_header(f'LMRt: job.run() >>> DONE!')


    def run_cfg(self, cfg_path, job_dirpath=None, recon_seeds=None, verbose=False, save_configs=True):
        self.load_configs(cfg_path, verbose=verbose)

        if job_dirpath is None:
            if os.path.isabs(self.configs['job_dirpath']):
                job_dirpath = self.configs['job_dirpath']
            else:
                job_dirpath = cfg_abspath(self.cfg_path, self.configs['job_dirpath'])
        else:
            job_dirpath = cwd_abspath(job_dirpath)

        self.configs['job_dirpath'] = job_dirpath
        os.makedirs(job_dirpath, exist_ok=True)
        if verbose:
            p_header(f'LMRt: job.load_configs() >>> job.configs["job_dirpath"] = {job_dirpath}')
            p_success(f'LMRt: job.load_configs() >>> {job_dirpath} created')

        proxydb_path = cfg_abspath(cfg_path, self.configs['proxydb_path'])
        ptype_psm = self.configs['ptype_psm']
        ptype_season = self.configs['ptype_season']
        prior_path = cfg_abspath(cfg_path, self.configs['prior_path'])
        prior_varname_dict = self.configs['prior_varname']
        prior_season = self.configs['prior_season']
        prior_regrid_ntrunc = self.configs['prior_regrid_ntrunc']
        obs_path = cfg_abspath(cfg_path, self.configs['obs_path'])
        obs_varname_dict = self.configs['obs_varname']
        anom_period = self.configs['anom_period']
        psm_calib_period = self.configs['psm_calib_period']

        try:
            seasonalized_prior_path = self.configs['precalc']['seasonalized_prior_path']
            seasonalized_obs_path = self.configs['precalc']['seasonalized_obs_path']
            prior_loc_path = self.configs['precalc']['prior_loc_path']
            obs_loc_path = self.configs['precalc']['obs_loc_path']
            calibed_psm_path = self.configs['precalc']['calibed_psm_path']
            prep_savepath = self.configs['precalc']['prep_savepath']
        except:
            seasonalized_prior_path = None
            seasonalized_obs_path = None
            prior_loc_path = None
            obs_loc_path = None
            calibed_psm_path = None
            prep_savepath = None

        if recon_seeds is None:
            recon_seeds = self.configs['recon_seeds']
        else:
            self.configs['recon_seeds'] = np.array(recon_seeds).tolist()
            if verbose: p_header(f'LMRt: job.run() >>> job.configs["recon_seeds"] = {recon_seeds}')

        recon_vars = self.configs['recon_vars']
        recon_period = self.configs['recon_period']
        recon_timescale = self.configs['recon_timescale']
        recon_loc_rad = self.configs['recon_loc_rad']
        recon_nens = self.configs['recon_nens']
        proxy_frac = self.configs['proxy_frac']

        try:
            compress_dict = self.configs['save_settings']['compress_dict']
            output_geo_mean = self.configs['save_settings']['output_geo_mean']
            target_lats = self.configs['save_settings']['target_lats']
            target_lons = self.configs['save_settings']['target_lons']
            output_full_ens = self.configs['save_settings']['output_full_ens']
            dtype_int = self.configs['save_settings']['dtype']
            if dtype_int == 32:
                dtype = np.float32
            elif dtype_int == 64:
                dtype = np.float64
            else:
                raise ValueError(f'Wrong dtype in: {cfg_path}! Should be either 32 or 64.')
        except:
            compress_dict={'zlib': True, 'least_significant_digit': 1}
            output_geo_mean=False
            target_lats=[]
            target_lons=[]
            output_full_ens=False
            dtype=np.float32
        

        self.prepare(job_dirpath, prep_savepath=prep_savepath, proxydb_path=proxydb_path, ptype_psm=ptype_psm, ptype_season=ptype_season,
            prior_path=prior_path, prior_varname_dict=prior_varname_dict, prior_season=prior_season, prior_regrid_ntrunc=prior_regrid_ntrunc,
            obs_path=obs_path, obs_varname_dict=obs_varname_dict, anom_period=anom_period,
            calib_period=psm_calib_period, seasonalized_prior_path=seasonalized_prior_path, seasonalized_obs_path=seasonalized_obs_path,
            prior_loc_path=prior_loc_path, obs_loc_path=obs_loc_path, calibed_psm_path=calibed_psm_path, verbose=verbose)

        self.save(prep_savepath=prep_savepath, verbose=verbose)

        self.run(recon_seeds=recon_seeds, recon_vars=recon_vars, recon_period=recon_period, save_configs=save_configs,
            recon_timescale=recon_timescale, recon_loc_rad=recon_loc_rad, nens=recon_nens, proxy_frac=proxy_frac, verbose=verbose,
            compress_dict=compress_dict, output_geo_mean=output_geo_mean, target_lats=target_lats, target_lons=target_lons,
            output_full_ens=output_full_ens, dtype=dtype)
