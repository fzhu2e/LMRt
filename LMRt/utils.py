''' The utility library
'''
import numpy as np
import pandas as pd
from collections import namedtuple
import xarray as xr
import netCDF4
from datetime import datetime
from datetime import timedelta
import random
from spharm import Spharmt, regrid
import prysm
import os
import sys
from tqdm import tqdm
import pickle
from scipy import signal
from scipy import optimize
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import glob
from scipy.stats.mstats import mquantiles
from scipy.stats.mstats import gmean
from scipy import spatial
from scipy.special import factorial
from scipy.stats import gaussian_kde
import cftime
from pprint import pprint
from time import time as ttime
from IPython import embed
from pathos.multiprocessing import ProcessingPool as Pool
import itertools
from scipy.stats import pearsonr
from sklearn import preprocessing

from . import load_gridded_data  # original file from LMR

Proxy = namedtuple(
    'Proxy',
    ['id', 'type', 'start_yr', 'end_yr', 'lat', 'lon', 'elev', 'seasonality', 'values', 'time', 'psm_obj']
)

PSM = namedtuple('PSM', ['psm_key', 'R', 'SNR'])

Grid = namedtuple('Grid', ['lat', 'lon', 'nlat', 'nlon', 'nens'])


def setup_cfg(cfg):
    proxy_db_cfg = {
        'LMRdb': cfg.proxies.LMRdb,
    }

    for db_name, db_cfg in proxy_db_cfg.items():
        db_cfg.proxy_type_mapping = {}
        for ptype, measurements in db_cfg.proxy_assim2.items():
            # Fetch proxy type name that occurs before underscore
            type_name = ptype.split('_', 1)[0]
            for measure in measurements:
                db_cfg.proxy_type_mapping[(type_name, measure)] = ptype

    return cfg


def get_prior(filepath, datatype, cfg, anom_reference_period=(1951, 1980), verbose=False, avgInterval=None):
    read_func = {
        'CMIP5': load_gridded_data.read_gridded_data_CMIP5_model,
    }

    prior_datadir = os.path.dirname(filepath)
    prior_datafile = os.path.basename(filepath)
    if type(cfg.prior.state_variables) is not dict:
        statevars = cfg.prior.state_variables.toDict()
    else:
        statevars = cfg.prior.state_variables
    statevars_info = cfg.prior.state_variables_info.toDict()

    if avgInterval is None:
        if cfg.core.recon_timescale == 1:
            avgInterval = {'annual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
        elif cfg.core.recon_timescale > 1:
            avgInterval = {'multiyear': [cfg.core.recon_timescale]}
        else:
            print('ERROR in config.: unrecognized job.cfg.core.recon_timescale!')
            raise SystemExit()

    detrend = cfg.prior.detrend

    datadict = read_func[datatype](
        prior_datadir, prior_datafile, statevars, avgInterval,
        detrend=detrend, anom_ref=anom_reference_period,
        var_info=statevars_info,
    )

    return datadict


def populate_ensemble(datadict, cfg, seed, verbose=False):
    ''' Populate the prior ensemble from gridded model/analysis data
    '''
    state_vect_info = {}
    Nx = 0
    timedim = []
    for var in datadict.keys():
        vartype = datadict[var]['vartype']
        dct = {}
        timedim.append(len(datadict[var]['years']))
        spacecoords = datadict[var]['spacecoords']
        dim1, dim2 = spacecoords
        ndim1, ndim2 = datadict[var][dim1].shape
        ndimtot = ndim1*ndim2
        dct['pos'] = (Nx, Nx+ndimtot-1)
        dct['spacecoords'] = spacecoords
        dct['spacedims'] = (ndim1, ndim2)
        dct['vartype'] = vartype
        state_vect_info[var] = dct
        Nx += ndimtot

    if verbose:
        print('State vector information:')
        print(state_vect_info)

    if all(x == timedim[0] for x in timedim):
        ntime = timedim[0]
    else:
        raise SystemExit('ERROR im populate_ensemble: time dimension not consistent across all state variables. Exiting!')

    Xb = np.zeros((Nx, cfg.core.nens))

    random.seed(seed)
    ind_ens = random.sample(list(range(ntime)), cfg.core.nens)

    if verbose:
        print(f'shape of Xb: ({Nx} x {cfg.core.nens})')
        print('seed=', seed)
        print('sampled inds=', ind_ens)

    Xb_coords = np.empty((Nx, 2))
    Xb_coords[:, :] = np.nan

    for var in datadict.keys():
        vartype = datadict[var]['vartype']
        indstart = state_vect_info[var]['pos'][0]
        indend = state_vect_info[var]['pos'][1]

        for i in range(cfg.core.nens):
            Xb[indstart:indend+1, i] = datadict[var]['value'][ind_ens[i], :, :].flatten()

        coordname1, coordname2 = state_vect_info[var]['spacecoords']
        coord1, coord2 = datadict[var][coordname1], datadict[var][coordname2]

        if len(coord1.shape) == 1 and len(coord2.shape) == 1:
            ndim1 = coord1.shape[0]
            ndim2 = coord2.shape[0]
            X_coord1 = np.array([coord1, ]*ndim2).transpose()
            X_coord2 = np.array([coord2, ]*ndim1)
        elif len(coord1.shape) == 2 and len(coord2.shape) == 2:
            ndim1, ndim2 = coord1.shape
            X_coord1 = coord1
            X_coord2 = coord2

        Xb_coords[indstart:indend+1, 0] = X_coord1.flatten()
        Xb_coords[indstart:indend+1, 1] = X_coord2.flatten()

        if np.any(np.isnan(Xb)):
            # Returning state vector Xb as masked array
            Xb_res = np.ma.masked_invalid(Xb)
            # Set fill_value to np.nan
            np.ma.set_fill_value(Xb_res, np.nan)
        else:
            Xb_res = Xb

    return Xb_res, ind_ens, Xb_coords, state_vect_info


def regrid_prior(cfg, X, verbose=False):
    nens = cfg.core.nens
    regrid_method = cfg.prior.regrid_method
    regrid_resolution = cfg.prior.regrid_resolution

    regrid_func = {
        'spherical_harmonics': regrid_sphere,
    }

    new_state_info = {}
    Nx = 0
    for var in list(X.full_state_info.keys()):
        dct = {}

        dct['vartype'] = X.full_state_info[var]['vartype']

        # variable indices in full state vector
        ibeg_full = X.full_state_info[var]['pos'][0]
        iend_full = X.full_state_info[var]['pos'][1]
        # extract array corresponding to state variable "var"
        var_array_full = X.ens[ibeg_full:iend_full+1, :]
        # corresponding spatial coordinates
        coords_array_full = X.coords[ibeg_full:iend_full+1, :]

        nlat = X.full_state_info[var]['spacedims'][0]
        nlon = X.full_state_info[var]['spacedims'][1]

        var_array_new, lat_new, lon_new = regrid_func[regrid_method](nlat, nlon, nens, var_array_full, regrid_resolution)

        nlat_new = np.shape(lat_new)[0]
        nlon_new = np.shape(lat_new)[1]

        if verbose:
            print(('=> Full array:      ' + str(np.min(var_array_full)) + ' ' +
                   str(np.max(var_array_full)) + ' ' + str(np.mean(var_array_full)) +
                   ' ' + str(np.std(var_array_full))))
            print(('=> Truncated array: ' + str(np.min(var_array_new)) + ' ' +
                   str(np.max(var_array_new)) + ' ' + str(np.mean(var_array_new)) +
                   ' ' + str(np.std(var_array_new))))

        # corresponding indices in truncated state vector
        ibeg_new = Nx
        iend_new = Nx+(nlat_new*nlon_new)-1
        # for new state info dictionary
        dct['pos'] = (ibeg_new, iend_new)
        dct['spacecoords'] = X.full_state_info[var]['spacecoords']
        dct['spacedims'] = (nlat_new, nlon_new)
        # updated dimension
        new_dims = (nlat_new*nlon_new)

        # array with new spatial coords
        coords_array_new = np.zeros(shape=[new_dims, 2])
        coords_array_new[:, 0] = lat_new.flatten()
        coords_array_new[:, 1] = lon_new.flatten()

        # fill in new state info dictionary
        new_state_info[var] = dct

        # if 1st time in loop over state variables, create Xb_one array as copy
        # of var_array_new
        if Nx == 0:
            Xb_one = np.copy(var_array_new)
            Xb_one_coords = np.copy(coords_array_new)
        else:  # if not 1st time, append to existing array
            Xb_one = np.append(Xb_one, var_array_new, axis=0)
            Xb_one_coords = np.append(Xb_one_coords, coords_array_new, axis=0)

        # making sure Xb_one has proper mask, if it contains
        # at least one invalid value
        if np.isnan(Xb_one).any():
            Xb_one = np.ma.masked_invalid(Xb_one)
            np.ma.set_fill_value(Xb_one, np.nan)

        # updating dimension of new state vector
        Nx = Nx + new_dims

    return Xb_one, Xb_one_coords, new_state_info


def get_proxy(cfg, proxies_df_filepath, metadata_df_filepath, precalib_filesdict=None, verbose=False,
              exclude_list=None, select_box_lf=None, select_box_ur=None):

    db_proxies = pd.read_pickle(proxies_df_filepath)
    db_metadata = pd.read_pickle(metadata_df_filepath)

    proxy_db_cfg = {
        'LMRdb': cfg.proxies.LMRdb,
    }
    db_name = cfg.proxies.use_from[0]

    if cfg.proxies.target_sites:
        # if target sites are given, then just use them regardeless of other filters
        all_proxy_ids = cfg.proxies.target_sites
    else:
        all_proxy_ids = []
        for name in proxy_db_cfg[db_name].proxy_order:
            archive = name.split('_', 1)[0]
            archive_mask = db_metadata['Archive type'] == archive

            measure_mask = db_metadata['Proxy measurement'] == 0
            measure_mask &= False

            for measure in proxy_db_cfg[db_name].proxy_assim2[name]:
                measure_mask |= db_metadata['Proxy measurement'] == measure

            resolution_mask = db_metadata['Resolution (yr)'] == 0
            resolution_mask &= False
            for proxy_resolution in proxy_db_cfg[db_name].proxy_resolution:
                resolution_mask |= db_metadata['Resolution (yr)'] == proxy_resolution

            dbase_mask = db_metadata['Databases'] == 0
            dbase_mask &= False
            for proxy_database in proxy_db_cfg[db_name].database_filter:
                sub_mask = []
                for p in db_metadata['Databases']:
                    sub_mask.append(proxy_database in p)

                dbase_mask |= sub_mask

            proxies = db_metadata['Proxy ID'][archive_mask & measure_mask & resolution_mask & dbase_mask]
            proxies_list = proxies.tolist()

            all_proxy_ids += proxies_list

    picked_proxies = []
    picked_proxy_ids = []
    #  start, finish = cfg.core.recon_period

    if exclude_list is not None:
        for pid in exclude_list:
            if pid in all_proxy_ids:
                all_proxy_ids.remove(pid)

    for site in all_proxy_ids:
        site_meta = db_metadata[db_metadata['Proxy ID'] == site]
        start_yr = site_meta['Oldest (C.E.)'].iloc[0]
        end_yr = site_meta['Youngest (C.E.)'].iloc[0]
        lat = site_meta['Lat (N)'].iloc[0]
        lon = site_meta['Lon (E)'].iloc[0]
        elev = site_meta['Elev'].iloc[0]
        seasonality = site_meta['Seasonality'].iloc[0]
        site_data = db_proxies[site]
        #  values = site_data[(site_data.index >= start) & (site_data.index <= finish)]
        values = site_data[:]
        values = values[values.notnull()]
        time = values.index.values

        if len(values) == 0:
            print(site_meta)
            raise ValueError('ERROR: No obs in specified time range!')
        if proxy_db_cfg[db_name].proxy_timeseries_kind == 'anom':
            values = values - np.mean(values)

        try:
            pmeasure = site_meta['Proxy measurement'].iloc[0]
            db_type = site_meta['Archive type'].iloc[0]
            proxy_type = proxy_db_cfg[db_name].proxy_type_mapping[(db_type, pmeasure)]
        except (KeyError, ValueError) as e:
            print(f'Proxy type/measurement not found in mapping: {e}')
            raise ValueError(e)

        psm_key = proxy_db_cfg[db_name].proxy_psm_type[proxy_type]

        # check if pre-calibration files are provided

        if precalib_filesdict and psm_key in precalib_filesdict.keys():
            psm_data = pd.read_pickle(precalib_filesdict[psm_key])
            try:
                if verbose:
                    print(site, psm_key)
                psm_site_data = psm_data[(proxy_type, site)]
                psm_obj = PSM(psm_key, psm_site_data['PSMmse'], psm_site_data['SNR'])
                pobj = Proxy(site, proxy_type, start_yr, end_yr, lat, lon, elev, seasonality, values, time, psm_obj)
                picked_proxies.append(pobj)
                picked_proxy_ids.append(site)

            except KeyError:
                #  err_msg = f'Proxy in database but not found in pre-calibration file {precalib_filesdict[psm_key]}...\nSkipping: {site}'
                err_msg = f'Proxy in database but not found in pre-calibration files...\nSkipping: {site}'
                if verbose:
                    print(err_msg)
        else:
            proxy_std = np.nanstd(values)
            proxy_var = np.nanvar(values)
            SNR = proxy_db_cfg[db_name].SNR[proxy_type]
            ob_err_std = proxy_std / SNR
            ob_err_var = ob_err_std**2

            psm_obj = PSM(psm_key, ob_err_var, SNR)
            pobj = Proxy(site, proxy_type, start_yr, end_yr, lat, lon, elev, seasonality, values, time, psm_obj)
            picked_proxies.append(pobj)
            picked_proxy_ids.append(site)

            if verbose:
                print(f'\npid={os.getpid()} >>> SNR = {SNR}, proxy_var = {proxy_var:.5f}, ob_err_var = {ob_err_var:.5f}')

        if select_box_lf is not None and select_box_ur is not None:
            # pick the proxies only inside the selected box defined with the
            # (lat, lon) of the lower-left and upper-right corner
            for idx, pobj in enumerate(picked_proxies):
                lf_lat, lf_lon = select_box_lf
                ur_lat, ur_lon = select_box_ur
                lf_lon = np.mod(lf_lon, 360)
                ur_lon = np.mod(ur_lon, 360)
                p_lon = np.mod(pobj.lon, 360)
                if pobj.lat < lf_lat or pobj.lat > ur_lat or p_lon < lf_lon or p_lon > ur_lon:
                    picked_proxies.pop(idx)
                    picked_proxy_ids.pop(idx)

    return picked_proxy_ids, picked_proxies


def get_precalib_data(psm_name, precalib_filepath):
    psm_data = pd.read_pickle(precalib_filepath)

    def get_linear_precalib_data(psm_data):
        pid_list = []
        slope_list = []
        intercept_list = []
        seasonality_list = []
        for i, v in psm_data.items():
            ptype, pid = i
            slope = v['PSMslope']
            intercept = v['PSMintercept']
            seasonality = np.unique(v['Seasonality'])

            pid_list.append(pid)
            slope_list.append(slope)
            intercept_list.append(intercept)
            seasonality_list.append(seasonality)

        precalib_data_dict = {
            'pid': pid_list,
            'slope': slope_list,
            'intercept': intercept_list,
            'seasonality': seasonality_list,
        }
        return precalib_data_dict

    def get_bilinear_precalib_data(psm_data):
        pid_list = []
        slope_temperature_list = []
        slope_moisture_list = []
        intercept_list = []
        seasonality_list = []
        for i, v in psm_data.items():
            ptype, pid = i
            slope_temperature = v['PSMslope_temperature']
            slope_moisture = v['PSMslope_moisture']
            intercept = v['PSMintercept']
            seasonality = np.unique(v['Seasonality'])

            pid_list.append(pid)
            slope_temperature_list.append(slope_temperature)
            slope_moisture_list.append(slope_moisture)
            intercept_list.append(intercept)
            seasonality_list.append(seasonality)

        precalib_data_dict = {
            'pid': pid_list,
            'slope_temperature': slope_temperature_list,
            'slope_moisture': slope_moisture_list,
            'intercept': intercept_list,
            'seasonality': seasonality_list,
        }
        return precalib_data_dict

    get_precalib_data_func = {
        'linear': get_linear_precalib_data,
        'bilinear': get_bilinear_precalib_data,
    }

    return get_precalib_data_func[psm_name](psm_data)


def calc_ye_linearPSM(proxy_manager, ptypes, psm_name,
                      precalib_data, precalc_avg, nproc=4, verbose=False):
    ''' Calculate Ye with linear/bilinear PSMs
    '''
    pid_map = {}
    ye_out = []

    pid_obs = [pid[1] for pid, v in precalib_data.items()]

    var_names = {
        'linear': ['T'],
        'bilinear': ['T', 'M'],
    }

    env_var, env_time, env_lat, env_lon = {}, {}, {}, {}
    for var_name in var_names[psm_name]:
        env_var[var_name] = precalc_avg[var_name]['var_ann_dict']
        env_time[var_name] = precalc_avg[var_name]['year_ann']
        env_lat[var_name] = precalc_avg[var_name]['lat']
        env_lon[var_name] = precalc_avg[var_name]['lon']

    def func_wrapper(pobj, idx, total_n):
        print(f'pid={os.getpid()} >>> {idx+1}/{total_n}: {pobj.id}')
        if pobj.type not in ptypes:
            # PSM not available; skip
            if verbose:
                print(f'pid={os.getpid()} >>> The proxy type {pobj.type} is not in specified types: {ptypes}; skipping ...')
            return None
        else:
            # PSM available
            if pobj.id not in pid_obs:
                print(f'pid={os.getpid()} >>> No calibration data; skipping {pobj.id} ...')
                return None
            else:
                lat_model = env_lat['T']
                lon_model = env_lon['T']
                lat_ind, lon_ind = find_closest_loc(lat_model, lon_model, pobj.lat, pobj.lon)
                if verbose:
                    print(f'pid={os.getpid()} >>> Target: ({pobj.lat}, {pobj.lon}); Found: ({lat_model[lat_ind]:.2f}, {lon_model[lon_ind]:.2f})')

                if psm_name == 'linear':
                    avgMonths = precalib_data[(pobj.type, pobj.id)]['Seasonality']

                    intercept = precalib_data[(pobj.type, pobj.id)]['PSMintercept']
                    slope = precalib_data[(pobj.type, pobj.id)]['PSMslope']

                    season_tag = '_'.join(str(m) for m in avgMonths)
                    tas_ann = env_var['T'][season_tag][:, lat_ind, lon_ind]

                    pseudo_value = slope*tas_ann + intercept

                elif psm_name == 'bilinear':
                    avgMonths_T, avgMonths_M = precalib_data[(pobj.type, pobj.id)]['Seasonality']

                    intercept = precalib_data[(pobj.type, pobj.id)]['PSMintercept']
                    slope_temperature = precalib_data[(pobj.type, pobj.id)]['PSMslope_temperature']
                    slope_moisture = precalib_data[(pobj.type, pobj.id)]['PSMslope_moisture']

                    season_tag_T = '_'.join(str(m) for m in avgMonths_T)
                    season_tag_M = '_'.join(str(m) for m in avgMonths_M)
                    tas_ann = env_var['T'][season_tag_T][:, lat_ind, lon_ind]
                    pr_ann = env_var['M'][season_tag_M][:, lat_ind, lon_ind]

                    t1, t2 = env_time['T'], env_time['M']
                    overlap_yrs = np.intersect1d(t1, t2)
                    ind1 = np.searchsorted(t1, overlap_yrs)
                    ind2 = np.searchsorted(t2, overlap_yrs)

                    pseudo_value = slope_temperature*tas_ann[ind1] + slope_moisture*pr_ann[ind2] + intercept

                if verbose:
                    mean_value = np.nanmean(pseudo_value)
                    std_value = np.nanstd(pseudo_value)
                    print(f'pid={os.getpid()} >>> shape: {np.shape(pseudo_value)}')
                    print(f'pid={os.getpid()} >>> mean: {mean_value:.2f}; std: {std_value:.2f}')

                return pseudo_value

    if nproc >= 2:
        with Pool(nproc) as pool:
            nproxies = len(proxy_manager.all_proxies)
            idx = np.arange(nproxies)
            total_n = [int(n) for n in np.ones(nproxies)*nproxies]
            res = pool.map(func_wrapper, proxy_manager.all_proxies, idx, total_n)

        k = 0
        for idx, pobj in enumerate(proxy_manager.all_proxies):
            if res[idx] is not None:
                ye_out.append(res[idx])
                pid_map[pobj.id] = k
                k += 1
    else:
        total_n = len(proxy_manager.all_proxies)
        k = 0
        for idx, pobj in enumerate(proxy_manager.all_proxies):
            res = func_wrapper(pobj, idx, total_n)
            if res is not None:
                ye_out.append(res)
                pid_map[pobj.id] = k
                k += 1

    ye_out = np.array(ye_out)
    return pid_map, ye_out


def calc_ye(proxy_manager, ptypes, psm_name,
            lat_model, lon_model, time_model,
            prior_vars,
            elev_model=None,
            match_std=True, match_mean=True,
            verbose=False, **psm_params):

    # load parameters for specific PSMs from precalculated files

    # load parameters for VS-Lite
    if 'vslite_params_path' in psm_params:
        with open(psm_params['vslite_params_path'], 'rb') as f:
            res = pickle.load(f)
            pid_obs = res['pid_obs']
            T1 = res['T1']
            T2 = res['T2']
            M1 = res['M1']
            M2 = res['M2']

    # load parameters for VS-Lite
    if 'coral_species_info' in psm_params:
        with open(psm_params['coral_species_info'], 'rb') as f:
            res = pickle.load(f)
            pid_obs = res['pid_obs']
            species_obs = res['species_obs']

    pid_map = {}
    ye_out = []
    count = 0
    k = 0
    # generate pseudoproxy values
    for idx, pobj in (enumerate(tqdm(proxy_manager.all_proxies, desc='Forward modeling')) if not verbose else enumerate(proxy_manager.all_proxies)):
        if pobj.type not in ptypes:
            # PSM not available; skip
            continue
        else:
            count += 1
            if verbose:
                print(f'\nProcessing #{count} - {pobj.id} ...')

            if 'vslite_params_path' in psm_params and pobj.id in pid_obs:
                # load parameters for VS-Lite
                ind = pid_obs.index(pobj.id)
                psm_params['T1'] = T1[ind]
                psm_params['T2'] = T2[ind]
                psm_params['M1'] = M1[ind]
                psm_params['M2'] = M2[ind]
                psm_params['pid'] = pobj.id

            if 'coral_species_info' in psm_params:
                # load parameters for coral d18O
                ind = pid_obs.index(pobj.id)
                psm_params['species'] = species_obs[ind]

            res = prysm.forward(
                psm_name, pobj.lat, pobj.lon,
                lat_model, lon_model, time_model,
                prior_vars,
                elev_obs=pobj.elev, elev_model=elev_model,
                verbose=verbose, **psm_params,
            )

            ye_time = res['pseudo_time']
            ye_tmp = res['pseudo_value']

            if np.all(np.isnan(ye_tmp)):
                print(f'Fail to forward; skipping {pobj.id} ...')
                continue

            res_dict = ts_matching(ye_time, ye_tmp, pobj.time, pobj.values.values, match_std=match_std, match_mean=match_mean)
            ye_tmp = res_dict['value_target']
            if verbose:
                factor = res_dict['factor']
                bias = res_dict['bias']
                print(f'TS matching: factor={factor:.2f}, bias={bias:.2f}')

            ye_out.append(ye_tmp)
            pid_map[pobj.id] = k
            k += 1

    ye_out = np.asarray(ye_out)

    return pid_map, ye_out


def combine_yes(ye_path_dict, output_path):
    # load Ye's
    ye = {}
    for k, v in ye_path_dict.items():
        ye[k] = np.load(v, allow_pickle=True)

    pid_index_map = {}
    ye_vals = {}
    for k, v in ye_path_dict.items():
        pid_index_map[k] = ye[k]['pid_index_map'][()]
        ye_vals[k] = ye[k]['ye_vals']

    # total ye
    ye_all_list = []
    for k in ye_path_dict.keys():
        ye_all_list.append(ye_vals[k])

    ye_all = np.concatenate(ye_all_list, axis=0)

    # total pid_map
    pid_map_all = {}

    # combine all of them
    tot_count = 0
    for k in ye_path_dict.keys():
        for pid, idx in pid_index_map[k].items():
            pid_map_all[pid] = idx + tot_count

        tot_count = np.size(list(pid_map_all))

    np.savez(output_path, pid_index_map=pid_map_all, ye_vals=ye_all)
    print(f"Saving the combined Ye's to: {output_path} ...")

    return ye_all, pid_map_all


def est_vslite_params(proxy_manager, tas, pr, lat_grid, lon_grid, time_grid,
                      matlab_path=None, func_path=None,
                      restart_matlab_period=100,
                      lat_lon_idx_path=None, save_lat_lon_idx_path=None,
                      nsamp=1000, errormod=0, gparpriors='fourbet',
                      pt_ests='med', nargout=10,
                      beta_params=np.matrix([
                          [9, 5, 0, 9],
                          [3.5, 3.5, 10, 24],
                          [1.5, 2.8, 0, 0.1],
                          [1.5, 2.5, 0.1, 0.5],
                      ]),
                      seed=0, syear=1901, eyear=2001, verbose=False):
    ''' Run the VSL parameter estimatino Matlab precedure

    Args:
        pt_ests (str): 'med' or 'mle'
        nsamp (int): the number of MCMC iterations
        errmod (int): 0: white noise, 1: AR(1) noise
        gparpriors (str): 'fourbet': beta distribution, 'uniform': uniform distribution
        beta_params (matrix): the beta distribution parameters for T1, T2, M1, M2
    '''
    from pymatbridge import Matlab

    pid_obs = []
    lat_obs = []
    lon_obs = []
    elev_obs = []
    values_obs = []
    for idx, pobj in enumerate(proxy_manager.all_proxies):
        if pobj.type == 'Tree Rings_WidthPages2':
            pid_obs.append(pobj.id)
            lat_obs.append(pobj.lat)
            lon_obs.append(pobj.lon)
            elev_obs.append(pobj.elev)
            values_obs.append(pobj.values)

    lon_grid = np.mod(lon_grid, 360)  # convert to range (0, 360)

    if lat_lon_idx_path is None:
        lat_ind, lon_ind = find_closest_loc(lat_grid, lon_grid, lat_obs, lon_obs, mode='latlon', verbose=verbose)
        if save_lat_lon_idx_path:
            with open(save_lat_lon_idx_path, 'wb') as f:
                pickle.dump([lat_ind, lon_ind], f)
            if verbose:
                print(f'Saving the found lat_ind, lon_ind to: {save_lat_lon_idx_path} ...')
    else:
        with open(lat_lon_idx_path, 'rb') as f:
            lat_ind, lon_ind = pickle.load(f)

    T = tas[:, lat_ind, lon_ind]
    P = pr[:, lat_ind, lon_ind]

    if matlab_path is None:
        raise ValueError('ERROR: matlab_path must be set!')
    if func_path is None:
        root_path = os.path.dirname(__file__)
        func_path = os.path.join(root_path, 'estimate_vslite_params_v2_3.m')
        if verbose:
            print(func_path)

    mlab = Matlab(matlab_path)
    mlab.start()

    T1 = []
    T2 = []
    M1 = []
    M2 = []
    params_est = []

    for i, trw_data in enumerate(tqdm(values_obs)):
        if verbose:
            print(f'#{i+1} - Target: ({lat_obs[i]}, {lon_obs[i]}); Found: ({lat_grid[lat_ind[i]]:.2f}, {lon_grid[lon_ind[i]]:.2f});', end=' ')
        trw_year = np.asarray(trw_data.index)
        trw_value = np.asarray(trw_data.values)
        trw_year, trw_value = pick_range(trw_year, trw_value, syear, eyear)
        grid_year, grid_tas = pick_years(trw_year, time_grid, T[:, i])
        grid_year, grid_pr = pick_years(trw_year, time_grid, P[:, i])
        nyr = int(len(grid_year)/12)

        if verbose:
            print(f'{nyr} available years')
            print(f'Running estimate_vslite_params_v2_3.m ...', end=' ')

        # restart Matlab kernel
        if np.mod(i+1, restart_matlab_period) == 0:
            mlab.stop()
            mlab.start()

        add_samps = 0
        converge_flag = 1
        while (converge_flag == 1):
            start_time = ttime()
            res = mlab.run_func(
                func_path,
                grid_tas.reshape(nyr, 12).T, grid_pr.reshape(nyr, 12).T, lat_obs[i], trw_value,
                'seed', seed, 'nsamp', nsamp+add_samps, 'errormod', errormod,
                'gparpriors', gparpriors, 'fourbetparams', beta_params,
                'pt_ests', pt_ests,
                nargout=nargout,
            )

            used_time = ttime() - start_time
            if verbose:
                print(res)
                print(f'{used_time:.2f} sec')

            converge_flag = res['result'][9]
            if converge_flag == 1:
                add_samps += 1000
                print(f'Inference not converged. Re-running with nsamp={nsamp+add_samps} ...')

        if add_samps > 1000:
            mlab.stop()
            mlab.start()

        T1_tmp = res['result'][0]
        T2_tmp = res['result'][1]
        M1_tmp = res['result'][2]
        M2_tmp = res['result'][3]

        if verbose:
            print(f'T1={T1_tmp}, T2={T2_tmp}, M1={M1_tmp}, M2={M2_tmp}\n')

        T1.append(T1_tmp)
        T2.append(T2_tmp)
        M1.append(M1_tmp)
        M2.append(M2_tmp)
        params_est.append(res['result'])

    res_dict = {
        'pid_obs': pid_obs,
        'lat_obs': lat_obs,
        'lon_obs': lon_obs,
        'elev_obs': elev_obs,
        'values_obs': values_obs,
        'T1': T1,
        'T2': T2,
        'M1': M1,
        'M2': M2,
        'params_est': params_est,
    }

    return res_dict


def est_vslite_params_from_df(df_TRW, tas_env, pr_env, lat_env, lon_env, time_env, latlon_ind_dict_path=None,
                              matlab_path=None, func_path=None, tas_bias_dict_path=None,
                              restart_matlab_period=100,
                              nsamp=1000, errormod=0, gparpriors='fourbet',
                              pt_ests='med', nargout=10,
                              beta_params=np.matrix([
                                  [9, 5, 0, 9],
                                  [3.5, 3.5, 10, 24],
                                  [1.5, 2.8, 0, 0.1],
                                  [1.5, 2.5, 0.1, 0.5],
                              ]),
                              seed=0, syear=1901, eyear=2001, verbose=False):
    ''' Run the VSL parameter estimatino Matlab precedure

    Args:
        tas_env (3-D array): monthly surface air temperature [degC]
        pr_env (3-D array): monthly accumulated precipitation [mm]
        lat_env (1-D array): latitude dim of tas/pr
        lon_env (1-D array): longitude dim of tas/pr
        time_env (1-D array): temporal dim of tas/pr
        pt_ests (str): 'med' or 'mle'
        nsamp (int): the number of MCMC iterations
        errmod (int): 0: white noise, 1: AR(1) noise
        gparpriors (str): 'fourbet': beta distribution, 'uniform': uniform distribution
        beta_params (matrix): the beta distribution parameters for T1, T2, M1, M2
    '''
    from pymatbridge import Matlab

    if matlab_path is None:
        raise ValueError('ERROR: matlab_path must be set!')
    if func_path is None:
        root_path = os.path.dirname(__file__)
        func_path = os.path.join(root_path, 'estimate_vslite_params_v2_3.m')
        if verbose:
            print(func_path)

    lon_env = np.mod(lon_env, 360)

    if tas_bias_dict_path is not None:
        with open(tas_bias_dict_path, 'rb') as f:
            tas_bias_dict = pickle.load(f)
    else:
        tas_bias_dict = None

    if os.path.exists(latlon_ind_dict_path):
        with open(latlon_ind_dict_path, 'rb') as f:
            ind_dict = pickle.load(f)
    else:
        print('Search nearest locations ...')
        ind_dict = {}
        i = 0
        for idx, row in df_TRW.iterrows():
            i += 1
            print(f'#{i}', end=' ')
            lat_obs = row['geo_meanLat']
            lon_obs = row['geo_meanLon']
            lon_obs = np.mod(lon_obs, 360)
            ind_id = (lat_obs, lon_obs)
            if ind_id not in ind_dict:
                lat_ind, lon_ind = find_closest_loc(lat_env, lon_env, lat_obs, lon_obs)
                ind_dict[ind_id] = (lat_ind, lon_ind)
                print(f'Target: ({lat_obs}, {lon_obs}); Found: ({lat_env[lat_ind]:.2f}, {lon_env[lon_ind]:.2f});')

        with open(latlon_ind_dict_path, 'wb') as f:
            pickle.dump(ind_dict, f)

    T1 = {}
    T2 = {}
    M1 = {}
    M2 = {}
    params_est = {}

    # run Matlab
    mlab = Matlab(matlab_path)
    mlab.start()
    i = 0

    for idx, row in df_TRW.iterrows():
        i += 1
        p2k_id = row['paleoData_pages2kID']
        lat_obs = row['geo_meanLat']
        lon_obs = row['geo_meanLon']
        lon_obs = np.mod(lon_obs, 360)
        elev_obs = row['geo_meanElev']
        trw_obs = row['paleoData_values']
        time_obs = row['year']
        time_obs, trw_obs = clean_ts(time_obs, trw_obs)
        lat_ind, lon_ind = ind_dict[(lat_obs, lon_obs)]
        print(f'#{i} {p2k_id} - Target: ({lat_obs}, {lon_obs}); Found: ({lat_env[lat_ind]:.2f}, {lon_env[lon_ind]:.2f});')

        T = tas_env[:, lat_ind, lon_ind]
        if tas_bias_dict is not None:
            T += tas_bias_dict[p2k_id]

        P = pr_env[:, lat_ind, lon_ind]

        trw_year, trw_value = pick_range(time_obs, trw_obs, syear, eyear)
        env_year, env_tas = pick_years(trw_year, time_env, T)
        env_year, env_pr = pick_years(trw_year, time_env, P)
        nyr = int(len(env_year)/12)

        if verbose:
            print(f'{nyr} available years')
            print(f'Running estimate_vslite_params_v2_3.m ...', end=' ')

        # restart Matlab kernel
        if np.mod(i, restart_matlab_period) == 0:
            mlab.stop()
            mlab.start()

        add_samps = 0
        converge_flag = 1
        while (converge_flag == 1):
            res = mlab.run_func(
                func_path,
                env_tas.reshape(nyr, 12).T, env_pr.reshape(nyr, 12).T, lat_obs, trw_value,
                'seed', seed, 'nsamp', nsamp+add_samps, 'errormod', errormod,
                'gparpriors', gparpriors, 'fourbetparams', beta_params,
                'pt_ests', pt_ests,
                nargout=nargout,
            )

            converge_flag = res['result'][9]
            if converge_flag == 1:
                add_samps += 1000
                print(f'Inference not converged. Re-running with nsamp={nsamp+add_samps} ...')

        if add_samps > 1000:
            mlab.stop()
            mlab.start()

        T1_tmp = res['result'][0]
        T2_tmp = res['result'][1]
        M1_tmp = res['result'][2]
        M2_tmp = res['result'][3]

        if verbose:
            print(f'T1={T1_tmp}, T2={T2_tmp}, M1={M1_tmp}, M2={M2_tmp}\n')

        T1[p2k_id] = T1_tmp
        T2[p2k_id] = T2_tmp
        M1[p2k_id] = M1_tmp
        M2[p2k_id] = M2_tmp
        params_est[p2k_id] = res['result']

    res_dict = {
        'T1': T1,
        'T2': T2,
        'M1': M1,
        'M2': M2,
        'params_est': params_est,
    }

    return res_dict


def est_vslite_params_single_site(
    p2k_id,
    tas_env, pr_env, lat_env, lon_env, time_env,
    trw_obs, lat_obs, lon_obs, time_obs,
    matlab_path=None, func_path=None, tas_bias_dict_path=None,
    nsamp=1000, errormod=0, gparpriors='fourbet',
    pt_ests='med', nargout=10,
    beta_params=np.matrix([
        [9, 5, 0, 9],
        [3.5, 3.5, 10, 24],
        [1.5, 2.8, 0, 0.1],
        [1.5, 2.5, 0.1, 0.5],
    ]),
    seed=0, syear=1901, eyear=2001, verbose=False):
    ''' Run the VSL parameter estimatino Matlab precedure

    Args:
        tas_env (1-D array): monthly surface air temperature [degC]
        pr_env (1-D array): monthly accumulated precipitation [mm]
        lat_env (float): latitude dim of tas/pr
        lon_env (float): longitude dim of tas/pr
        time_env (1-D array): temporal dim of tas/pr
        pt_ests (str): 'med' or 'mle'
        nsamp (int): the number of MCMC iterations
        errmod (int): 0: white noise, 1: AR(1) noise
        gparpriors (str): 'fourbet': beta distribution, 'uniform': uniform distribution
        beta_params (matrix): the beta distribution parameters for T1, T2, M1, M2
    '''
    from pymatbridge import Matlab

    if matlab_path is None:
        raise ValueError('ERROR: matlab_path must be set!')
    if func_path is None:
        root_path = os.path.dirname(__file__)
        func_path = os.path.join(root_path, 'estimate_vslite_params_v2_3.m')
        if verbose:
            print(func_path)

    lon_env = np.mod(lon_env, 360)

    if tas_bias_dict_path is not None:
        with open(tas_bias_dict_path, 'rb') as f:
            tas_bias_dict = pickle.load(f)
    else:
        tas_bias_dict = None

    if tas_bias_dict is not None:
        tas_env += tas_bias_dict[p2k_id]

    trw_year, trw_value = pick_range(time_obs, trw_obs, syear, eyear)
    env_year, env_tas = pick_years(trw_year, time_env, tas_env)
    env_year, env_pr = pick_years(trw_year, time_env, pr_env)
    nyr = int(len(env_year)/12)

    if verbose:
        print(f'{nyr} available years')
        print(f'Running estimate_vslite_params_v2_3.m ...', end=' ')

    # run Matlab
    mlab = Matlab(matlab_path)
    mlab.start()

    add_samps = 0
    converge_flag = 1
    while (converge_flag == 1):
        res = mlab.run_func(
            func_path,
            env_tas.reshape(nyr, 12).T, env_pr.reshape(nyr, 12).T, lat_obs, trw_value,
            'seed', seed, 'nsamp', nsamp+add_samps, 'errormod', errormod,
            'gparpriors', gparpriors, 'fourbetparams', beta_params,
            'pt_ests', pt_ests,
            nargout=nargout,
        )

        converge_flag = res['result'][9]
        if converge_flag == 1:
            add_samps += 1000
            print(f'Inference not converged. Re-running with nsamp={nsamp+add_samps} ...')

    mlab.stop()

    T1 = res['result'][0]
    T2 = res['result'][1]
    M1 = res['result'][2]
    M2 = res['result'][3]
    params_est = res['result']

    if verbose:
        print(f'T1={T1}, T2={T2}, M1={M1}, M2={M2}\n')

    res_dict = {
        'T1': T1,
        'T2': T2,
        'M1': M1,
        'M2': M2,
        'params_est': params_est,
    }

    return res_dict


def est_vsl_params_bc(site, latlon_ind_dict_path,
                      tas_ref, pr_ref, lat_ref, lon_ref, time_ref,
                      p_bar=0.2,
                      matlab_path=None, func_path=None,
                      nsamp=1000, errormod=0, gparpriors='fourbet',
                      pt_ests='med', nargout=10,
                      beta_params=np.matrix([
                          [9, 5, 0, 9],
                          [3.5, 3.5, 10, 24],
                          [1.5, 2.8, 0, 0.1],
                          [1.5, 2.5, 0.1, 0.5],
                      ]), verbose=False):

    p2k_id = site['paleoData_pages2kID']
    trw_obs = site['paleoData_values']
    time_obs = site['year']
    lat_obs = site['geo_meanLat']
    lon_obs = site['geo_meanLon']
    lon_obs = np.mod(lon_obs, 360)
    time_obs, trw_obs = clean_ts(time_obs, trw_obs)

    with open(latlon_ind_dict_path, 'rb') as f:
        ind_dict = pickle.load(f)

    lat_ind, lon_ind = ind_dict[(lat_obs, lon_obs)]
    lat_env = lat_ref[lat_ind]
    lon_env = lon_ref[lon_ind]
    time_env = time_ref

    tas_env = np.copy(tas_ref[:, lat_ind, lon_ind])
    pr_env = np.copy(pr_ref[:, lat_ind, lon_ind])

    print(f'>>> ID: {p2k_id} - Target: ({lat_obs}, {lon_obs}); Found: ({lat_env:.2f}, {lon_env:.2f});')

    # loop
    T_bias = 0

    loop_flag = True
    dec_times = 0
    while loop_flag is True and dec_times <= 3:
        vsl_params = est_vslite_params_single_site(
            p2k_id, tas_env, pr_env, lat_env, lon_env, time_env,
            trw_obs, lat_obs, lon_obs, time_obs,
            matlab_path=matlab_path, func_path=func_path,
            nsamp=nsamp, errormod=errormod, gparpriors=gparpriors,
            pt_ests=pt_ests, nargout=nargout,
            beta_params=beta_params, verbose=verbose)

        T1 = vsl_params['T1']
        T2 = vsl_params['T2']

        tas_med = np.nanmedian(tas_env)
        T_within = tas_env[(tas_env>=T1) & (tas_env<=T2)]
        percentage = np.size(T_within)/np.size(tas_env)
        if percentage >= p_bar:
            # stop loop
            print(f'>>> T1: {T1:.2f}, T2: {T2:.2f}, median(tas_env): {tas_med:.2f}, percentage: {percentage:.2f}')
            loop_flag = False
        else:
            # bias
            # correction
            if tas_med > T2:
                delta_T = -1
            elif tas_med < T1:
                delta_T = 1
            else:
                # skip
                print(f'>>> T1: {T1:.2f}, T2: {T2:.2f}, median(tas_env): {tas_med:.2f}, percentage: {percentage:.2f}')
                loop_flag = False
                continue

            while percentage < p_bar:
                p_last = np.copy(percentage)
                T_within = tas_env[(tas_env>=T1) & (tas_env<=T2)]
                percentage = np.size(T_within)/np.size(tas_env)
                print(f'>>> T1: {T1:.2f}, T2: {T2:.2f}, median(tas_env): {np.nanmedian(tas_env):.2f}, percentage: {percentage:.2f}')
                if percentage < p_last:
                    dec_times += 1
                    tas_env -= delta_T
                    T_bias -= delta_T
                    print(f'>>> percentage is decreasing, break; finalize median(tas_env): {np.nanmedian(tas_env):.2f}')
                    break
                if percentage < p_bar:
                    tas_old = np.copy(tas_env)
                    tas_env += delta_T
                    T_bias += delta_T
                    print(f'>>> median(tas_env): {np.nanmedian(tas_old):.2f} -> {np.nanmedian(tas_env):.2f}')

    return vsl_params, T_bias


def calibrate_psm(
    proxy_manager, ptypes, psm_name,
    precalc_avg,
    ref_period=[1951, 1980],
    calib_period=[1850, 2015],
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
    }, nproc=4, nobs_lb=25, verbose=False):
    ''' Calibrate linear/bilinear PSMs

    Args:
        proxy_manager (namedtuple): the proxy_manager
        ptypes (list of str): the list of proxy types
        psm_name (str): 'linear' or 'bilinear'
        precalc_avg (dict): the dict that stores the seasonal-averaged environmental variables
        seasonality (dict): the seasonality information for each proxy type

    Returns:
        precalib_dict (dict): a dict that stores the calibration information
    '''

    var_names = {
        'linear': ['T'],
        'bilinear': ['T', 'M'],
    }

    # loop over proxy records
    precalib_dict = {}

    ref_var, ref_time, ref_lat, ref_lon = {}, {}, {}, {}
    for var_name in var_names[psm_name]:
        ref_var[var_name] = precalc_avg[var_name]['var_ann_dict']
        ref_time[var_name] = precalc_avg[var_name]['year_ann']
        ref_lat[var_name] = precalc_avg[var_name]['lat']
        ref_lon[var_name] = precalc_avg[var_name]['lon']

    def func_wrapper(pobj, idx, total_n):
        print(f'pid={os.getpid()} >>> {idx+1}/{total_n}: {pobj.id}')
        if pobj.type not in ptypes:
            # PSM not available; skip
            if verbose:
                print(f'\nThe proxy type {pobj.type} is not in specified types: {ptypes}. Skipping ...')
            return None
        else:
            # PSM available
            if verbose:
                print(f'\nProcessing {pobj.id}, {pobj.type} ...')

            seasons = {}
            ref_value = {}
            for var_name in var_names[psm_name]:
                # find the reference data closest to the proxy location
                lat_ind, lon_ind = find_closest_loc(ref_lat[var_name], ref_lon[var_name], pobj.lat, pobj.lon, mode='latlon')
                #  dist = get_distance(pobj.lon, pobj.lat, ref_lon[var_name], ref_lat[var_name])
                #  lat_idx, lon_idx = np.unravel_index(dist.argmin(), dist.shape)

                if verbose:
                    print(f'>>> Target: ({pobj.lat}, {pobj.lon}); Found: ({ref_lat[var_name][lat_ind]:.2f}, {ref_lon[var_name][lon_ind]:.2f})')

                # load seasons for each variable
                if pobj.type in seasonality.keys():
                    seasons[var_name] = seasonality[pobj.type][f'seasons_{var_name}']
                    meta_season_in_list = False
                    for avgMonths in seasons[var_name]:
                        if list(pobj.seasonality) == avgMonths:
                            meta_season_in_list = True
                            break

                    if meta_season_in_list is False:
                        seasons[var_name].append(list(pobj.seasonality))
                else:
                    seasons[var_name] = [list(pobj.seasonality)]

                if verbose:
                    print(f'seasons[{var_name}]:', seasons[var_name])

                ref_value[var_name] = {}
                for season, field in ref_var[var_name].items():
                    ref_value[var_name][season] = field[:, lat_ind, lon_ind]

            pobj_time_int = np.array([int(t) for t in pobj.time])
            mask = (pobj_time_int >= calib_period[0]) & (pobj_time_int <= calib_period[-1])
            # test each season combination and find the optimal one
            if psm_name == 'linear':
                var_name = var_names[psm_name][0]
                optimal_seasonality, optimal_reg = linear_regression(
                    pobj.time[mask], pobj.values.values[mask],
                    ref_time[var_name], ref_value[var_name], seasons[var_name],
                    verbose=verbose, nobs_lb=nobs_lb,
                )

                if optimal_reg is None:
                    # not enough data for regression; skip
                    return None
                else:
                    t1, y1, t2, y2 = overlap_ts(pobj.time[mask], pobj.values.values[mask], optimal_reg.resid.index, optimal_reg.resid.values)
                    std_proxy = np.nanstd(y1)
                    precalib_dict_pobj = {
                        'lat': pobj.lat,
                        'lon': pobj.lon,
                        'elev': pobj.elev,
                        'Seasonality': optimal_seasonality,
                        'NbCalPts': int(optimal_reg.nobs),
                        'PSMintercept': optimal_reg.params[0],
                        'PSMslope': optimal_reg.params[1],
                        'PSMcorrel': np.sqrt(optimal_reg.rsquared),
                        'PSMmse': np.mean(optimal_reg.resid**2),
                        'fitAIC': optimal_reg.aic,
                        'fitBIC': optimal_reg.bic,
                        'fitR2adj': optimal_reg.rsquared_adj,
                        'PSMresid': optimal_reg.resid,
                        'SNR': std_proxy / np.sqrt(np.mean(optimal_reg.resid**2)),
                        #  'linreg': optimal_reg,
                    }
                    if verbose:
                        pprint(precalib_dict_pobj)

            elif psm_name == 'bilinear':
                var_name_1 = var_names[psm_name][0]
                var_name_2 = var_names[psm_name][1]

                optimal_seasonality_1, optimal_seasonality_2, optimal_reg = bilinear_regression(
                    pobj.time[mask], pobj.values.values[mask],
                    ref_time[var_name_1], ref_value[var_name_1], seasons[var_name_1],
                    ref_time[var_name_2], ref_value[var_name_2], seasons[var_name_2],
                    verbose=verbose, nobs_lb=nobs_lb,
                )
                if optimal_reg is None:
                    # not enough data for regression; skip
                    return None
                else:
                    t1, y1, t2, y2 = overlap_ts(pobj.time[mask], pobj.values.values[mask], optimal_reg.resid.index, optimal_reg.resid.values)
                    std_proxy = np.nanstd(y1)
                    precalib_dict_pobj = {
                        'lat': pobj.lat,
                        'lon': pobj.lon,
                        'elev': pobj.elev,
                        f'Seasonality': (optimal_seasonality_1, optimal_seasonality_2),
                        'NbCalPts': int(optimal_reg.nobs),
                        'PSMintercept': optimal_reg.params[0],
                        f'PSMslope_temperature': optimal_reg.params[1],
                        f'PSMslope_moisture': optimal_reg.params[2],
                        'PSMcorrel': np.sqrt(optimal_reg.rsquared),
                        'PSMmse': np.mean(optimal_reg.resid**2),
                        'fitAIC': optimal_reg.aic,
                        'fitBIC': optimal_reg.bic,
                        'fitR2adj': optimal_reg.rsquared_adj,
                        'PSMresid': optimal_reg.resid,
                        'SNR': std_proxy / np.sqrt(np.mean(optimal_reg.resid**2)),
                        #  'linreg': optimal_reg,
                    }

                    if verbose:
                        pprint(precalib_dict_pobj)

        return precalib_dict_pobj

    if nproc >= 2:
        with Pool(nproc) as pool:
            nproxies = len(proxy_manager.all_proxies)
            idx = np.arange(nproxies)
            total_n = [int(n) for n in np.ones(nproxies)*nproxies]
            res = pool.map(func_wrapper, proxy_manager.all_proxies, idx, total_n)

        for idx, pobj in enumerate(proxy_manager.all_proxies):
            if res[idx] is not None:
                precalib_dict[(pobj.type, pobj.id)] = res[idx]
    else:
        total_n = len(proxy_manager.all_proxies)
        for idx, pobj in enumerate(proxy_manager.all_proxies):
            res = func_wrapper(pobj, idx, total_n)
            if res is not None:
                precalib_dict[(pobj.type, pobj.id)] = res

    return precalib_dict


def linear_regression(proxy_time, proxy_value, ref_time, ref_value, seasons, verbose=False, nobs_lb=25):
    metric_list = []
    reg_res_list = []
    df_list = []
    for i, avgMonths in enumerate(seasons):
        season_tag = '_'.join(str(m) for m in avgMonths)
        ref_var_avg = ref_value[season_tag]
        yr_ann = ref_time

        df = pd.DataFrame({'time': proxy_time, 'Proxy': proxy_value})
        frame = pd.DataFrame({'time': yr_ann, 'Temperature': ref_var_avg})
        df = df.merge(frame, how='outer', on='time')
        df.set_index('time', drop=True, inplace=True)
        df.sort_index(inplace=True)
        df.astype(np.float)
        try:
            reg_res = smf.ols(formula='Proxy ~ Temperature', data=df).fit()
            R2_adj = reg_res.rsquared_adj
            nobs = int(reg_res.nobs)
            if verbose:
                print(f'SeasonT: {avgMonths}, nobs: {nobs}, R2_adj: {R2_adj:4f}')
                #  print(df.to_string())
        except:
            nobs = 0

        if nobs < nobs_lb:
            # Insufficent observation/calibration overlap to calibrate psm.
            reg_res_list.append(None)
            metric_list.append(np.nan)
            df_list.append(None)
            continue
        else:
            metric_list.append(R2_adj)
            reg_res_list.append(reg_res)
            df_list.append(df)

    if np.all(np.isnan(metric_list)):
        optimal_seasonality = None
        optimal_reg = None
    else:
        indmax = np.nanargmax(metric_list)
        optimal_seasonality = seasons[indmax]
        optimal_reg = reg_res_list[indmax]

    return optimal_seasonality, optimal_reg


def bilinear_regression(proxy_time, proxy_value,
                        ref_time_1, ref_value_1, seasons_1,
                        ref_time_2, ref_value_2, seasons_2,
                        verbose=False, nobs_lb=25):

    i_idx_list = []
    j_idx_list = []
    metric_list = []
    reg_res_list = []
    df_list = []
    for i, avgMonths_1 in enumerate(seasons_1):
        for j, avgMonths_2 in enumerate(seasons_2):
            season_tag_1 = '_'.join(str(m) for m in avgMonths_1)
            season_tag_2 = '_'.join(str(m) for m in avgMonths_2)

            ref_var_avg_1 = ref_value_1[season_tag_1]
            ref_var_avg_2 = ref_value_2[season_tag_2]
            yr_ann_1 = ref_time_1
            yr_ann_2 = ref_time_2

            df = pd.DataFrame({'time': proxy_time, 'Proxy': proxy_value})
            frameT = pd.DataFrame({'time': yr_ann_1, 'Temperature': ref_var_avg_1})
            df = df.merge(frameT, how='outer', on='time')
            frameP = pd.DataFrame({'time': yr_ann_2, 'Moisture': ref_var_avg_2})
            df = df.merge(frameP, how='outer', on='time')
            df.set_index('time', drop=True, inplace=True)
            df.sort_index(inplace=True)
            df.astype(np.float)
            try:
                reg_res = smf.ols(formula='Proxy ~ Temperature + Moisture', data=df).fit()
                nobs = int(reg_res.nobs)
                R2_adj = reg_res.rsquared_adj
                if verbose:
                    print(f'SeasonT: {avgMonths_1}, SeasonP: {avgMonths_2}, nobs: {nobs}, R2_adj: {R2_adj:.4f}')
                    #  print(df.to_string())
            except:
                nobs = 0

            if nobs < nobs_lb:
                # Insufficent observation/calibration overlap to calibrate psm.
                reg_res_list.append(None)
                metric_list.append(np.nan)
                i_idx_list.append(None)
                j_idx_list.append(None)
                df_list.append(None)
                continue
            else:
                metric_list.append(R2_adj)
                reg_res_list.append(reg_res)
                i_idx_list.append(i)
                j_idx_list.append(j)
                df_list.append(df)

    if np.all(np.isnan(metric_list)):
        optimal_seasonality_1 = None
        optimal_seasonality_2 = None
        optimal_reg = None
    else:
        indmax = np.nanargmax(metric_list)
        optimal_seasonality_1 = seasons_1[i_idx_list[indmax]]
        optimal_seasonality_2 = seasons_2[j_idx_list[indmax]]
        optimal_reg = reg_res_list[indmax]

    return optimal_seasonality_1, optimal_seasonality_2, optimal_reg


def generate_proxy_ind(cfg, nsites, seed):
    nsites_assim = int(nsites * cfg.proxies.proxy_frac)

    random.seed(seed)

    ind_assim = random.sample(range(nsites), nsites_assim)
    ind_assim.sort()

    ind_eval = list(set(range(nsites)) - set(ind_assim))
    ind_eval.sort()

    return ind_assim, ind_eval


def get_ye(proxy_manager, prior_sample_idxs, ye_filesdict, proxy_set, verbose=False):
    if verbose:
        print(f'-------------------------------------------')
        print(f'Loading Ye files for proxy set: {proxy_set}')
        print(f'-------------------------------------------')

    num_samples = len(prior_sample_idxs)

    sites_proxy_objs = {
        'assim': proxy_manager.sites_assim_proxy_objs,
        'eval': proxy_manager.sites_eval_proxy_objs,
    }

    num_proxies = {
        'assim': len(proxy_manager.ind_assim),
        'eval': len(proxy_manager.ind_eval),
    }

    psm_keys = {
        'assim': list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_assim_proxy_objs])),
        'eval': list(set([pobj.psm_obj.psm_key for pobj in proxy_manager.sites_eval_proxy_objs])),
    }

    ye_all = np.zeros((num_proxies[proxy_set], num_samples))
    ye_all_coords = np.zeros((num_proxies[proxy_set], 2))

    precalc_files = {}
    for psm_key in psm_keys[proxy_set]:
        if verbose:
            print(f'Loading precalculated Ye from:\n {ye_filesdict[psm_key]}\n')

        precalc_files[psm_key] = np.load(ye_filesdict[psm_key], allow_pickle=True)

    if verbose:
        print('Now extracting proxy type-dependent Ye values...\n')

    for i, pobj in enumerate(sites_proxy_objs[proxy_set]):
        psm_key = pobj.psm_obj.psm_key
        pid_idx_map = precalc_files[psm_key]['pid_index_map'][()]
        precalc_vals = precalc_files[psm_key]['ye_vals']

        pidx = pid_idx_map[pobj.id]
        ye_all[i] = precalc_vals[pidx, prior_sample_idxs]
        ye_all_coords[i] = np.asarray([pobj.lat, pobj.lon], dtype=np.float64)

    # delete nans
    delete_site_rows = []
    for i, ye in enumerate(ye_all):
        if any(np.isnan(ye)):
            delete_site_rows.append(i)

    ye_all = np.delete(ye_all, delete_site_rows, 0)
    ye_all_coords = np.delete(ye_all_coords, delete_site_rows, 0)

    return ye_all, ye_all_coords


def get_valid_proxies(cfg, proxy_manager, target_year, Ye_assim, Ye_assim_coords, proxy_inds=None, verbose=False):
    recon_timescale = cfg.core.recon_timescale
    if verbose:
        print(f'finding proxy records for year: {target_year}')
        print(f'recon_timescale = {recon_timescale}')

    start_yr = int(target_year-recon_timescale//2)
    end_yr = int(target_year+recon_timescale//2)

    vY = []
    vR = []
    vP = []
    vT = []
    #  for proxy_idx, Y in enumerate(proxy_manager.sites_assim_proxy_objs()):
    for proxy_idx, Y in enumerate(proxy_manager.sites_assim_proxy_objs):
        # Check if we have proxy ob for current time interval
        if recon_timescale > 1:
            # exclude lower bound to not include same obs in adjacent time intervals
            Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
        else:
            # use all available proxies from config.yml
            if proxy_inds is None:
                Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                #  Yvals = Y.values[(Y.time == target_year)]
                # use only the selected proxies (e.g., randomly filtered post-config)
            else:
                if proxy_idx in proxy_inds:
                    #Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]
                    Yvals = Y.values[(Y.time == target_year)]
                else:
                    Yvals = pd.DataFrame()

        if Yvals.empty:
            #  print('empty:', proxy_idx)
            if verbose: print('no obs for this year')
            pass
        else:
            nYobs = len(Yvals)
            Yobs = Yvals.mean()
            ob_err = Y.psm_obj.R/nYobs
            #           if (target_year >=start_yr) & (target_year <= end_yr):
            vY.append(Yobs)
            vR.append(ob_err)
            vP.append(proxy_idx)
            vT.append(Y.type)

    vYe = Ye_assim[vP, :]
    vYe_coords = Ye_assim_coords[vP, :]

    return vY, vR, vP, vYe, vT, vYe_coords


def make_grid(prior):
    lat, lon = sorted(list(set(prior.coords[:, 0]))), sorted(list(set(prior.coords[:, 1])))
    nlat, nlon = np.size(lat), np.size(lon)
    nens = np.shape(prior.ens)[-1]

    g = Grid(lat, lon, nlat, nlon, nens)
    return g


def update_year_lite(target_year, cfg, Xb_one, grid, proxy_manager, ye_all, ye_all_coords,
                     Xb_one_aug, Xb_one_coords, X,
                     ibeg_tas, iend_tas,
                     proxy_inds=None, da_solver='ESRF',
                     verbose=False):

    vY, vR, vP, vYe, vT, vYe_coords = get_valid_proxies(
        cfg, proxy_manager, target_year, ye_all, ye_all_coords, proxy_inds=proxy_inds, verbose=verbose)

    if da_solver == 'ESRF':
        xam, Xap, Xa = Kalman_ESRF(
            cfg, vY, vR, vYe, Xb_one,
            proxy_manager, X, Xb_one_aug, Xb_one_coords, verbose=verbose
        )
    elif da_solver == 'optimal':
        xam, Xap, _ = Kalman_optimal(vY, vR, vYe, Xb_one, verbose=verbose)
    else:
        raise ValueError('ERROR: Wrong da_solver!!!')

    nens = grid.nens
    gmt_ens = np.zeros(nens)
    nhmt_ens = np.zeros(nens)
    shmt_ens = np.zeros(nens)

    for k in range(nens):
        xam_lalo = np.reshape(Xa[ibeg_tas:iend_tas+1, k], [grid.nlat, grid.nlon])
        gmt_ens[k], nhmt_ens[k], shmt_ens[k] = global_hemispheric_means(xam_lalo, grid.lat)

    return gmt_ens, nhmt_ens, shmt_ens


def update_year(yr_idx, target_year,
                cfg, Xb_one_aug, Xb_one_coords, X, sites_assim_proxy_objs,
                assim_proxy_count, eval_proxy_count, grid,
                ibeg, iend, verbose=False):

    recon_timescale = cfg.core.recon_timescale
    start_yr = int(target_year-recon_timescale//2)
    end_yr = int(target_year+recon_timescale//2)

    Xb = Xb_one_aug.copy()

    inflate = cfg.core.inflation_fact

    for proxy_idx, Y in enumerate(sites_assim_proxy_objs):
        try:
            if recon_timescale > 1:
                # exclude lower bound to not include same obs in adjacent time intervals
                Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
            else:
                Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]

            if Yvals.empty:
                raise KeyError()
            nYobs = len(Yvals)
            Yobs = Yvals.mean()

        except KeyError:
            continue  # skip to next loop iteration (proxy record)

        loc = cov_localization(cfg.core.loc_rad, Y, X, Xb_one_coords)

        Ye = Xb[proxy_idx - (assim_proxy_count+eval_proxy_count)]

        ob_err = Y.psm_obj.R/float(nYobs)

        if cfg.core.output_details:
            details_dict = enkf_update_array(Xb, Yobs, Ye, ob_err, loc=loc, inflate=inflate, output_details=True)
            Xa = details_dict['Xa']
            os.makedirs(cfg.core.output_details_dirpath, exist_ok=True)
            filename = f'enkf_details_{yr_idx}_{Y.id}.pkl'
            with open(os.path.join(cfg.core.output_details_dirpath, filename), 'wb') as f:
                pickle.dump(details_dict, f)
        else:
            Xa = enkf_update_array(Xb, Yobs, Ye, ob_err, loc=loc, inflate=inflate, output_details=False)

        xbvar = Xb.var(axis=1, ddof=1)
        xavar = Xa.var(axis=1, ddof=1)
        vardiff = xavar - xbvar
        if (not np.isfinite(np.min(vardiff))) or (not np.isfinite(np.max(vardiff))):
            print('ERROR: Reconstruction has blown-up. Exiting!')
            print(f'Y.id={Y.id}')
            print(f'np.max(Xb)={np.max(Xb)}')
            print(f'Y.psm_obj.psm_key={Y.psm_obj.psm_key}')
            print(f'Y.psm_obj.R={Y.psm_obj.R}')
            print(f'np.min(xbvar)={np.min(xbvar)}, np.max(xbvar)={np.max(xbvar)}')
            print(f'np.min(xavar)={np.min(xavar)}, np.max(xavar)={np.max(xavar)}')
            raise SystemExit(1)

        if verbose:
            print('min/max change in variance: ('+str(np.min(vardiff))+','+str(np.max(vardiff))+')')

        Xb = Xa

    xam_lalo = {}
    for name in ibeg.keys():
        xam_lalo[name] = Xb[ibeg[name]:iend[name]+1, :].T.reshape(grid.nens, grid.nlat, grid.nlon)

    return xam_lalo


def update_year_optimal(yr_idx, target_year,
                        cfg, Xb_one_aug, Xb_one_coords, X, sites_assim_proxy_objs,
                        assim_proxy_count, eval_proxy_count, grid,
                        ibeg, iend, verbose=False):

    recon_timescale = cfg.core.recon_timescale
    start_yr = int(target_year-recon_timescale//2)
    end_yr = int(target_year+recon_timescale//2)

    Xb = Xb_one_aug.copy()

    v_Yobs = []
    v_loc = []
    v_Ye = []
    v_ob_err = []
    for proxy_idx, Y in enumerate(sites_assim_proxy_objs):
        try:
            if recon_timescale > 1:
                # exclude lower bound to not include same obs in adjacent time intervals
                Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)]
            else:
                Yvals = Y.values[(Y.values.index >= start_yr) & (Y.values.index <= end_yr)]

            if Yvals.empty:
                raise KeyError()
            nYobs = len(Yvals)
            v_Yobs.append(Yvals.mean())

        except KeyError:
            continue  # skip to next loop iteration (proxy record)

        v_loc.append(cov_localization(cfg.core.loc_rad, Y, X, Xb_one_coords))
        v_Ye.append(Xb[proxy_idx - (assim_proxy_count+eval_proxy_count)])
        v_ob_err.append(Y.psm_obj.R/float(nYobs))

    if len(v_Ye) == 0:
        Xa = Xb
    else:
        v_Yobs = np.asarray(v_Yobs)
        v_loc = np.asarray(v_loc)
        v_Ye = np.asarray(v_Ye)
        v_ob_err = np.asarray(v_ob_err)

        # EnKF update
        nens = np.shape(Xb)[-1]
        nobs = len(v_Ye)

        xbm = np.mean(Xb, axis=1)
        Xbp = np.subtract(Xb, xbm[:, None])
        mye = np.mean(v_Ye, axis=1)
        ye = np.subtract(v_Ye, mye[:, None])

        Risr = np.diag(1/np.sqrt(v_ob_err))
        Htp = np.dot(Risr, ye) / np.sqrt(nens-1)
        Htm = np.dot(Risr, mye[:, None])
        Yt = np.dot(Risr, v_Yobs)
        # numpy svd quirk: V is actually V^T
        U, s, V = np.linalg.svd(Htp, full_matrices=True)
        nsvs = len(s) - 1
        innov = np.dot(U.T, Yt-np.squeeze(Htm))
        # Kalman gain
        Kpre = s[0:nsvs]/(s[0:nsvs]*s[0:nsvs] + 1)
        K = np.zeros([nens, nobs])
        np.fill_diagonal(K, Kpre)
        # ensemble-mean analysis increment in transformed space
        xhatinc = np.dot(K, innov)
        # ensemble-mean analysis increment in the transformed ensemble space
        xtinc = np.dot(V.T, xhatinc)/np.sqrt(nens-1)

        # ensemble-mean analysis increment in the original space
        xinc = np.dot(Xbp, xtinc)
        # ensemble mean analysis in the original space
        xam = xbm + xinc

        # transform the ensemble perturbations
        lam = np.zeros([nobs, nens])
        np.fill_diagonal(lam, s[0:nsvs])
        tmp = np.linalg.inv(np.dot(lam, lam.T) + np.identity(nobs))
        sigsq = np.identity(nens) - np.dot(np.dot(lam.T, tmp), lam)
        sig = np.sqrt(sigsq)
        T = np.dot(V.T, sig)
        Xap = np.dot(Xbp, T)
        Xa = Xap + xam[:, None]

    xam_lalo = {}
    for name in ibeg.keys():
        xam_lalo[name] = Xa[ibeg[name]:iend[name]+1, :].T.reshape(grid.nens, grid.nlat, grid.nlon)

    return xam_lalo


def regrid_sphere(nlat, nlon, Nens, X, ntrunc):
    """ Truncate lat,lon grid to another resolution in spherical harmonic space. Triangular truncation

    Inputs:
    nlat            : number of latitudes
    nlon            : number of longitudes
    Nens            : number of ensemble members
    X               : data array of shape (nlat*nlon,Nens)
    ntrunc          : triangular truncation (e.g., use 42 for T42)

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)

    Originator: Greg Hakim
                University of Washington
                May 2015
    """
    # create the spectral object on the original grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

    # create new lat,lon grid arrays
    # Note: AP - According to github.com/jswhit/pyspharm documentation the
    #  latitudes will not include the equator or poles when nlats is even.
    if nlat_new % 2 == 0:
        include_poles = False
    else:
        include_poles = True

    lat_new, lon_new, _, _ = generate_latlon(nlat_new, nlon_new,
                                             include_endpts=include_poles)

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    for k in range(Nens):
        X_lalo = np.reshape(X[:,k],(nlat,nlon))
        Xbtrunc = regrid(specob_lmr, specob_new, X_lalo, ntrunc=nlat_new-1, smooth=None)
        vectmp = Xbtrunc.flatten()
        X_new[:,k] = vectmp

    return X_new,lat_new,lon_new


def regrid_sphere_field(nlat, nlon, var_field, ntrunc):
    ''' Regrid a field
    '''
    # create the spectral object on the original grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

    # create new lat,lon grid arrays
    # Note: AP - According to github.com/jswhit/pyspharm documentation the
    #  latitudes will not include the equator or poles when nlats is even.
    if nlat_new % 2 == 0:
        include_poles = False
    else:
        include_poles = True

    lat_new, lon_new, _, _ = generate_latlon(nlat_new, nlon_new,
                                             include_endpts=include_poles)

    # transform each ensemble member, one at a time
    var_regridded = []
    for i in range(np.shape(var_field)[0]):
        var_regridded.append(
            regrid(specob_lmr, specob_new, var_field, ntrunc=nlat_new-1, smooth=None)
        )

    return var_regridded, lat_new, lon_new


def find_closest_loc(lat, lon, target_lat, target_lon, mode=None, verbose=False):
    ''' Find the closet model sites (lat, lon) based on the given target (lat, lon) list

    Args:
        lat, lon (array): the model latitude and longitude arrays
        target_lat, target_lon (array): the target latitude and longitude arrays
        mode (str):
        + latlon: the model lat/lon is a 1-D array
        + mesh: the model lat/lon is a 2-D array

    Returns:
        lat_ind, lon_ind (array): the indices of the found closest model sites

    '''
    if mode is None:
        if len(np.shape(lat)) == 1:
            mode = 'latlon'
        elif len(np.shape(lat)) == 2:
            mode = 'mesh'
        else:
            raise ValueError('ERROR: The shape of the lat/lon cannot be processed !!!')

    if mode is 'latlon':
        # model locations
        mesh = np.meshgrid(lon, lat)

        list_of_grids = list(zip(*(grid.flat for grid in mesh)))
        model_lon, model_lat = zip(*list_of_grids)

    elif mode is 'mesh':
        model_lat = lat.flatten()
        model_lon = lon.flatten()

    elif mode is 'list':
        model_lat = lat
        model_lon = lon

    model_locations = []

    for m_lat, m_lon in zip(model_lat, model_lon):
        model_locations.append((m_lat, m_lon))

    # target locations
    if np.size(target_lat) > 1:
        #  target_locations_dup = list(zip(target_lat, target_lon))
        #  target_locations = list(set(target_locations_dup))  # remove duplicated locations
        target_locations = list(zip(target_lat, target_lon))
        n_loc = np.shape(target_locations)[0]
    else:
        target_locations = [(target_lat, target_lon)]
        n_loc = 1

    lat_ind = np.zeros(n_loc, dtype=int)
    lon_ind = np.zeros(n_loc, dtype=int)

    # get the closest grid
    for i, target_loc in (enumerate(tqdm(target_locations)) if verbose else enumerate(target_locations)):
        X = target_loc
        Y = model_locations
        distance, index = spatial.KDTree(Y).query(X)
        closest = Y[index]
        nlon = np.shape(lon)[-1]

        if mode == 'list':
            lat_ind[i] = index % nlon
        else:
            lat_ind[i] = index // nlon
        lon_ind[i] = index % nlon

        #  if np.size(target_lat) > 1:
            #  df_ind[i] = target_locations_dup.index(target_loc)

    if np.size(target_lat) > 1:
        #  return lat_ind, lon_ind, df_ind
        return lat_ind, lon_ind
    else:
        return lat_ind[0], lon_ind[0]


def search_nearest_not_nan(field, lat_ind, lon_ind, distance=3):
    fix_sum = []
    lat_fix_list = []
    lon_fix_list = []
    for lat_fix, lon_fix in itertools.product(np.arange(-distance, distance+1), np.arange(-distance, distance+1)):
        lat_fix_list.append(lat_fix)
        lon_fix_list.append(lon_fix)
        fix_sum.append(np.abs(lat_fix)+np.abs(lon_fix))

    lat_fix_list = np.asarray(lat_fix_list)
    lon_fix_list = np.asarray(lon_fix_list)

    sort_i = np.argsort(fix_sum)

    for lat_fix, lon_fix in zip(lat_fix_list[sort_i], lon_fix_list[sort_i]):
        target = np.asarray(field[:, lat_ind+lat_fix, lon_ind+lon_fix])
        if np.all(np.isnan(target)):
            continue
        else:
            print(f'Found not nan with (lat_fix, lon_fix): ({lat_fix}, {lon_fix})')
            return target, lat_fix, lon_fix

    print(f'Fail to find value not nan!')
    return np.nan, np.nan, np.nan


def generate_latlon(nlats, nlons, include_endpts=False,
                    lat_bnd=(-90,90), lon_bnd=(0, 360)):
    """ Generate regularly spaced latitude and longitude arrays where each point
    is the center of the respective grid cell.

    Parameters
    ----------
    nlats: int
        Number of latitude points
    nlons: int
        Number of longitude points
    lat_bnd: tuple(float), optional
        Bounding latitudes for gridcell edges (not centers).  Accepts values
        in range of [-90, 90].
    lon_bnd: tuple(float), optional
        Bounding longitudes for gridcell edges (not centers).  Accepts values
        in range of [-180, 360].
    include_endpts: bool
        Include the poles in the latitude array.

    Returns
    -------
    lat_center_2d:
        Array of central latitide points (nlat x nlon)
    lon_center_2d:
        Array of central longitude points (nlat x nlon)
    lat_corner:
        Array of latitude boundaries for all grid cells (nlat+1)
    lon_corner:
        Array of longitude boundaries for all grid cells (nlon+1)
    """

    if len(lat_bnd) != 2 or len(lon_bnd) != 2:
        raise ValueError('Bound tuples must be of length 2')
    if np.any(np.diff(lat_bnd) < 0) or np.any(np.diff(lon_bnd) < 0):
        raise ValueError('Lower bounds must be less than upper bounds.')
    if np.any(abs(np.array(lat_bnd)) > 90):
        raise ValueError('Latitude bounds must be between -90 and 90')
    if np.any(abs(np.diff(lon_bnd)) > 360):
        raise ValueError('Longitude bound difference must not exceed 360')
    if np.any(np.array(lon_bnd) < -180) or np.any(np.array(lon_bnd) > 360):
        raise ValueError('Longitude bounds must be between -180 and 360')

    lon_center = np.linspace(lon_bnd[0], lon_bnd[1], nlons, endpoint=False)

    if include_endpts:
        lat_center = np.linspace(lat_bnd[0], lat_bnd[1], nlats)
    else:
        tmp = np.linspace(lat_bnd[0], lat_bnd[1], nlats+1)
        lat_center = (tmp[:-1] + tmp[1:]) / 2.

    lon_center_2d, lat_center_2d = np.meshgrid(lon_center, lat_center)
    lat_corner, lon_corner = calculate_latlon_bnds(lat_center, lon_center)

    return lat_center_2d, lon_center_2d, lat_corner, lon_corner


def calculate_latlon_bnds(lats, lons):
    """ Calculate the bounds for regularly gridded lats and lons.

    Parameters
    ----------
    lats: ndarray
        Regularly spaced latitudes.  Must be 1-dimensional and monotonically
        increase with index.
    lons:  ndarray
        Regularly spaced longitudes.  Must be 1-dimensional and monotonically
        increase with index.

    Returns
    -------
    lat_bnds:
        Array of latitude boundaries for each input latitude of length
        len(lats)+1.
    lon_bnds:
        Array of longitude boundaries for each input longitude of length
        len(lons)+1.

    """
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError('Expected 1D-array for input lats and lons.')
    if np.any(np.diff(lats) < 0) or np.any(np.diff(lons) < 0):
        raise ValueError('Expected monotonic value increase with index for '
                         'input latitudes and longitudes')

    # Note: assumes lats are monotonic increase with index
    dlat = abs(lats[1] - lats[0]) / 2.
    dlon = abs(lons[1] - lons[0]) / 2.

    # Check that inputs are regularly spaced
    lat_space = np.diff(lats)
    lon_space = np.diff(lons)

    lat_bnds = np.zeros(len(lats)+1)
    lon_bnds = np.zeros(len(lons)+1)

    lat_bnds[1:-1] = lats[:-1] + lat_space/2.
    lat_bnds[0] = lats[0] - lat_space[0]/2.
    lat_bnds[-1] = lats[-1] + lat_space[-1]/2.

    if lat_bnds[0] < -90:
        lat_bnds[0] = -90.
    if lat_bnds[-1] > 90:
        lat_bnds[-1] = 90.

    lon_bnds[1:-1] = lons[:-1] + lon_space/2.
    lon_bnds[0] = lons[0] - lon_space[0]/2.
    lon_bnds[-1] = lons[-1] + lon_space[-1]/2.

    return lat_bnds, lon_bnds


def cov_localization(locRad, Y, X, X_coords):
    """ Originator: R. Tardif, Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
        locRad : Localization radius (distance in km beyond which cov are forced to zero)
             Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
             X : Prior object, needed to get state vector info.
      X_coords : Array containing geographic location information of state vector elements

     Output:
        covLoc : Localization vector (weights) applied to ensemble covariance estimates.
                 Dims = (Nx x 1), with Nx the dimension of the state vector.

     Note: Uses the Gaspari-Cohn localization function.

    """

    # declare the localization array, filled with ones to start with (as in no localization)
    stateVectDim, nbdimcoord = X_coords.shape
    covLoc = np.ones(shape=[stateVectDim],dtype=np.float64)

    # Mask to identify elements of state vector that are "localizeable"
    # i.e. fields with (lat,lon)
    localizeable = covLoc == 1. # Initialize as True

    for var in X.trunc_state_info.keys():
        [var_state_pos_begin,var_state_pos_end] =  X.trunc_state_info[var]['pos']
        # if variable is not a field with lats & lons, tag localizeable as False
        if X.trunc_state_info[var]['spacecoords'] != ('lat', 'lon'):
            localizeable[var_state_pos_begin:var_state_pos_end+1] = False

    # array of distances between state vector elements & proxy site
    # initialized as zeros: this is important!
    dists = np.zeros(shape=[stateVectDim])

    # geographic location of proxy site
    site_lat = Y.lat
    site_lon = Y.lon
    # geographic locations of elements of state vector
    X_lon = X_coords[:,1]
    X_lat = X_coords[:,0]

    # calculate distances for elements tagged as "localizeable".
    dists[localizeable] = np.array(haversine(site_lon, site_lat,
                                                       X_lon[localizeable],
                                                       X_lat[localizeable]),dtype=np.float64)

    # those not "localizeable" are assigned with a disdtance of "nan"
    # so these elements will not be included in the indexing
    # according to distances (see below)
    dists[~localizeable] = np.nan

    # Some transformation to variables used in calculating localization weights
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    # values for distances very near the localization radius
    # TODO: revisit calculations to minimize round-off errors
    covLoc[covLoc < 0.0] = 0.0

    return covLoc


def haversine(lon1, lat1, lon2, lat2):
    """ Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km


def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, inflate=None, output_details=False):
    """ Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    Revisions:

    1 September 2017:
                    - changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1)
                    for an unbiased calculation of the variance.
                    (G. Hakim - U. Washington)

    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens)
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens x 1)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb

    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)

    # Option to inflate the covariances by a certain factor
    #if inflate is not None:
    #    kcov = inflate * kcov # This implementation is not correct. To be revised later.

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc)

    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    # Return the full state
    if output_details:
        details_dict = {
            'xbm': xbm,
            'xam': xam,
            'Xbp': Xbp,
            'Xap': Xap,
            'Xa': Xa,
            'kmat': kmat,
            'kcov': kcov,
            'kdenom': kcov,
            'varye': varye,
            'beta': beta,
            'ob_err': ob_err,
            'innov': innov,
        }
        return details_dict
    else:
        return Xa


def Kalman_ESRF(cfg, vY, vR, vYe, Xb_in,
                proxy_manager, X, Xb_one_aug, Xb_one_coords, verbose=False):

    if verbose:
        print('Ensemble square root filter...')

    begin_time = time()

    # number of state variables
    nx = Xb_in.shape[0]

    # augmented state vector with Ye appended
    # Xb = np.append(Xb_in, vYe, axis=0)
    Xb = Xb_one_aug

    inflate = cfg.core.inflation_fact

    # need to add code block to compute localization factor
    nobs = len(vY)
    if verbose: print('appended state...')
    for k in range(nobs):
        #if np.mod(k,100)==0: print k
        obvalue = vY[k]
        ob_err = vR[k]
        Ye = Xb[nx+k,:]
        Y = proxy_manager.sites_assim_proxy_objs[k]
        loc = cov_localization(cfg.core.loc_rad, Y, X, Xb_one_coords)
        Xa = enkf_update_array(Xb, obvalue, Ye, ob_err, loc=loc, inflate=inflate)
        Xb = Xa

    # ensemble mean and perturbations
    Xap = Xa[0:nx,:] - Xa[0:nx,:].mean(axis=1,keepdims=True)
    xam = Xa[0:nx,:].mean(axis=1)

    elapsed_time = time() - begin_time
    if verbose:
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    return xam, Xap, Xa


def Kalman_optimal(Y, vR, Ye, Xb, loc_rad=None, nsvs=None, transform_only=False, verbose=False):
    ''' Kalman Filter

    Args:
        Y: observation vector (p x 1)
        vR: observation error variance vector (p x 1)
        Ye: prior-estimated observation vector (p x n)
        Xbp: prior ensemble perturbation matrix (m x n)

    Originator:

        Greg Hakim
        University of Washington
        26 February 2018

    Modifications:
    11 April 2018: Fixed bug in handling singular value matrix (rectangular, not square)
    '''

    if verbose:
        print('\n all-at-once solve...\n')

    begin_time = time()

    nobs = Ye.shape[0]
    nens = Ye.shape[1]
    ndof = np.min([nobs, nens])

    if verbose:
        print('number of obs: '+str(nobs))
        print('number of ensemble members: '+str(nens))

    # ensemble prior mean and perturbations
    xbm = Xb.mean(axis=1)
    #Xbp = Xb - Xb.mean(axis=1,keepdims=True)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    #  R = np.diag(vR)
    Risr = np.diag(1./np.sqrt(vR))
    # (suffix key: m=ensemble mean, p=perturbation from ensemble mean; f=full value)
    # keepdims=True needed for broadcasting to work; (p,1) shape rather than (p,)
    Yem = Ye.mean(axis=1, keepdims=True)
    Yep = Ye - Yem
    Htp = np.dot(Risr, Yep)/np.sqrt(nens-1)
    Htm = np.dot(Risr, Yem)
    Yt = np.dot(Risr, Y)
    # numpy svd quirk: V is actually V^T!
    U, s, V = np.linalg.svd(Htp, full_matrices=True)
    if not nsvs:
        nsvs = len(s) - 1
    if verbose:
        print('ndof :'+str(ndof))
        print('U :'+str(U.shape))
        print('s :'+str(s.shape))
        print('V :'+str(V.shape))
        print('recontructing using ' + str(nsvs) + ' singular values')

    innov = np.dot(U.T, Yt-np.squeeze(Htm))
    # Kalman gain
    Kpre = s[0:nsvs]/(s[0:nsvs]*s[0:nsvs] + 1)
    K = np.zeros([nens, nobs])
    np.fill_diagonal(K, Kpre)
    # ensemble-mean analysis increment in transformed space
    xhatinc = np.dot(K, innov)
    # ensemble-mean analysis increment in the transformed ensemble space
    xtinc = np.dot(V.T, xhatinc)/np.sqrt(nens-1)
    if transform_only:
        xam = []
        Xap = []
    else:
        # ensemble-mean analysis increment in the original space
        xinc = np.dot(Xbp, xtinc)
        # ensemble mean analysis in the original space
        xam = xbm + xinc

        # transform the ensemble perturbations
        lam = np.zeros([nobs, nens])
        np.fill_diagonal(lam, s[0:nsvs])
        tmp = np.linalg.inv(np.dot(lam, lam.T) + np.identity(nobs))
        sigsq = np.identity(nens) - np.dot(np.dot(lam.T, tmp), lam)
        sig = np.sqrt(sigsq)
        T = np.dot(V.T, sig)
        Xap = np.dot(Xbp, T)
        # perturbations must have zero mean
        #Xap = Xap - Xap.mean(axis=1,keepdims=True)
        if verbose:
            print('min s:', np.min(s))
    elapsed_time = time() - begin_time
    if verbose:
        print('shape of U: ' + str(U.shape))
        print('shape of s: ' + str(s.shape))
        print('shape of V: ' + str(V.shape))
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    #  readme =
    #  The SVD dictionary contains the SVD matrices U,s,V where V
    #  is the transpose of what numpy returns. xtinc is the ensemble-mean
    #  analysis increment in the intermediate space; *any* state variable
    #  can be reconstructed from this matrix.
    SVD = {
        'U': U,
        's': s,
        'V': np.transpose(V),
        'xtinc': xtinc,
        #  'readme': readme,
    }
    return xam, Xap, SVD


def save_to_netcdf(prior, field_ens_save, recon_years, seed, save_dirpath, dtype=np.float32):
    grid = make_grid(prior)
    lats = grid.lat
    lons = grid.lon
    nens = grid.nens

    nyr = np.size(recon_years)

    var_names = prior.trunc_state_info.keys()

    field_ens_mean = {}
    for name in var_names:
        field_ens_mean[name] = np.average(field_ens_save[name], axis=1)

    output_dict = {}
    for name in var_names:
        field_tmp = np.array(field_ens_mean[name], dtype=dtype)
        output_dict[name] = (('year', 'lat', 'lon'), field_tmp)

        gm_ens = np.zeros((nyr, nens))
        nhm_ens = np.zeros((nyr, nens))
        shm_ens = np.zeros((nyr, nens))

        for k in range(nens):
            gm_ens[:, k], nhm_ens[:, k], shm_ens[:, k] = global_hemispheric_means(
                field_ens_save[name][:, k, :, :], lats)

        # compress to float32
        gm_ens = np.array(gm_ens, dtype=dtype)
        nhm_ens = np.array(nhm_ens, dtype=dtype)
        shm_ens = np.array(shm_ens, dtype=dtype)

        output_dict[f'{name}_gm_ens'] = (('year', 'ens'), gm_ens)
        output_dict[f'{name}_nhm_ens'] = (('year', 'ens'), nhm_ens)
        output_dict[f'{name}_shm_ens'] = (('year', 'ens'), shm_ens)

        if name == 'tas_sfc_Amon':
            # calculate NINO indices
            nino_ind = nino_indices(field_ens_save[name], lats, lons)
            nino12 = nino_ind['nino1+2']
            nino3 = nino_ind['nino3']
            nino34 = nino_ind['nino3.4']
            nino4 = nino_ind['nino4']

            nino12 = np.array(nino12, dtype=dtype)
            nino3 = np.array(nino3, dtype=dtype)
            nino34 = np.array(nino34, dtype=dtype)
            nino4 = np.array(nino4, dtype=dtype)

            output_dict['nino1+2'] = (('year', 'ens'), nino12)
            output_dict['nino3'] = (('year', 'ens'), nino3)
            output_dict['nino3.4'] = (('year', 'ens'), nino34)
            output_dict['nino4'] = (('year', 'ens'), nino4)

            # calculate tripole index (TPI)
            tpi = calc_tpi(field_ens_save[name], lats, lons)
            output_dict['tpi'] = (('year', 'ens'), tpi)

    os.makedirs(save_dirpath, exist_ok=True)
    save_path = os.path.join(save_dirpath, f'job_r{seed:02d}.nc')

    ds = xr.Dataset(
        data_vars=output_dict,
        coords={
            'year': recon_years,
            'lat': lats,
            'lon': lons,
            'ens': np.arange(nens)
        },
    )

    ds.to_netcdf(save_path)

# ===============================================
#  Time axis handling
# -----------------------------------------------
def ymd2year_float(year, month, day):
    ''' Convert a set of (year, month, day) to an array of floats in unit of year
    '''

    year_float = []
    for y, m, d in zip(year, month, day):
        date = datetime(year=y, month=m, day=d)
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_part = date - fst_day
        year_length = lst_day - fst_day
        year_float.append(y + year_part/year_length)

    year_float = np.asarray(year_float)
    return year_float


def datetime2year_float(date):
    year = [d.year for d in date]
    month = [d.month for d in date]
    day = [d.day for d in date]

    year_float = ymd2year_float(year, month, day)

    return year_float


def year_float2datetime(year_float, resolution='day'):
    if np.min(year_float) < 0:
        raise ValueError('Cannot handel negative years. Please truncate first.')
        return None

    ''' Convert an array of floats in unit of year to a datetime time; accuracy: one day
    '''
    year = np.array([int(y) for y in year_float], dtype=int)
    month = np.zeros(np.size(year), dtype=int)
    day = np.zeros(np.size(year), dtype=int)

    for i, y in enumerate(year):
        fst_day = datetime(year=y, month=1, day=1)
        lst_day = datetime(year=y+1, month=1, day=1)
        year_length = lst_day - fst_day

        year_part = (year_float[i] - y)*year_length + timedelta(minutes=1)  # to fix the numerical error
        date = year_part + fst_day
        month[i] = date.month
        day[i] = date.day

    if resolution == 'day':
        time = [cftime.DatetimeNoLeap(y, m, d, 0, 0, 0, 0, 0, 0) for y, m, d in zip(year, month, day)]
    elif resolution == 'month':
        time = [cftime.DatetimeNoLeap(y, m, 1, 0, 0, 0, 0, 0, 0) for y, m in zip(year, month)]
    return time


def seasonal_var_xarray(var, year_float, avgMonths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    ''' Annualize a variable array based on seasonality

    Args:
        var (ndarray): the target variable array with 1st dim to be year
        year_float (1-D array): the time axis of the variable array

    Returns:
        var_ann (ndarray): the annualized variable array
        year_ann (1-D array): the time axis of the annualized variable array
    '''
    var = np.array(var)
    year_float = np.array(year_float)

    ndims = len(np.shape(var))
    dims = ['time']
    for i in range(ndims-1):
        dims.append(f'dim{i+1}')

    time = year_float2datetime(year_float)
    var_da = xr.DataArray(var, dims=dims, coords={'time': time})

    m_start, m_end = avgMonths[0], avgMonths[-1]

    if m_start > 0:
        offset_str = 'A'
    else:
        offset_alias = {
           -12: 'DEC',
           -11: 'NOV',
           -10: 'OCT',
           -9: 'SEP',
           -8: 'AUG',
           -7: 'JUL',
           -6: 'JUN',
           -5: 'MAY',
           -4: 'APR',
           -3: 'MAR',
           -2: 'FEB',
           -1: 'JAN',
        }
        offset_str = f'AS-{offset_alias[m_start]}'

    if avgMonths==[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        var_ann = var_da.groupby('time.year').mean('time')
        year_ann = np.array(list(set(var_ann['year'].values)))
    else:
        month = var_da.groupby('time.month').apply(lambda x: x).month

        var_ann = var_da.where(
            (month >= m_start) & (month <= m_end)
        ).resample(time=offset_str).mean('time')

        if m_start > 0:
            year_ann = np.array(list(set(var_ann['time.year'].values)))
        else:
            year_ann = np.array(list(set(var_ann['time.year'].values))) + 1
            var_ann = var_ann[:-1]
            year_ann = year_ann[:-1]

    var_tmp = np.copy(var_ann)
    var_ann = var_ann[~np.isnan(var_tmp)]
    year_ann = year_ann[~np.isnan(var_tmp)]
    return var_ann, year_ann


def seasonal_var(var, year_float, avgMonths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], make_yr_mm_nan=True):
    ''' Annualize a variable array based on seasonality

    Args:
        var (ndarray): the target variable array with 1st dim to be year
        year_float (1-D array): the time axis of the variable array
        make_yr_mm_nan (bool): make year with missing months nan or not

    Returns:
        var_ann (ndarray): the annualized variable array
        year_ann (1-D array): the time axis of the annualized variable array
    '''
    var = np.array(var)
    var_shape = np.shape(var)
    ndim = len(var_shape)
    year_float = np.array(year_float)

    time = year_float2datetime(year_float)

    nbmonths = len(avgMonths)
    cyears = np.asarray(list(set([t.year for t in time])))
    year_ann = cyears
    nbcyears = len(cyears)

    var_ann_shape = np.copy(var_shape)
    var_ann_shape[0] = nbcyears
    var_ann = np.zeros(shape=var_ann_shape)
    var_ann[:, ...] = np.nan # initialize with nan's
    for i in range(nbcyears):
        # monthly data from current year
        indsyr = [j for j,v in enumerate(time) if v.year == cyears[i] and v.month in avgMonths]
        # check if data from previous year is to be included
        indsyrm1 = []
        if any(m < 0 for m in avgMonths):
            year_before = [abs(m) for m in avgMonths if m < 0]
            indsyrm1 = [j for j,v in enumerate(time) if v.year == cyears[i]-1. and v.month in year_before]
        # check if data from following year is to be included
        indsyrp1 = []
        if any(m > 12 for m in avgMonths):
            year_follow = [m-12 for m in avgMonths if m > 12]
            indsyrp1 = [j for j,v in enumerate(time) if v.year == cyears[i]+1. and v.month in year_follow]

        inds = indsyrm1 + indsyr + indsyrp1

        if ndim == 1:
            tmp = np.nanmean(var[inds], axis=0)
            nancount = np.isnan(var[inds]).sum(axis=0)
            if nancount > 0:
                tmp = np.nan
        else:
            tmp = np.nanmean(var[inds, ...], axis=0)
            nancount = np.isnan(var[inds, ...]).sum(axis=0)
            tmp[nancount > 0] = np.nan

        if make_yr_mm_nan and len(inds) != nbmonths:
                tmp = np.nan

        var_ann[i, ...] = tmp

    return var_ann, year_ann


def make_xr(var, year_float):
    var = np.array(var)
    year_float = np.array(year_float)

    ndims = len(np.shape(var))
    dims = ['time']
    for i in range(ndims-1):
        dims.append(f'dim{i+1}')

    time = year_float2datetime(year_float)
    var_da = xr.DataArray(var, dims=dims, coords={'time': time})
    return var_da


def annualize_var(var, year_float, resolution='month', weights=None):
    ''' Annualize a variable array

    Args:
        var (ndarray): the target variable array with 1st dim to be year
        year_float (1-D array): the time axis of the variable array
        weights (ndarray): the weights that shares the same shape of the target variable array

    Returns:
        var_ann (ndarray): the annualized variable array
        year_ann (1-D array): the time axis of the annualized variable array
    '''
    var = np.array(var)
    year_float = np.array(year_float)

    ndims = len(np.shape(var))
    dims = ['time']
    for i in range(ndims-1):
        dims.append(f'dim{i+1}')

    time = year_float2datetime(year_float, resolution=resolution)

    if weights is not None:
        weights_da = xr.DataArray(weights, dims=dims, coords={'time': time})

        coeff = np.ndarray(np.shape(weights))
        for i, gp in enumerate(list(weights_da.groupby('time.year'))):
            year, value = gp
            k = np.shape(value)[0]
            coeff[k*i:k*(i+1)] = value / np.sum(value, axis=0)

        del weights, weights_da  # save the memory

        var = np.multiply(coeff, var)
        var_da = xr.DataArray(var, dims=dims, coords={'time': time})
        var_ann = var_da.groupby('time.year').sum('time')

    else:
        var_da = xr.DataArray(var, dims=dims, coords={'time': time})
        var_ann = var_da.groupby('time.year').mean('time')

    var_ann = var_ann.values

    year_ann = np.sort(list(set([t.year for t in time])))
    return var_ann, year_ann


def get_nc_vars(filepath, varnames, useLib='xarray'):
    ''' Get variables from given ncfile
    '''
    var_list = []

    if type(varnames) is str:
        varnames = [varnames]

    def load_with_xarray():
        with xr.open_dataset(filepath) as ds:

            for varname in varnames:
                if varname == 'year_float':
                    time = ds['time'].values
                    if type(time[0]) is np.datetime64:
                        time = pd.DatetimeIndex(time)

                    year = [d.year for d in time]
                    month = [d.month for d in time]
                    day = [d.day for d in time]

                    year_float = ymd2year_float(year, month, day)
                    var_list.append(year_float)

                else:
                    var_tmp = ds[varname].values
                    if varname == 'lon':
                        if np.min(var_tmp) < 0:
                            var_tmp = np.mod(var_tmp, 360)  # convert from (-180, 180) to (0, 360)
                    var_list.append(var_tmp)

        return var_list

    def load_with_netCDF4():
        with netCDF4.Dataset(filepath, 'r') as ds:
            for varname in varnames:
                if varname == 'year_float':
                    time = ds.variables['time']
                    time_convert = netCDF4.num2date(time[:], time.units, time.calendar)

                    year_float = []
                    for t in time_convert:
                        y, m, d = t.year, t.month, t.day
                        date = datetime(year=y, month=m, day=d)
                        fst_day = datetime(year=y, month=1, day=1)
                        lst_day = datetime(year=y+1, month=1, day=1)
                        year_part = date - fst_day
                        year_length = lst_day - fst_day
                        year_float.append(y + year_part/year_length)

                    var_list.append(np.asarray(year_float))

                else:
                    var_tmp = ds.variables[varname][:]
                    if varname == 'lon':
                        if np.min(var_tmp) < 0:
                            var_tmp = np.mod(var_tmp, 360)  # convert from (-180, 180) to (0, 360)
                    var_list.append(var_tmp)

        return var_list

    load_nc = {
        'xarray': load_with_xarray,
        'netCDF4': load_with_netCDF4,
    }

    var_list = load_nc[useLib]()

    if len(var_list) == 1:
        var_list = var_list[0]

    return var_list


def get_anomaly(var, year_float, ref_period=(1951, 1980)):
    var_da = make_xr(var, year_float)
    if ref_period[0] > np.max(year_float):
        print(f'Time axis not overlap with the reference period {ref_period}; use its own time period as reference {(int(np.min(year_float)), int(np.max(year_float)))}.')
        var_ref = var_da
    else:
        var_ref = var_da.loc[str(ref_period[0]):str(ref_period[-1])]
    climatology = var_ref.groupby('time.month').mean('time')
    var_anom = var_da.groupby('time.month') - climatology
    return var_anom.values


def get_env_vars(prior_filesdict,
                 rename_vars={'tmp': 'tas', 'pre': 'pr', 'd18O': 'd18Opr', 'tos': 'sst', 'sos': 'sss'},
                 useLib='xarray', lat_str='lat', lon_str='lon',
                 calc_anomaly=False, ref_period=(1951, 1980),
                 factor_vars={'tas': None, 'pr': None, 'dNone8Opr': None, 'sst': None, 'sss': None},
                 bias_vars={'tas': None, 'pr': None, 'd18Opr': None, 'sst': None, 'sss': None},
                 verbose=False):

    prior_vars = {}

    first_item = True
    for prior_varname, prior_filepath in prior_filesdict.items():
        if verbose:
            print(f'Loading [{prior_varname}] from {prior_filepath} ...')
        if first_item:
            lat_model, lon_model, time_model, prior_vars[prior_varname] = get_nc_vars(
                prior_filepath, [lat_str, lon_str, 'year_float', prior_varname], useLib=useLib,
            )
            first_item = False
        else:
            prior_vars[prior_varname] = get_nc_vars(prior_filepath, prior_varname, useLib=useLib)

        # calculate the anomaly
        if calc_anomaly:
            prior_vars[prior_varname] = get_anomaly(prior_vars[prior_varname], time_model, ref_period=ref_period)

    if rename_vars:
        for old_name, new_name in rename_vars.items():
            if old_name in prior_vars:
                print(f'Renaming var: {old_name} -> {new_name}')
                prior_vars[new_name] = prior_vars.pop(old_name)

    for prior_varname, prior_var in prior_vars.items():
        if prior_varname in factor_vars and factor_vars[prior_varname] is not None:
            factor = factor_vars[prior_varname]
        else:
            factor = 1

        if prior_varname in bias_vars and bias_vars[prior_varname] is not None:
            bias = bias_vars[prior_varname]
        else:
            bias = 0

        if factor != 1 or bias != 0:
            prior_vars[prior_varname] = prior_var * factor + bias
            print(f'Converting {prior_varname}: {prior_varname} = {prior_varname} * ({factor}) + ({bias})')

    return lat_model, lon_model, time_model, prior_vars


def pick_range(ts, ys, lb, ub):
    ''' Pick range [lb, ub) from a timeseries pair (ts, ys)
    '''
    range_mask = (ts >= lb) & (ts < ub)
    return ts[range_mask], ys[range_mask]


def pick_years(year_int, time_grid, var_grid):
    ''' Pick years from a timeseries pair (time_grid, var_grid) based on year_int
    '''
    year_int = [int(y) for y in year_int]
    mask = []
    for year_float in time_grid:
        if int(year_float) in year_int:
            mask.append(True)
        else:
            mask.append(False)
    return time_grid[mask], var_grid[mask]


def clean_ts(ts, ys, start=None):
    # delete NaNs if there is any
    ys = np.asarray(ys, dtype=np.float)
    ts = np.asarray(ts, dtype=np.float)

    ys_tmp = np.copy(ys)
    ys = ys[~np.isnan(ys_tmp)]
    ts = ts[~np.isnan(ys_tmp)]
    ts_tmp = np.copy(ts)
    ys = ys[~np.isnan(ts_tmp)]
    ts = ts[~np.isnan(ts_tmp)]

    # sort the time series so that the time axis will be ascending
    sort_ind = np.argsort(ts)
    ys = ys[sort_ind]
    ts = ts[sort_ind]

    if start:
        mask = ts >= start
        ts = ts[mask]
        ys = ys[mask]

    return ts, ys


def overlap_ts(t1, y1, t2, y2):
    t1, y1 = clean_ts(t1, y1)
    t2, y2 = clean_ts(t2, y2)

    # get overlap range
    overlap_yrs = np.intersect1d(t1, t2)
    ind1 = np.searchsorted(t1, overlap_yrs)
    ind2 = np.searchsorted(t2, overlap_yrs)
    y1_overlap = y1[ind1]
    y2_overlap = y2[ind2]
    t1_overlap = t1[ind1]
    t2_overlap = t2[ind2]

    return t1_overlap, y1_overlap, t2_overlap, y2_overlap


def rolling_avg(ts, ys, win_len, rolling_kws={}):
    data = {'time': ts, 'value': ys}
    df = pd.DataFrame(data)
    df = df.set_index('time')
    df_avg = df.rolling(win_len, **rolling_kws).mean()

    ts_out = df_avg.index.values
    ys_out = df_avg['value'].values

    return ts_out, ys_out


def rolling_std(ts, ys, win_len, rolling_kws={}):
    data = {'time': ts, 'value': ys}
    df = pd.DataFrame(data)
    df = df.set_index('time')
    df_std = df.rolling(win_len, **rolling_kws).std()

    ts_out = df_std.index.values
    ys_out = df_std['value'].values

    return ts_out, ys_out


def get_distance(lon_pt, lat_pt, lon_ref, lat_ref):
    """
    Vectorized calculation the great circle distances between lat-lon points
    on the Earth (lat/lon are specified in decimal degrees)

    Input:
    lon_pt , lat_pt  : longitude, latitude of site w.r.t. which distances
                       are to be calculated. Both should be scalars.
    lon_ref, lat_ref : longitudes, latitudes of reference field
                       (e.g. calibration dataset, reconstruction grid)
                       May be scalar, 1D arrays, or 2D arrays.

    Output: Returns array containing distances between (lon_pt, lat_pt) and all other points
            in (lon_ref,lat_ref). Array has dimensions [dim(lon_ref),dim(lat_ref)].

    Originator: R. Tardif, Atmospheric sciences, U. of Washington, January 2016

    """

    # Convert decimal degrees to radians
    lon_pt, lat_pt, lon_ref, lat_ref = list(map(np.radians, [lon_pt, lat_pt, lon_ref, lat_ref]))

    # check dimension of lon_ref and lat_ref
    dims_ref = len(lon_ref.shape)

    if dims_ref == 0:
        lats = lat_ref
        lons = lon_ref
    elif dims_ref == 1:
        lon_dim = len(lon_ref)
        lat_dim = len(lat_ref)
        nbpts = lon_dim*lat_dim
        lats = np.array([lat_ref,]*lon_dim).transpose()
        lons = np.array([lon_ref,]*lat_dim)
    elif dims_ref == 2:
        lats = lat_ref
        lons = lon_ref
    else:
        print('ERROR in get_distance!')
        raise SystemExit()

    # Haversine formula using arrays as input
    dlon = lons - lon_pt
    dlat = lats - lat_pt

    a = np.sin(dlat/2.)**2 + np.cos(lat_pt) * np.cos(lats) * np.sin(dlon/2.)**2
    c = 2. * np.arcsin(np.sqrt(a))
    km = 6367.0 * c

    return km


def calc_seasonal_avg(
    var, year_float,
    seasonality=[
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [3, 4, 5, 6, 7, 8],
        [3, 4, 5],
        [4],
        [4, 5, 6, 7, 8, 9],
        [4, 5, 6],
        [6],
        [6, 7, 8],
        [6, 7, 8, 9, 10, 11],
        [7, 8, 9],
        [9, 10, 11],
        [6, 7],
        [7],
        [-12, 1, 2, 3],
        [-5, -6, -7, -8, -9, -10, -11, -12, 1, 2, 3, 4],
        [-9, -10, -11, -12, 1, 2],
        [-12, 1, 2],
        [-10, -11, -12, 1, 2, 3, 4],
        [-12, -11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [-12, 1, 2, 3, 4, 5],
        [-9, -10, -11, -12, 1, 2, 3, 4],
        [-12, -11, -10, -9, 1, 2, 3, 4, 5, 6, 7, 8],
        [-12, -11, -10, -9, -8, -7, 1, 2, 3, 4, 5, 6]
    ],
    lat=None, lon=None,
    save_path=None,
    make_yr_mm_nan=True,
    verbose=False,
):
    ''' Pre-calculate the seasonal average of the target variable
    '''
    var_ann_dict = {}
    for avgMonths in seasonality:
        if verbose:
            print(f'>>> Processing {avgMonths}')

        var_ann, year_ann = seasonal_var(var, year_float, avgMonths=avgMonths, make_yr_mm_nan=make_yr_mm_nan)

        season_tag = '_'.join(str(m) for m in avgMonths)
        var_ann_dict[season_tag] = var_ann

    if save_path:
        with open(save_path, 'wb') as f:
            if lat is None or lon is None:
                pickle.dump([var_ann_dict, year_ann], f)
            else:
                pickle.dump([var_ann_dict, year_ann, lat, lon], f)

    return var_ann_dict, year_ann


def ts_matching(time_target, value_target, time_ref, value_ref, match_std=True, match_mean=True, verbose=False):
    ''' Perform mean correction and variance matching against the reference timeseries over overlapped time interval

    Args:
        time_target (1-D array): the time axis of the target timeseries
        value_target (1-D array): the value axis of the target timeseries
        time_ref (1-D array): the time axis of the reference timeseries
        value_ref (1-D array): the value axis of the reference timeseries

    Returns:
        time_corrected (1-D array): the time axis of the target timeseries after correction
        value_corrected (1-D array): the value axis of the target timeseries after correction
    '''

    value_target = np.array(value_target)
    value_ref = np.array(value_ref)

    value_target_bak = np.copy(value_target)

    if match_std is True:
        t_target, v_target, t_ref, v_ref = overlap_ts(time_target, value_target, time_ref, value_ref)
        factor = np.std(v_ref) / np.std(v_target)
        value_target = factor * value_target
    else:
        factor = np.nan

    # order matters: match_std first and then match_mean
    if match_mean is True:
        t_target, v_target, t_ref, v_ref = overlap_ts(time_target, value_target, time_ref, value_ref)
        bias = np.mean(v_ref) - np.mean(v_target)
        value_target = value_target + bias
    else:
        bias = np.nan

    if verbose:
        t_target, v_target_bak, t_ref, v_ref = overlap_ts(time_target, value_target_bak, time_ref, value_ref)
        t_target, v_target, t_ref, v_ref = overlap_ts(time_target, value_target, time_ref, value_ref)

        print(f'---------------------------')
        print(f'overlapped timepoints: {len(t_ref)}')
        print(f'---------------------------')
        print(f'\tmean\tstd')
        print(f'---------------------------')
        print(f'ref:\t{np.mean(v_ref):.2f}\t{np.std(v_ref):.2f}')
        print(f'before:\t{np.mean(v_target_bak):.2f}\t{np.std(v_target_bak):.2f}')
        print(f'after:\t{np.mean(v_target):.2f}\t{np.std(v_target):.2f}')
        print()

    res_dict = {
        'value_target': value_target,
        'factor': factor,
        'bias': bias,
    }

    return res_dict


def compute_annual_means(time_raw,data_raw,valid_frac,year_type):
    """
    Computes annual-means from raw data.
    Inputs:
        time_raw   : Original time axis
        data_raw   : Original data
        valid_frac : The fraction of sub-annual data necessary to create annual mean.  Otherwise NaN.
        year_type  : "calendar year" (Jan-Dec) or "tropical year" (Apr-Mar)

    Outputs: time_annual, data_annual

    Authors: R. Tardif, Univ. of Washington; M. Erb, Univ. of Southern California

    """

    # Check if dealing with multiple chronologies in one data stream (for NCDC files)
    array_shape = data_raw.shape
    if len(array_shape) == 2:
        nbtimes, nbvalid = data_raw.shape
    elif len(array_shape) == 1:
        nbtimes, = data_raw.shape
        nbvalid = 1
    else:
        raise SystemExit('ERROR in compute_annual_means: Unrecognized shape of data input array.')

    # -------------------------------------------
    # Determine temporal resolution of the record
    # -------------------------------------------

    # time differences between adjacent data pts
    time_between_records = np.diff(time_raw, n=1)

    # Temporal resolution taken as the mode of the time differences
    mode_time_between_records = stats.mode(time_between_records)
    time_resolution = abs(mode_time_between_records.mode[0])

    # check if time_resolution = 0.0 !!! sometimes adjacent records are tagged at same time ...
    if time_resolution == 0.0:
        print('***WARNING! Found adjacent records with same times!')
        inderr = np.where(time_between_records == 0.0)
        print(inderr)
        time_between_records = np.delete(time_between_records,inderr)
        time_resolution = abs(stats.mode(time_between_records)[0][0])

    # some extra checks
    if mode_time_between_records.count[0] == 1:
        # all unique time differences
        # subannual or annual+ ?
        years_in_dataset = np.floor(time_raw)
        years_between_records = np.diff(years_in_dataset, n=1)
        mode_years_between_records = stats.mode(years_between_records)
        if mode_years_between_records.mode[0] > 0:
            # annual+ resolution
            # take resolution as smallest detected interval that is annual or longer
            time_resolution = np.min(years_between_records[years_between_records>0.])
        else:
            # subannual
            time_resolution = np.mean(time_between_records)

    if time_resolution <= 1.0:
        proxy_resolution = int(1.0) # coarse-graining to annual
    else:
        proxy_resolution = int(time_resolution)


    # Get rounded integer values of all years present in record.
    years_all = [int(np.floor(time_raw[k])) for k in range(0,len(time_raw))]
    years = list(set(years_all)) # 'set' is used to get unique values in list
    years = sorted(years) # sort the list

    years = np.insert(years,0,years[0]-1) # M. Erb

    # bounds, for calendar year : [years_beg,years_end[
    years_beg = np.asarray(years,dtype=np.float64) # inclusive lower bound
    years_end = years_beg + 1.                     # exclusive upper bound

    # If some of the time values are floats (sub-annual resolution)
    # and year_type is tropical_year, adjust the years to cover the
    # tropical year (Apr-Mar).
    if np.equal(np.mod(time_raw,1),0).all() == False and year_type == 'tropical year':
        print("Tropical year averaging...")

        # modify bounds defining the "year"
        for i, yr in enumerate(years):
            # beginning of interval
            if calendar.isleap(yr):
                years_beg[i] = float(yr)+((31+29+31)/float(366))
            else:
                years_beg[i] = float(yr)+((31+28+31)/float(365))
            # end of interval
            if calendar.isleap(yr+1):
                years_end[i] = float(yr+1)+((31+29+31)/float(366))
            else:
                years_end[i] = float(yr+1)+((31+28+31)/float(365))

    time_annual = np.asarray(years,dtype=np.float64)
    data_annual = np.zeros(shape=[len(years),nbvalid], dtype=np.float64)
    # fill with NaNs for default values
    data_annual[:] = np.NAN

    # Calculate the mean of all data points with the same year.
    for i in range(len(years)):
        ind = [j for j, year in enumerate(time_raw) if (year >= years_beg[i]) and (year < years_end[i])]
        nbdat = len(ind)

        # TODO: check nb of non-NaN values !!!!! ... ... ... ... ... ...

        if time_resolution <= 1.0:
            max_nb_per_year = int(1.0/time_resolution)
            frac = float(nbdat)/float(max_nb_per_year)
            if frac > valid_frac:
                data_annual[i,:] = np.nanmean(data_raw[ind],axis=0)
        else:
            if nbdat > 1:
                print('***WARNING! Found multiple records in same year in data with multiyear resolution!')
                print('   year= %d %d' %(years[i], nbdat))
            # Note: this calculates the mean if multiple entries found
            data_annual[i,:] = np.nanmean(data_raw[ind],axis=0)


    # check and modify time_annual array to reflect only the valid data present in the annual record
    # for correct tagging of "Oldest" and "Youngest" data
    indok = np.where(np.isfinite(data_annual))[0]
    keep = np.arange(indok[0],indok[-1]+1,1)


    return time_annual[keep], data_annual[keep,:], proxy_resolution


def make_groups(ys, window, apply_func=None, apply_kws={}):
    nt = np.size(ys)
    ngrp = nt // window
    grp = np.ndarray((ngrp, window))
    grp[:] = np.nan
    for i in range(ngrp):
        grp[i] = ys[i:i+window]

    if apply_func is None:
        res = grp
    else:
        res = np.empty(ngrp)
        for i, g in enumerate(grp):
            res[i] = apply_func(g, **apply_kws)

    return res

# ===============================================
#  Multivariate bias correction
# -----------------------------------------------
def mbc(tas, pr, time,
         ref_tas, ref_pr, ref_time,
         seed=0, Rlib_path='/Library/Frameworks/R.framework/Versions/3.6/Resources/library'):
    ''' Perform multivariate bias correction

    Args:
        tas (1-D array): the temperature timeseries at the proxy location
        pr (1-D array): the precipitation timeseries at the proxy location
    '''

    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    import rpy2.robjects as ro
    rpy2.robjects.numpy2ri.activate()

    random.seed(seed)

    if np.nanmean(ref_tas) < 200:
        ref_tas += 273.15  # convert to [K]
    if np.nanmean(ref_pr) > 1:
        ref_pr = ref_pr / (3000*24*30)  # convert to precipitation rate in [kg/m2/s]

    # make the resolution of the time axis be month
    ref_date = year_float2datetime(ref_time, resolution='month')
    ref_time_fix = datetime2year_float(ref_date)

    model_date = year_float2datetime(time, resolution='month')
    model_time_fix = datetime2year_float(model_date)

    # get the overlapped timespan for calibration
    overlap_yrs = np.intersect1d(ref_time_fix, model_time_fix)
    ind_ref = np.searchsorted(ref_time, overlap_yrs)
    ind_model = np.searchsorted(time, overlap_yrs)

    ref_tas_c = ref_tas[ind_ref]
    ref_pr_c = ref_pr[ind_ref]
    ref_vars_c = np.array([ref_pr_c, ref_tas_c]).T

    model_tas_c = tas[ind_model]
    model_pr_c = pr[ind_model]
    model_vars_c = np.array([model_pr_c, model_tas_c]).T
    model_vars = np.array([pr, tas]).T
    # run the multivariate bias correction function
    ro.r('.libPaths("{}")'.format(Rlib_path))
    MBCn_R = importr('MBC').MBCn

    res = MBCn_R(o_c=ref_vars_c, m_c=model_vars_c, m_p=model_vars)
    res_array = np.array(res[1])

    pr_corrected = res_array[:, 0]
    tas_corrected = res_array[:, 1]

    return tas_corrected, pr_corrected

# ===============================================
#  Noise
# -----------------------------------------------
def ar1_noise(ts, ys, g=None, sig_noise=1, nt_noise=None, seed=0):
    '''Returns the AR1 noise
    '''
    np.random.seed(seed)
    ts, ys = clean_ts(ts, ys)

    if nt_noise is None:
        nt_noise = np.size(ts)

    nt = np.size(ts)
    dts = np.diff(ts)
    dt_mean = np.mean(dts)
    if any(dt == dt_mean for dt in dts):
        evenly_spaced = True
    else:
        evenly_spaced = False

    if evenly_spaced:
        # evenly spaced case
        if g is None:
            ar1_mod = sm.tsa.AR(ys, missing='drop').fit(maxlag=1)
            g = ar1_mod.params[1]

        ar = np.r_[1, -g]
        ma = np.r_[1, 0.0]

        noise = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=nt_noise, burnin=50)

    else:
        # unevenly spaced case
        def ar1_func(a):
            return np.sum((ys[1:] - ys[:-1]*a**dts)**2)

        if g is None:
            a_est = optimize.minimize_scalar(ar1_func, bounds=[0, 1], method='bounded').x
            g = a_est
        else:
            a_est = g

        tau_est = -1 / np.log(a_est)

        noise = np.ones(nt_noise)
        for i in range(1, nt_noise):
            scaled_dt =  (ts[i] - ts[i-1]) / tau_est
            rho = np.exp(-scaled_dt)
            err = np.random.normal(0, np.sqrt(1- rho**2), 1)
            noise[i] = noise[i-1]*rho + err

    noise = noise / np.nanstd(noise) * sig_noise

    return noise, g


def colored_noise_2regimes(alpha1, alpha2, f_break, t, f0=None, m=None, seed=None):
    ''' Generate a colored noise timeseries with two regimes

    Args:
        alpha1, alpha2 (float): the exponent of the 1/f^alpha noise
        f_break (float): the frequency where the scaling breaks
        t (float): time vector of the generated noise
        f0 (float): fundamental frequency
        m (int): maximum number of the waves, which determines the
            highest frequency of the components in the synthetic noise

    Returns:
        y (array): the generated 1/f^alpha noise

    References:
        Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
            Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).
    '''
    n = np.size(t)  # number of time points
    y = np.zeros(n)

    if f0 is None:
        f0 = 1/n  # fundamental frequency
    if m is None:
        m = n//2  # so the aliasing is limited

    k = np.arange(m) + 1  # wave numbers

    if seed is not None:
        np.random.seed(seed)

    theta = np.random.rand(int(m))*2*np.pi  # random phase

    f_vec = k*f0
    regime1= k*f0>=f_break
    regime2= k*f0<=f_break
    f_vec1 = f_vec[regime1]
    f_vec2 = f_vec[regime2]
    s = np.exp(alpha1/alpha2*np.log(f_vec1[0])) / f_vec2[-1]

    for j in range(n):
        coeff = np.ndarray((np.size(f_vec)))
        coeff[regime1] = f_vec1**(-alpha1/2)
        coeff[regime2] = (s*f_vec2)**(-alpha2/2)
        sin_func = np.sin(2*np.pi*k*f0*t[j] + theta)
        y[j] = np.sum(coeff*sin_func)

    return y


def colored_noise(alpha, t, f0=None, m=None, seed=None):
    ''' Generate a colored noise timeseries

    Args:
        alpha (float): exponent of the 1/f^alpha noise
        t (float): time vector of the generated noise
        f0 (float): fundamental frequency
        m (int): maximum number of the waves, which determines the
            highest frequency of the components in the synthetic noise

    Returns:
        y (array): the generated 1/f^alpha noise

    References:
        Eq. (15) in Kirchner, J. W. Aliasing in 1/f(alpha) noise spectra: origins, consequences, and remedies.
            Phys Rev E Stat Nonlin Soft Matter Phys 71, 066110 (2005).
    '''
    n = np.size(t)  # number of time points
    y = np.zeros(n)

    if f0 is None:
        f0 = 1/n  # fundamental frequency
    if m is None:
        m = n//2

    k = np.arange(m) + 1  # wave numbers

    if seed is not None:
        np.random.seed(seed)

    theta = np.random.rand(int(m))*2*np.pi  # random phase
    for j in range(n):
        coeff = (k*f0)**(-alpha/2)
        sin_func = np.sin(2*np.pi*k*f0*t[j] + theta)
        y[j] = np.sum(coeff*sin_func)

    return y

# ===============================================
#  Post processing
# -----------------------------------------------
def global_hemispheric_means(field, lat):
    """ Adapted from LMR_utils.py by Greg Hakim & Robert Tardif | U. of Washington

     compute global and hemispheric mean valuee for all times in the input (i.e. field) array
     input:  field[ntime,nlat,nlon] or field[nlat,nlon]
             lat[nlat,nlon] in degrees

     output: gm : global mean of "field"
            nhm : northern hemispheric mean of "field"
            shm : southern hemispheric mean of "field"
    """

    # Originator: Greg Hakim
    #             University of Washington
    #             August 2015
    #
    # Modifications:
    #           - Modified to handle presence of missing values (nan) in arrays
    #             in calculation of spatial averages [ R. Tardif, November 2015 ]
    #           - Enhanced flexibility in the handling of missing values
    #             [ R. Tardif, Aug. 2017 ]

    # set number of times, lats, lons; array indices for lat and lon
    if len(np.shape(field)) == 3: # time is a dimension
        ntime,nlat,nlon = np.shape(field)
        lati = 1
        loni = 2
    else: # only spatial dims
        ntime = 1
        nlat,nlon = np.shape(field)
        field = field[None,:] # add time dim of size 1 for consistent array dims
        lati = 1
        loni = 2

    # latitude weighting
    lat_weight = np.cos(np.deg2rad(lat))
    tmp = np.ones([nlon,nlat])
    W = np.multiply(lat_weight,tmp).T

    # define hemispheres
    eqind = nlat//2

    if lat[0] > 0:
        # data has NH -> SH format
        W_NH = W[0:eqind+1]
        field_NH = field[:,0:eqind+1,:]
        W_SH = W[eqind+1:]
        field_SH = field[:,eqind+1:,:]
    else:
        # data has SH -> NH format
        W_NH = W[eqind:]
        field_NH = field[:,eqind:,:]
        W_SH = W[0:eqind]
        field_SH = field[:,0:eqind,:]

    gm  = np.zeros(ntime)
    nhm = np.zeros(ntime)
    shm = np.zeros(ntime)

    # Check for valid (non-NAN) values & use numpy average function (includes weighted avg calculation)
    # Get arrays indices of valid values
    indok    = np.isfinite(field)
    indok_nh = np.isfinite(field_NH)
    indok_sh = np.isfinite(field_SH)
    for t in range(ntime):
        if lati == 0:
            # Global
            gm[t]  = np.average(field[indok],weights=W[indok])
            # NH
            nhm[t] = np.average(field_NH[indok_nh],weights=W_NH[indok_nh])
            # SH
            shm[t] = np.average(field_SH[indok_sh],weights=W_SH[indok_sh])
        else:
            # Global
            indok_2d    = indok[t,:,:]
            if indok_2d.any():
                field_2d    = np.squeeze(field[t,:,:])
                gm[t]       = np.average(field_2d[indok_2d],weights=W[indok_2d])
            else:
                gm[t] = np.nan
            # NH
            indok_nh_2d = indok_nh[t,:,:]
            if indok_nh_2d.any():
                field_nh_2d = np.squeeze(field_NH[t,:,:])
                nhm[t]      = np.average(field_nh_2d[indok_nh_2d],weights=W_NH[indok_nh_2d])
            else:
                nhm[t] = np.nan
            # SH
            indok_sh_2d = indok_sh[t,:,:]
            if indok_sh_2d.any():
                field_sh_2d = np.squeeze(field_SH[t,:,:])
                shm[t]      = np.average(field_sh_2d[indok_sh_2d],weights=W_SH[indok_sh_2d])
            else:
                shm[t] = np.nan

    return gm, nhm, shm


def nino_indices(sst, lats, lons):
    ''' Calculate Nino indices

    Args:
        sst: sea-surface temperature, the last two dimensions are assumed to be (lat, lon)
        lats: the latitudes in format of (-90, 90)
        lons: the longitudes in format of (0, 360)
    '''
    def lon360(lon):
        # convert from (-180, 180) to (0, 360)
        return np.mod(lon, 360)

    lats = np.asarray(lats)
    lons = np.asarray(lons)

    lat_mask = {}
    lon_mask = {}
    ind = {}

    lat_mask['nino1+2'] = (lats >= -10) & (lats <= 0)
    lon_mask['nino1+2'] = (lons >= lon360(-90)) & (lons <= lon360(-80))

    lat_mask['nino3'] = (lats >= -5) & (lats <= 5)
    lon_mask['nino3'] = (lons >= lon360(-150)) & (lons <= lon360(-90))

    lat_mask['nino3.4'] = (lats >= -5) & (lats <= 5)
    lon_mask['nino3.4'] = (lons >= lon360(-170)) & (lons <= lon360(-120))

    lat_mask['nino4'] = (lats >= -5) & (lats <= 5)
    lon_mask['nino4'] = (lons >= lon360(160)) & (lons <= lon360(-150))

    for region in lat_mask.keys():
        sst_sub = sst[..., lon_mask[region]]
        sst_sub = sst_sub[..., lat_mask[region], :]
        ind[region] = np.average(
            np.average(sst_sub, axis=-1),
            axis=-1,
            weights=np.cos(np.deg2rad(lats[lat_mask[region]])),
        )
    return ind


def calc_tpi(sst, lats, lons):
    ''' Calculate Nino indices

    Args:
        sst: sea-surface temperature, the last two dimensions are assumed to be (lat, lon)
        lats: the latitudes in format of (-90, 90)
        lons: the longitudes in format of (0, 360)
    '''
    def lon360(lon):
        # convert from (-180, 180) to (0, 360)
        return np.mod(lon, 360)

    lats = np.asarray(lats)
    lons = np.asarray(lons)

    lat_mask = {}
    lon_mask = {}
    ssta = {}

    lat_mask['1'] = (lats >= 25) & (lats <= 45)
    lon_mask['1'] = (lons >= lon360(140)) & (lons <= lon360(-145))

    lat_mask['2'] = (lats >= -10) & (lats <= 10)
    lon_mask['2'] = (lons >= lon360(170)) & (lons <= lon360(-90))

    lat_mask['3'] = (lats >= -50) & (lats <= -15)
    lon_mask['3'] = (lons >= lon360(150)) & (lons <= lon360(-160))

    for region in lat_mask.keys():
        sst_sub = sst[..., lon_mask[region]]
        sst_sub = sst_sub[..., lat_mask[region], :]
        ssta[region] = np.average(
            np.average(sst_sub, axis=-1),
            axis=-1,
            weights=np.cos(np.deg2rad(lats[lat_mask[region]])),
        )

    tpi = ssta['2'] - (ssta['1'] + ssta['3'])/2
    return tpi


def pobjs2df(pobjs,
             col_names=['id', 'type', 'start_yr', 'end_yr', 'lat', 'lon', 'elev', 'seasonality', 'values', 'time']):

    df = pd.DataFrame(index=range(len(pobjs)), columns=col_names)

    for i, pobj in enumerate(pobjs):
        pobj_dict = pobj._asdict()
        for name in col_names:
            if name == 'values':
                entry = pobj_dict[name].values
            else:
                entry = pobj_dict[name]

            df.loc[i, name] = entry

    return df


def compare_ts(t1, y1, t2, y2, stats=['corr', 'ce', 'rmse'], valid_frac=0.5, mask_period=None,
               ref_period=None, detrend=False, detrend_kws={}, npts_lb=25, corr_method='corr_sig',
               corr_sig_nsim=1000, corr_sig_method='isospectral', corr_sig_alpha=0.05):
    # mask the timeseries
    if mask_period is not None:
        mask_ref1 = (t1 >= mask_period[0]) & (t1 <= mask_period[1])
        mask_ref2 = (t2 >= mask_period[0]) & (t2 <= mask_period[1])
        y1 = y1[mask_ref1]
        y2 = y2[mask_ref2]
        t1 = t1[mask_ref1]
        t2 = t2[mask_ref2]

    # remove mean over ref_period
    if ref_period is not None:
        mask_ref1 = (t1 >= ref_period[0]) & (t1 <= ref_period[1])
        mask_ref2 = (t2 >= ref_period[0]) & (t2 <= ref_period[1])
        y1 -= np.nanmean(y1[mask_ref1])
        y2 -= np.nanmean(y2[mask_ref2])

    if detrend:
        y1 = signal.detrend(y1, **detrend_kws)
        y2 = signal.detrend(y2, **detrend_kws)

    t1_overlap, y1_overlap, t2_overlap, y2_overlap = overlap_ts(t1, y1, t2, y2)

    res = {}
    if 'corr' in stats:
        if np.size(y1_overlap) < npts_lb or np.size(y2_overlap) < npts_lb:
            print('compare_ts() >>> Warning: overlapped timeseries is too short for correlation calculation (npts < 25). Returnning NaN ...')
            res['corr'] = np.nan
        else:
            if corr_method == 'corr_sig':
                res['corr'], res['signif'], res['pvalue'] = corr_sig(
                    y1_overlap, y2_overlap,
                    nsim=corr_sig_nsim, method=corr_sig_method, alpha=corr_sig_alpha
                )
            else:
                res['corr'] = np.corrcoef(y1_overlap, y2_overlap)[1, 0]

    if 'rmse' in stats:
        res['rmse'] = np.sqrt(((y1_overlap - y2_overlap)**2).mean())

    if 'ce' in stats:
        res['ce'] = coefficient_efficiency(y1_overlap, y2_overlap, valid_frac)

    return res


def coefficient_efficiency(ref, test, valid=None):
    """ Compute the coefficient of efficiency for a test time series, with respect to a reference time series.

    Inputs:
    test:  test array
    ref:   reference array, of same size as test
    valid: fraction of valid data required to calculate the statistic

    Note: Assumes that the first dimension in test and ref arrays is time!!!

    Outputs:
    CE: CE statistic calculated following Nash & Sutcliffe (1970)
    """

    # check array dimensions
    dims_test = test.shape
    dims_ref  = ref.shape
    #print('dims_test: ', dims_test, ' dims_ref: ', dims_ref)

    if len(dims_ref) == 3:   # 3D: time + 2D spatial
        dims = dims_ref[1:3]
    elif len(dims_ref) == 2: # 2D: time + 1D spatial
        dims = dims_ref[1:2]
    elif len(dims_ref) == 1: # 0D: time series
        dims = 1
    else:
        print('In coefficient_efficiency(): Problem with input array dimension! Exiting...')
        SystemExit(1)

    CE = np.zeros(dims)

    # error
    error = test - ref

    # CE
    numer = np.nansum(np.power(error,2),axis=0)
    denom = np.nansum(np.power(ref-np.nanmean(ref,axis=0),2),axis=0)
    CE    = 1. - np.divide(numer,denom)

    if valid:
        nbok  = np.sum(np.isfinite(ref),axis=0)
        nball = float(dims_ref[0])
        ratio = np.divide(nbok,nball)
        indok  = np.where(ratio >= valid)
        indbad = np.where(ratio < valid)
        dim_indbad = len(indbad)
        testlist = [indbad[k].size for k in range(dim_indbad)]
        if not all(v == 0 for v in testlist):
            if isinstance(dims,(tuple,list)):
                CE[indbad] = np.nan
            else:
                CE = np.nan

    return CE


def calc_ens_calib_ratio(pobjs, ye_filepath, calc_period=[1850, 2000], verbose=False, exclude_list=None):
    ye = np.load(ye_filepath, allow_pickle=True)
    pid_index_map = ye['pid_index_map'][()]
    ye_vals = ye['ye_vals']

    start_yr, end_yr = calc_period

    calib_ratio = []
    for Y in pobjs:
        pid = Y.id
        if exclude_list is None or pid not in exclude_list:
            if pid in pid_index_map:
                idx = pid_index_map[pid]
                ye_mean = np.mean(ye_vals[idx])
                Yvals = Y.values[(Y.values.index > start_yr) & (Y.values.index <= end_yr)].values
                ye_mean_err = ye_mean - Yvals
                ye_mean_err_var = np.var(ye_mean_err, ddof=1)
                evar = np.var(ye_vals[idx], ddof=1)

                ratio = ye_mean_err_var / (evar + Y.psm_obj.R)
                if not np.isnan(ratio):
                    calib_ratio.append(ratio)
                if verbose:
                    print(f'{pid}, ye_mean: {ye_mean:.2f}, ye_mean_err_var: {ye_mean_err_var:.2f}, evar: {evar:.2f}, Y.psm_obj.R: {Y.psm_obj.R:.2f}, ratio: {ratio:.2f}')

    return calib_ratio

# -----------------------------------------------
# Correlation and coefficient Efficiency (CE)
# -----------------------------------------------
def load_ts_from_jobs(exp_dir, qs=[0.05, 0.5, 0.95], var='tas_sfc_Amon_gm_ens', ref_period=[1951, 1980],
                      return_MC_mean=False):
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    paths = sorted(glob.glob(os.path.join(exp_dir, 'job_r*')))

    with xr.open_dataset(paths[0]) as ds:
        ts_tmp = ds[var].values
        year = ds['year'].values

    nt = np.shape(ts_tmp)[0]
    nEN = np.shape(ts_tmp)[-1]
    nMC = len(paths)

    ts = np.ndarray((nt, nEN*nMC))
    ts_MC = np.ndarray((nt, nMC))
    for i, path in enumerate(paths):
        with xr.open_dataset(path) as ds:
            ts_ens = ds[var].values

        ts[:, nEN*i:nEN+nEN*i] = ts_ens
        ts_MC[:, i] = np.average(ts_ens, axis=-1)

    if return_MC_mean:
        ts_qs = ts_MC

    else:
        if qs:
            ts_qs = mquantiles(ts, qs, axis=-1)
        else:
            ts_qs = ts

    return ts_qs, year


def load_field_from_jobs(exp_dir, var='tas_sfc_Amon', average_iter=True):
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    paths = sorted(glob.glob(os.path.join(exp_dir, 'job_r*')))

    field_ens_mean = []
    for i, path in enumerate(paths):
        if i == 0:
            with xr.open_dataset(path) as ds:
                year = ds['year'].values
                lat = ds['lat'].values
                lon = ds['lon'].values

        with xr.open_dataset(path) as ds:
            field_ens_mean.append(ds[var].values)

    if average_iter:
        field_em = np.average(field_ens_mean, axis=0)
    else:
        field_em = field_ens_mean

    return field_em, year, lat, lon


def rotate_lon(field, lon):
    ''' Make lon to be sorted with range (0, 360)

    Args:
        field (ndarray): the last axis is assumed to be lon
        lon (1d array): the longitude axis

    Returns:
        field (ndarray): the field with longitude rotated
        lon (1d array): the sorted longitude axis with range (0, 360)
    '''
    if np.min(lon) < 0:
        lon = np.mod(lon, 360)

    sorted_lon = sorted(lon)
    idx = []
    for lon_gs in sorted_lon:
        idx.append(list(lon).index(lon_gs))
    lon = lon[idx]
    field = field[..., idx]

    return field, lon


def load_inst_analyses(ana_pathdict, var='gm', verif_yrs=np.arange(1880, 2000), ref_period=[1951, 1980],
                       sort_lon=True, avgInterval=list(range(1, 13))):
    load_func = {
        'GISTEMP': load_gridded_data.read_gridded_data_GISTEMP,
        'HadCRUT': load_gridded_data.read_gridded_data_HadCRUT,
        'BerkeleyEarth': load_gridded_data.read_gridded_data_BerkeleyEarth,
        'MLOST': load_gridded_data.read_gridded_data_MLOST,
        'ERA20-20C': load_gridded_data.read_gridded_data_CMIP5_model,
        '20CR-V2': load_gridded_data.read_gridded_data_CMIP5_model,
        'GPCC': load_gridded_data.read_gridded_data_GPCC,
        'PREC': get_env_vars,
        '20CR-V2C': get_env_vars,
    }

    calib_vars = {
        'GISTEMP': ['Tsfc'],
        'HadCRUT': ['Tsfc'],
        'BerkeleyEarth': ['Tsfc'],
        'MLOST': ['air'],
        'ERA20-20C': {'tas_sfc_Amon': 'anom'},
        '20CR-V2': {'tas_sfc_Amon': 'anom'},
        'GPCC': ['precip'],
        'PREC': 'precip',
        '20CR-V2C': 'precip',
    }

    inst_field = {}
    inst_lat = {}
    inst_lon = {}
    inst_gm = {}
    inst_nhm = {}
    inst_shm = {}
    inst_time = {}

    for name, path in ana_pathdict.items():
        print(f'Loading {name}: {path} ...')
        if name in ['ERA20-20C', '20CR-V2']:
            # load_gridded_data.read_gridded_data_CMIP5_model
            dd = load_func[name](
                os.path.dirname(path),
                os.path.basename(path),
                calib_vars[name],
                outtimeavg=avgInterval,
                anom_ref=ref_period,
            )
            time_grid = dd['tas_sfc_Amon']['years']
            lat_grid = dd['tas_sfc_Amon']['lat'][:, 0]
            lon_grid = dd['tas_sfc_Amon']['lon'][0, :]
            anomaly_grid = dd['tas_sfc_Amon']['value']

        elif name == 'GPCC':
            # load_gridded_data.read_gridded_data_GPCC
            time_grid, lat_grid, lon_grid, anomaly_grid = load_func[name](
                os.path.dirname(path),
                os.path.basename(path),
                calib_vars[name],
                True,
                ref_period,
                avgInterval,
            )

        elif name == 'PREC':
            # get_env_vars
            lat_grid, lon_grid, time_grid, vars_grid = load_func[name](
                {'precip': path},
                calc_anomaly=True,
                ref_period=ref_period,
            )
            prate = vars_grid['precip']/24/3600  # convert from mm/day to kg/m^2/s
            if avgInterval == list(range(1, 13)):
                anomaly_grid, time_grid = annualize_var(vars_grid['prate'], time_grid)
            else:
                anomaly_grid, time_grid = seasonal_var(vars_grid['prate'], time_grid, avgMonths=avgInterval)
            time_grid = year_float2datetime(time_grid)

        elif name == '20CR-V2C':
            # get_env_vars
            lat_grid, lon_grid, time_grid, vars_grid = load_func[name](
                {'prate': path},
                calc_anomaly=True,
                ref_period=ref_period,
            )
            if avgInterval == list(range(1, 13)):
                anomaly_grid, time_grid = annualize_var(vars_grid['prate'], time_grid)
            else:
                anomaly_grid, time_grid = seasonal_var(vars_grid['prate'], time_grid, avgMonths=avgInterval)

            time_grid = year_float2datetime(time_grid)

        else:
            # GISTEMP, MLOST, HadCRUT, BerkeleyEarth
            if avgInterval == list(range(1, 13)):
                outfreq = 'annual'
            else:
                outfreq = 'monthly'

            time_grid, lat_grid, lon_grid, anomaly_grid = load_func[name](
                os.path.dirname(path),
                os.path.basename(path),
                calib_vars[name],
                outfreq=outfreq,
                ref_period=ref_period,
            )
            year_float = datetime2year_float(time_grid)

            if outfreq == 'monthly':
                anomaly_grid, time_grid = seasonal_var(anomaly_grid, year_float, avgMonths=avgInterval)
                time_grid = year_float2datetime(time_grid)


        if sort_lon:
            anomaly_grid, lon_grid = rotate_lon(anomaly_grid, lon_grid)

        gm, nhm, shm = global_hemispheric_means(anomaly_grid, lat_grid)
        year = np.array([t.year for t in time_grid])
        month = np.array([t.month for t in time_grid])
        day = np.array([t.day for t in time_grid])
        year_float = ymd2year_float(year, month, day)

        if verif_yrs is not None:
            syear, eyear = verif_yrs[0], verif_yrs[-1]
            year_int = np.array([int(y) for y in year_float])
            mask = (year_int >= syear) & (year_int <= eyear)

            inst_gm[name] = gm[mask]
            inst_nhm[name] = nhm[mask]
            inst_shm[name] = shm[mask]
            inst_time[name] = year_float[mask]
            inst_field[name] = anomaly_grid[mask]
        else:
            inst_gm[name] = gm
            inst_nhm[name] = nhm
            inst_shm[name] = shm
            inst_time[name] = year_float
            inst_field[name] = anomaly_grid

        inst_lat[name] = lat_grid
        inst_lon[name] = lon_grid

    if var == 'field':
        return inst_field, inst_time, inst_lat, inst_lon
    elif var == 'gm':
        return inst_gm, inst_time
    elif var == 'nhm':
        return inst_nhm, inst_time
    elif var == 'shm':
        return inst_shm, inst_time


def calc_field_inst_corr_ce(exp_dir, ana_pathdict, verif_yrs=np.arange(1880, 2000), ref_period=[1951, 1980], field='LMR',
                            valid_frac=0.5, var_name='tas_sfc_Amon', avgInterval=list(range(1, 13)), detrend=False, detrend_kws={}):
    ''' Calculate corr and CE between LMR and instrumental fields

    Note: The time axis of the LMR field is assumed to fully cover the range of verif_yrs
    '''
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    if field == 'LMR':
        field_em, year, lat, lon = load_field_from_jobs(exp_dir, var=var_name)
    else:
        filepath = exp_dir
        filesdict = {
            var_name: filepath,
        }
        lat, lon, time_model, prior_vars = get_env_vars(filesdict, calc_anomaly=False)
        var_model =  prior_vars[var_name]
        field_em, year = seasonal_var(var_model, time_model, avgMonths=avgInterval)

    syear, eyear = verif_yrs[0], verif_yrs[-1]

    if syear < np.min(year) or eyear > np.max(year):
        raise ValueError(f'ERROR: The time axis of the {field} field is not fully covering the range of verif_yrs!!!')

    mask = (year >= syear) & (year <= eyear)
    mask_ref = (year >= ref_period[0]) & (year <= ref_period[-1])
    field_lmr = field_em[mask] - np.nanmean(field_em[mask_ref, :, :], axis=0)  # remove the mean w.r.t. the ref_period

    inst_field, inst_time, inst_lat, inst_lon = load_inst_analyses(
        ana_pathdict, var='field', verif_yrs=verif_yrs, ref_period=ref_period, avgInterval=avgInterval)

    nlat_lmr = np.size(lat)
    nlon_lmr = np.size(lon)
    specob_lmr = Spharmt(nlon_lmr, nlat_lmr, gridtype='regular', legfunc='computed')

    corr = {}
    ce = {}
    for name in inst_field.keys():
        mask_ref = (inst_time[name] >= ref_period[0]) & (inst_time[name] <= ref_period[-1])
        inst_field[name] -= np.nanmean(inst_field[name][mask_ref, :, :], axis=0)  # remove the mean w.r.t. the ref_period

        print(f'Regridding {field} onto {name} ...')

        nlat_inst = np.size(inst_lat[name])
        nlon_inst = np.size(inst_lon[name])

        corr[name] = np.ndarray((nlat_inst, nlon_inst))
        ce[name] = np.ndarray((nlat_inst, nlon_inst))

        specob_inst = Spharmt(nlon_inst, nlat_inst, gridtype='regular', legfunc='computed')

        overlap_yrs = np.intersect1d(verif_yrs, inst_time[name])
        ind_lmr = np.searchsorted(verif_yrs, overlap_yrs)
        ind_inst = np.searchsorted(inst_time[name], overlap_yrs)

        lmr_on_inst = []
        for i in ind_lmr:
            lmr_on_inst_each_yr = regrid(specob_lmr, specob_inst, field_lmr[i], ntrunc=None, smooth=None)
            lmr_on_inst.append(lmr_on_inst_each_yr)

        lmr_on_inst = np.asarray(lmr_on_inst)

        for i in range(nlat_inst):
            for j in range(nlon_inst):
                ts_inst = inst_field[name][ind_inst, i, j]
                ts_lmr = lmr_on_inst[:, i, j]

                ts_inst_notnan = ts_inst[~np.isnan(ts_inst)]
                ts_lmr_notnan = ts_lmr[~np.isnan(ts_inst)]
                nt = len(ind_inst)
                nt_notnan = np.shape(ts_inst_notnan)[0]

                if detrend:
                    ts_inst_notnan = signal.detrend(ts_inst_notnan, **detrend_kws)
                    ts_lmr_notnan = signal.detrend(ts_lmr_notnan, **detrend_kws)

                if nt_notnan/nt >= valid_frac:

                    corr[name][i, j] = np.corrcoef(ts_inst_notnan, ts_lmr_notnan)[1, 0]
                else:
                    corr[name][i, j] = np.nan

                ce[name][i, j] = coefficient_efficiency(ts_inst, ts_lmr, valid_frac)

    return corr, ce, inst_lat, inst_lon


def calc_field_corr_ce(exp_dir, field_model, time_model, lat_model, lon_model,
                       verif_yrs=np.arange(1880, 2000), ref_period=[1951, 1980],
                       valid_frac=0.5, var_name='tas_sfc_Amon',
                       avgMonths=[1,2,3,4,5,6,7,8,9,10,11,12], detrend=False, detrend_kws={}):
    ''' Calculate corr and CE between LMR and model field

    Note: The time axis of the LMR field is assumed to fully cover the range of verif_yrs
    '''
    if avgMonths is not None:
        field_model, time_model = seasonal_var(field_model, time_model, avgMonths=avgMonths)

    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    field_em, year, lat, lon = load_field_from_jobs(exp_dir, var=var_name)
    syear, eyear = verif_yrs[0], verif_yrs[-1]

    if syear < np.min(year) or eyear > np.max(year):
        raise ValueError('ERROR: The time axis of the LMR field is not fully covering the range of verif_yrs!!!')

    mask = (year >= syear) & (year <= eyear)
    mask_ref = (year >= ref_period[0]) & (year <= ref_period[-1])
    field_lmr = field_em[mask] - np.nanmean(field_em[mask_ref, :, :], axis=0)  # remove the mean w.r.t. the ref_period

    nlat_lmr = np.size(lat)
    nlon_lmr = np.size(lon)
    specob_lmr = Spharmt(nlon_lmr, nlat_lmr, gridtype='regular', legfunc='computed')

    mask_ref = (time_model >= ref_period[0]) & (time_model <= ref_period[-1])
    field_model -= np.nanmean(field_model[mask_ref, :, :], axis=0)  # remove the mean w.r.t. the ref_period

    print(f'Regridding LMR onto model grid ...')

    nlat_model = np.size(lat_model)
    nlon_model = np.size(lon_model)

    corr = np.ndarray((nlat_model, nlon_model))
    ce = np.ndarray((nlat_model, nlon_model))

    specob_model = Spharmt(nlon_model, nlat_model, gridtype='regular', legfunc='computed')

    overlap_yrs = np.intersect1d(verif_yrs, time_model)
    ind_lmr = np.searchsorted(verif_yrs, overlap_yrs)
    ind_model = np.searchsorted(time_model, overlap_yrs)

    lmr_on_model = []
    for i in ind_lmr:
        lmr_on_model_each_yr = regrid(specob_lmr, specob_model, field_lmr[i], ntrunc=None, smooth=None)
        lmr_on_model.append(lmr_on_model_each_yr)

    lmr_on_model = np.asarray(lmr_on_model)

    for i in range(nlat_model):
        for j in range(nlon_model):
            ts_model = field_model[ind_model, i, j]
            ts_lmr = lmr_on_model[:, i, j]

            ts_model_notnan = ts_model[~np.isnan(ts_model)]
            ts_lmr_notnan = ts_lmr[~np.isnan(ts_model)]
            nt = len(ind_model)
            nt_notnan = np.shape(ts_model_notnan)[0]

            if detrend:
                ts_model_notnan = signal.detrend(ts_model_notnan, **detrend_kws)
                ts_lmr_notnan = signal.detrend(ts_lmr_notnan, **detrend_kws)

            if nt_notnan/nt >= valid_frac:
                corr[i, j] = np.corrcoef(ts_model_notnan, ts_lmr_notnan)[1, 0]
            else:
                corr[i, j] = np.nan

            ce[i, j] = coefficient_efficiency(ts_model, ts_lmr, valid_frac)

    return corr, ce, lat_model, lon_model


def calc_field_cov(field, lat, lon, year,
                   target_field, target_lat, target_lon, target_year,
                   verif_yrs=np.arange(1880, 2000), verbose=False,
                   npts_lb=25, detrend=False, detrend_kws={}):
    ''' Calculate the correlation map between the field and the timeseries of a target field at the target location

    Args:
        field (ndarray): the field in dims (nt x nlat x nlon)
        lat/lon (array): the lat/lon array of the field
        year (array): the time axis in float
        target_field (ndarray): the target field in dims (nt x nlat x nlon)
        target_lat/lon (float): the target location
        verif_yrs (tuple): the time period to calculate correlation

    Returns:
        corr (ndarray): the correlation map in dims (nlat x nlon)
    '''
    nt, nlat, nlon = np.shape(field)
    lat_ind, lon_ind = find_closest_loc(lat, lon, target_lat, target_lon)
    found_lat, found_lon = lat[lat_ind], lon[lon_ind]
    if verbose:
        print(f'Target: ({target_lat}, {target_lon}); Found: ({found_lat:.2f}, {found_lon:.2f})')

    syear, eyear = verif_yrs[0], verif_yrs[-1]
    if syear < np.min(year) or eyear > np.max(year):
        raise ValueError('ERROR: The time axis of the field is not fully covering the range of verif_yrs!!!')
    mask = (year >= syear) & (year <= eyear)

    corr = np.ndarray((nlat, nlon))
    target_ts = target_field[:, lat_ind, lon_ind]
    target_year, target_ts = clean_ts(target_year, target_ts)

    for i in range(nlat):
        for j in range(nlon):
            ij_ts = field[mask, i, j]
            ij_year, ij_ts = clean_ts(year[mask], ij_ts)

            overlap_yrs = np.intersect1d(ij_year, target_year)
            ind1 = np.searchsorted(ij_year, overlap_yrs)
            ind2 = np.searchsorted(target_year, overlap_yrs)

            if np.size(ind1) < npts_lb or np.size(ind2) < npts_lb:
                print('(i, j) >>> Warning: overlapped timeseries is too short for correlation calculation (npts < 25). Returnning NaN ...')
                corr[i, j] = np.nan
            else:
                ij_ts_overlap = ij_ts[ind1]
                target_ts_overlap = target_ts[ind2]
                if detrend:
                    target_ts_overlap = signal.detrend(target_ts_overlap, **detrend_kws)
                    ij_ts_overlap = signal.detrend(ij_ts_overlap, **detrend_kws)

                corr[i, j] = np.corrcoef(target_ts_overlap, ij_ts_overlap)[1, 0]

    res = {
        'corr': corr,
        'found_lat': found_lat,
        'found_lon': found_lon,
    }
    return res


def calc_corr_between_fields(
    field1, time1, lat1, lon1,
    field2, time2, lat2, lon2,
    verif_yrs=np.arange(1880, 2000),
    verbose=False, detrend=False, detrend_kws={}
):
    syear, eyear = verif_yrs[0], verif_yrs[-1]
    mask1 = (time1 >= syear) & (time1 <= eyear)
    mask2 = (time2 >= syear) & (time2 <= eyear)
    field1_inside = field1[mask1]
    field2_inside = field2[mask2]
    time1_inside = time1[mask1]
    time2_inside = time2[mask2]

    overlap_yrs = np.intersect1d(time1_inside, time2_inside)
    if verbose:
        print(f'Timespan: {np.min(overlap_yrs)} - {np.max(overlap_yrs)}')
    ind1 = np.searchsorted(time1_inside, overlap_yrs)
    ind2 = np.searchsorted(time2_inside, overlap_yrs)

    nlat1 = np.size(lat1)
    nlon1 = np.size(lon1)
    nlat2 = np.size(lat2)
    nlon2 = np.size(lon2)

    if nlat1 == nlat2 and nlon1 == nlon2:
        field1_on_field2 = field1_inside[ind1]
    else:
        specob1 = Spharmt(nlon1, nlat1, gridtype='regular', legfunc='computed')
        specob2 = Spharmt(nlon2, nlat2, gridtype='regular', legfunc='computed')

        field1_on_field2 = []
        for i in ind1:
            field1_on_field2_each_yr = regrid(specob1, specob2, field1_inside[i], ntrunc=None, smooth=None)
            field1_on_field2.append(field1_on_field2_each_yr)

        field1_on_field2 = np.asarray(field1_on_field2)

    corr = np.ndarray((nlat2, nlon2))

    for i in range(nlat2):
        for j in range(nlon2):
            ts1 = field1_on_field2[ind1, i, j]
            ts2 = field2_inside[ind2, i, j]

            if np.isnan(ts1).all() or np.isnan(ts2).all():
                corr[i, j] = np.nan
                continue
            else:
                corr[i, j] = compare_ts(
                    time1_inside[ind1], ts1, time2_inside[ind2], ts2,
                    detrend=detrend, detrend_kws=detrend_kws
                )['corr']

    return corr

# -----------------------------------------------
#  Superposed Epoch Analysis
# -----------------------------------------------
def tsPad(x,t,params=(2,1,2),padFrac=0.1,diag=False):
    """ tsPad: pad a timeseries based on timeseries model predictions

        Args:
        - x: Evenly-spaced timeseries [np.array]
        - t: Time axis  [np.array]
        - params: ARIMA model order parameters (p,d,q)
        - padFrac: padding fraction (scalar) such that padLength = padFrac*length(series)
        - diag: if True, outputs diagnostics of the fitted ARIMA model

        Output:
         - xp, tp, padded timeseries and augmented axis

        Author: Julien Emile-Geay
    """
    padLength =  np.round(len(t)*padFrac).astype(np.int64)

    #if (n-p1 <0)
    #   disp('Timeseries is too short for desired padding')
    #elseif p1 > round(n/5) % Heuristic Bound to ensure AR model stability
    #   p = round(n/5);
    #else
    #   p= p1;
    #end

    if not (np.std(np.diff(t)) == 0):
        raise ValueError("t needs to be composed of even increments")
    else:
        dt = np.diff(t)[0] # computp time interval

    # fit ARIMA model
    fwd_mod = sm.tsa.ARIMA(x,params).fit()  # model with time going forward
    bwd_mod = sm.tsa.ARIMA(np.flip(x,0),params).fit()  # model with time going backwards

    # predict forward & backward
    fwd_pred  = fwd_mod.forecast(padLength); xf = fwd_pred[0]
    bwd_pred  = bwd_mod.forecast(padLength); xb = np.flip(bwd_pred[0],0)

    # define extra time axes
    tf = np.linspace(max(t)+dt, max(t)+padLength*dt,padLength)
    tb = np.linspace(min(t)-padLength*dt, min(t)-1, padLength)

    # extend time series
    tp = np.arange(t[0]-padLength*dt,t[-1]+padLength*dt+1,dt)
    xp = np.empty(len(tp))
    xp[np.isin(tp,t)] =x
    xp[np.isin(tp,tb)]=xb
    xp[np.isin(tp,tf)]=xf

    return xp, tp


def butterworth(x,fc,fs=1, filter_order=3,pad='reflect',reflect_type='odd',params=(2,1,2),padFrac=0.1):
    '''Applies a Butterworth filter with frequency fc, with padding

    Arguments:
        - X = 1d numpy array
        - fc = cutoff frequency. If scalar, it is interpreted as a low-frequency cutoff (lowpass)
                 If fc is a 2-tuple,  it is interpreted as a frequency band (f1, f2), with f1 < f2 (bandpass)
        - fs = sampling frequency
        - filter_order = order n of Butterworth filter
        - pad = boolean indicating whether tsPad needs to be applied
        - params = model parameters for ARIMA model (if pad = True)
        - padFrac = fraction of the series to be padded

    Output : xf, filtered array

    Author: Julien Emile-Geay
    '''
    nyq = 0.5 * fs

    if isinstance(fc, list) and len(fc) == 2:
        fl = fc[0] / nyq
        fh = fc[1] / nyq
        b, a = signal.butter(filter_order, [fl, fh], btype='bandpass')
    else:
        fl = fc / nyq
        b, a = signal.butter(filter_order, fl      , btype='lowpass')

    t = np.arange(len(x)) # define time axis
    padLength =  np.round(len(x)*padFrac).astype(np.int64)

    if (pad=='ARIMA'):
        xp,tp = tsPad(x,t,params=params)
    elif (pad=='reflect'):
        # extend time series
        xp = np.pad(x,(padLength,padLength),mode='reflect',reflect_type=reflect_type)
        tp = np.arange(t[0]-padLength,t[-1]+padLength+1,1)
    else:
        xp = x; tp = t

    xpf = signal.filtfilt(b, a, xp)
    xf  = xpf[np.isin(tp,t)]

    return xf


def sea(X, events, start_yr=0, preyr=3, postyr=10, qs=[0.05, 0.5, 0.95], highpass=False, verbose=False):
    '''Applies superposed Epoch Analysis to N-dim array X, at indices 'events',
        and on a window [-preyr,postyr]
    Inputs:
        - X: numpy array [time assumed to be the first dimension]
        - events: indices of events of interest
        - start_yr (int): the start year of X
        - preyr: # years over which the pre-event mean is computed
        - postyr: length of post-event window

    Outputs:
        - Xevents : X lined up on events; removes mean of "preyr" years in shape of (time, ensemble, events)
        - Xcomp  : composite of Xevents (same dimensions as X, minus the last one)
        - tcomp  : the time axis relative to events

    by Julien Emile-Geay
    '''

    events = np.array(events)

    # exception handling : the first extreme year must not happen within the "preyr" indices
    if any(np.isin(events,np.arange(0,preyr)+start_yr)) or any(events+postyr>=X.shape[0]+start_yr):
        print("event outside range (either before 'tmin-preyr' or after 'tmax+postyr')")
        sys.exit()

    tcomp = np.arange(-preyr,postyr+1) # time axis
    # reshape X to 2d
    if highpass:
        # high-pass filter X along first axis
        fc = 1/len(tcomp)
        sh = list(X.shape)
        Xr = np.reshape(X,(sh[0],np.prod(sh[1:])))
        Xr_hp = np.empty_like(Xr)
        ncols = Xr.shape[1]
        for k in range(ncols):
            Xlp = butterworth(Xr[:, k], fc)
            Xr_hp[:,k] = Xr[:, k] - Xlp

        Xhp = np.reshape(Xr_hp,sh)

    else:
        Xhp = X

    n_events = len(events)
    # define array shape
    sh = list(Xhp.shape)
    sh.append(n_events) # add number of events
    sh[0] = preyr+postyr+1  # replace time axis by time relative to window
    Xevents = np.empty(sh) # define empty array to hold the result


    for i in range(n_events):
        Xevents[...,i] = Xhp[events[i]-preyr-start_yr:events[i]+postyr+1-start_yr,...]
        Xevents[...,i] -= np.mean(Xevents[0:preyr,...,i],axis=0) # remove mean over "preyr" of window

    if verbose:
        print('SEA >>> shape(Xevents):', np.shape(Xevents))

    Xcomp = np.mean(Xevents, axis=-1) # compute composite

    composite = Xcomp.T
    ndim = len(np.shape(composite))
    if ndim > 1:
        composite_qs = mquantiles(composite, qs, axis=0)

        res = {
            'events': events,
            'composite': composite,
            'qs': qs,
            'composite_qs': composite_qs,
            'composite_yr': tcomp,
        }
    else:
        res = {
            'events': events,
            'composite': composite,
            'composite_yr': tcomp,
        }

    if verbose:
        print(f'SEA >>> shape(composite): {np.shape(composite)}')
        print(f'SEA >>> res.keys(): {list(res.keys())}')

    return res


def sea_dbl(time, value, events, preyr=5, postyr=15, seeds=None, nsample=10,
            qs=[0.05, 0.5, 0.95], qs_signif=[0.01, 0.05, 0.10, 0.90, 0.95, 0.99],
            nboot_event=1000, verbose=False):
    ''' A double bootstrap approach to Superposed Epoch Analysis to evaluate response uncertainty

    Args:
        time (1-D array): time axis
        value (1-D array): value axis
        events (1-D array): event years

    Returns:
        res (dict): result dictionary

    References:
        Rao MP, Cook ER, Cook BI, et al (2019) A double bootstrap approach to Superposed Epoch Analysis to evaluate response uncertainty.
            Dendrochronologia 55:119–124. doi: 10.1016/j.dendro.2019.05.001
    '''
    if type(events) is list:
        events = np.array(events)

    nevents = np.size(events)
    total_draws = factorial(nevents)/factorial(nsample)/factorial(nevents-nsample)
    nyr = preyr + postyr + 1

    # embed()
    # avoid edges
    time_inner = time[preyr:-postyr]
    events_inner = events[(events>=np.min(time_inner)) & (events<=np.max(time_inner))]

    if verbose:
        print(f'SEA >>> valid events: {events_inner}')
        print(f'SEA >>> nevents: {nevents}, nsample: {nsample}, total draws: {total_draws:g}')
        print(f'SEA >>> nboot_event: {nboot_event}')
        print(f'SEA >>> preyr: {preyr}, postyr: {postyr}, window length: {nyr}')
        print(f'SEA >>> qs: {qs}, qs_signif: {qs_signif}')

    # generate unique draws without replacement
    draws = []
    draws_signif = []

    for i in range(nboot_event):
        if seeds is not None:
            np.random.seed(seeds[i])

        draw_tmp = np.random.choice(events_inner, nsample, replace=False)
        draws.append(np.sort(draw_tmp))

        draw_tmp = np.random.choice(time_inner, nsample, replace=False)
        draws_signif.append(np.sort(draw_tmp))

    draws = np.array(draws)
    draws_signif = np.array(draws_signif)

    # generate composite ndarrays
    composite_raw = np.ndarray((nboot_event, nsample, nyr))
    composite_raw_signif = np.ndarray((nboot_event, nsample, nyr))

    for i in range(nboot_event):
        sample_yrs = draws[i]
        sample_yrs_signif = draws_signif[i]

        for j in range(nsample):
            center_yr = list(time).index(sample_yrs[j])
            composite_raw[i, j, :] = value[center_yr-preyr:center_yr+postyr+1]

            center_yr_signif = list(time).index(sample_yrs_signif[j])
            composite_raw_signif[i, j, :] = value[center_yr_signif-preyr:center_yr_signif+postyr+1]

    # normalization: remove the mean of the pre-years
    composite_norm = composite_raw - np.average(composite_raw[:, :, :preyr], axis=-1)[:, :, np.newaxis]
    composite_norm_signif = composite_raw_signif - np.average(composite_raw_signif[:, :, :preyr], axis=-1)[:, :, np.newaxis]

    composite = np.average(composite_norm, axis=1)
    composite_qs = mquantiles(composite, qs, axis=0)

    composite_signif = np.average(composite_norm_signif, axis=1)
    composite_qs_signif = mquantiles(composite_signif, qs_signif, axis=0)

    composite_yr = np.arange(-preyr, postyr+1)

    res = {
        'events': events_inner,
        'draws': draws,
        'composite': composite,
        'qs': qs,
        'composite_qs': composite_qs,
        'draws_signif': draws_signif,
        'composite_signif': composite_signif,
        'qs_signif': qs_signif,
        'composite_qs_signif': composite_qs_signif,
        'composite_yr': composite_yr,
    }

    if verbose:
        print(f'SEA >>> shape(composite): {np.shape(composite)}')
        print(f'SEA >>> res.keys(): {list(res.keys())}')

    return res


def sea_field(time, field, events, preyr=5, post_avg_range=[0]):
    ''' Perform a simple SEA on a 3-D field

    Args:
        post_avg_range (list): e.g. [0] refers to the event year; [1, 5] refers to the 1-5 yrs after the event

    '''
    def post_avg_func(field, i, post_avg_range):
        if len(post_avg_range) == 1:
            return field[i+post_avg_range[0]]
        else:
            return np.average(field[i+post_avg_range[0]:i+post_avg_range[-1]+1], axis=0)

    idx = np.array([list(time).index(t) for t in events])
    field_anom = []
    for i in idx:
        pre_avg = np.average(field[i-preyr:i], axis=0)
        post_avg = post_avg_func(field, i, post_avg_range)
        field_anom_tmp = post_avg - pre_avg
        field_anom.append(field_anom_tmp)

    composite = np.average(np.array(field_anom), axis=0)

    return composite


def sea_dbl_field(time, field, events, preyr=5, post_avg_range=[0], seeds=None, nsample=10,
            qs=[5, 50, 95], qs_signif=[1, 5, 10, 90, 95, 99],
            nboot_event=1000, verbose=False):
    ''' A double bootstrap approach to Superposed Epoch Analysis to evaluate response uncertainty

    Args:
        time (1-D array): time axis
        field (3-D array): filed with 1st dim be time
        events (1-D array): event years

    Returns:
        res (dict): result dictionary

    References:
        Rao MP, Cook ER, Cook BI, et al (2019) A double bootstrap approach to Superposed Epoch Analysis to evaluate response uncertainty.
            Dendrochronologia 55:119–124. doi: 10.1016/j.dendro.2019.05.001
    '''
    if type(events) is list:
        events = np.array(events)

    nevents = np.size(events)
    total_draws = factorial(nevents)/factorial(nsample)/factorial(nevents-nsample)

    # avoid edges
    if post_avg_range[-1] == 0:
        time_inner = time[preyr:]
    else:
        time_inner = time[preyr:-post_avg_range[-1]]

    events_inner = events[(events>=np.min(time_inner)) & (events<=np.max(time_inner))]

    if verbose:
        print(f'SEA >>> valid events: {events_inner}')
        print(f'SEA >>> nevents: {nevents}, nsample: {nsample}, total draws: {total_draws:g}')
        print(f'SEA >>> nboot_event: {nboot_event}')
        print(f'SEA >>> preyr: {preyr}, post_avg_range: {post_avg_range}')
        print(f'SEA >>> qs: {qs}, qs_signif: {qs_signif}')

    # generate unique draws without replacement
    draws = []
    draws_signif = []

    for i in range(nboot_event):
        if seeds is not None:
            np.random.seed(seeds[i])

        draw_tmp = np.random.choice(events_inner, nsample, replace=False)
        draws.append(np.sort(draw_tmp))

        draw_tmp = np.random.choice(time_inner, nsample, replace=False)
        draws_signif.append(np.sort(draw_tmp))

    draws = np.array(draws)
    draws_signif = np.array(draws_signif)

    # calculate composites
    composite = []
    composite_signif = []

    for i in range(nboot_event):
        sample_yrs = draws[i]
        sample_yrs_signif = draws_signif[i]

        composite.append(
            sea_field(time, field, sample_yrs, preyr=preyr, post_avg_range=post_avg_range)
        )
        composite_signif.append(
            sea_field(time, field, sample_yrs_signif, preyr=preyr, post_avg_range=post_avg_range)
        )

    composite = np.array(composite)
    composite_qs = np.percentile(composite, qs, axis=0)

    composite_signif = np.array(composite_signif)
    composite_qs_signif = np.percentile(composite_signif, qs_signif, axis=0)

    # return results
    res = {
        'events': events_inner,
        'draws': draws,
        'composite': composite,
        'qs': qs,
        'composite_qs': composite_qs,
        'draws_signif': draws_signif,
        'composite_signif': composite_signif,
        'qs_signif': qs_signif,
        'composite_qs_signif': composite_qs_signif,
    }

    if verbose:
        print(f'SEA >>> shape(composite): {np.shape(composite)}')
        print(f'SEA >>> res.keys(): {list(res.keys())}')

    return res
# ===============================================

# -----------------------------------------------
#  corr_sig adapated from Pyleoclim
# -----------------------------------------------
def corr_sig(y1, y2, nsim=1000, method='isospectral', alpha=0.05):
    """ Estimates the significance of correlations between non IID time series by 3 independent methods:
    1) 'ttest': T-test where d.o.f are corrected for the effect of serial correlation
    2) 'isopersistent': AR(1) modeling of x and y.
    3) 'isospectral': phase randomization of original inputs. (default)
    The T-test is parametric test, hence cheap but usually wrong except in idyllic circumstances.
    The others are non-parametric, but their computational requirements scales with nsim.

    Args:
        y1, y2 (array)- vector of (real) numbers of identical length, no NaNs allowed
        nsim (int)- the number of simulations [1000]
        method (str)- methods 1-3 above ['isospectral']
        alpha (float)- significance level for critical value estimation [0.05]

    Returns:
         r (real): correlation between x and y \n
         signif (boolean): true (1) if significant; false (0) otherwise \n
         p (real): Fraction of time series with higher correlation coefficents than observed (approximates the p-value). \n
            Note that signif = True if and only if p <= alpha.
    """
    y1 = np.array(y1, dtype=float)
    y2 = np.array(y2, dtype=float)

    assert np.size(y1) == np.size(y2), 'The size of X and the size of Y should be the same!'

    if method == 'ttest':
        (r, signif, p) = corr_ttest(y1, y2, alpha=alpha)
    elif method == 'isopersistent':
        (r, signif, p) = corr_isopersist(y1, y2, alpha=alpha, nsim=nsim)
    elif method == 'isospectral':
        (r, signif, p) = corr_isospec(y1, y2, alpha=alpha, nsim=nsim)
    else:
        raise KeyError(f'{method} is not a valid method')

    return r, signif, p


def corr_ttest(y1, y2, alpha=0.05):
    """ Estimates the significance of correlations between 2 time series using
    the classical T-test with degrees of freedom modified for autocorrelation.
    This function creates 'nsim' random time series that have the same power
    spectrum as the original time series but with random phases.

    Args:
        y1, y2 (array): vectors of (real) numbers with identical length, no NaNs allowed
        alpha (real): significance level for critical value estimation [default: 0.05]

    Returns:
        r (real)- correlation between x and y \n
        signif (boolean)- true (1) if significant; false (0) otherwise \n
        pval (real)- test p-value (the probability of the test statstic exceeding the observed one by chance alone)
    """
    r = pearsonr(y1, y2)[0]

    g1 = ar1_fit(y1)
    g2 = ar1_fit(y2)

    N = np.size(y1)

    Ney1 = N * (1-g1) / (1+g1)
    Ney2 = N * (1-g2) / (1+g2)

    Ne = gmean([Ney1+Ney2])
    assert Ne >= 10, f'Too few effective d.o.f. to apply this method! Ne={Ne}, Ney1={Ney1:.2f}, Ney2={Ney2:.2f}, g1={g1:.2f}, g2={g2:.2f}'

    df = Ne - 2
    t = np.abs(r) * np.sqrt(df/(1-r**2))

    pval = 2 * stu.cdf(-np.abs(t), df)

    signif = pval <= alpha

    return r, signif, pval


def corr_isopersist(y1, y2, alpha=0.05, nsim=1000):
    ''' Computes correlation between two timeseries, and their significance.
    The latter is gauged via a non-parametric (Monte Carlo) simulation of
    correlations with nsim AR(1) processes with identical persistence
    properties as x and y ; the measure of which is the lag-1 autocorrelation (g).

    Args:
        y1, y2 (array): vectors of (real) numbers with identical length, no NaNs allowed
        alpha (real): significance level for critical value estimation [default: 0.05]
        nsim (int): number of simulations [default: 1000]

    Returns:
        r (real) - correlation between x and y \n
        signif (boolean) - true (1) if significant; false (0) otherwise \n
        pval (real) - test p-value (the probability of the test statstic exceeding the observed one by chance alone)

    Remarks:
        The probability of obtaining a test statistic at least as extreme as the one actually observed,
        assuming that the null hypothesis is true. \n
        The test is 1 tailed on |r|: Ho = { |r| = 0 }, Ha = { |r| > 0 } \n
        The test is rejected (signif = 1) if pval <= alpha, otherwise signif=0; \n
        (Some Rights Reserved) Hepta Technologies, 2009 \n
        v1.0 USC, Aug 10 2012, based on corr_signif.m
    '''

    r = pearsonr(y1, y2)[0]
    ra = np.abs(r)

    y1_red, g1 = isopersistent_rn(y1, nsim)
    y2_red, g2 = isopersistent_rn(y2, nsim)

    rs = np.zeros(nsim)
    for i in np.arange(nsim):
        rs[i] = pearsonr(y1_red[:, i], y2_red[:, i])[0]

    rsa = np.abs(rs)

    xi = np.linspace(0, 1.1*np.max([ra, np.max(rsa)]), 200)
    kde = gaussian_kde(rsa)
    prob = kde(xi).T

    diff = np.abs(ra - xi)
    #  min_diff = np.min(diff)
    pos = np.argmin(diff)

    pval = np.trapz(prob[pos:], xi[pos:])

    rcrit = np.percentile(rsa, 100*(1-alpha))
    signif = ra >= rcrit

    return r, signif, pval


def corr_isospec(y1, y2, alpha=0.05, nsim=1000):
    ''' Phase randomization correltation estimates

    Estimates the significance of correlations between non IID
    time series by phase randomization of original inputs.
    This function creates 'nsim' random time series that have the same power
    spectrum as the original time series but random phases.

    Args:
        y1, y2 (array): vectors of (real) numbers with identical length, no NaNs allowed
        alpha (real): significance level for critical value estimation [default: 0.05]
        nsim (int): number of simulations [default: 1000]

    Returns:
        r (real): correlation between y1 and y2 \n
        signif (boolean): true (1) if significant; false (0) otherwise \n
        F : Fraction of time series with higher correlation coefficents than observed (approximates the p-value).

    References:
        - Ebisuzaki, W, 1997: A method to estimate the statistical
        significance of a correlation when the data are serially correlated.
        J. of Climate, 10, 2147-2153.
        - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
        with Several Simultaneously Measured Variables (1994)
        Physical Review Letters, Vol 73, Number 7
        (Some Rights Reserved) USC Climate Dynamics Lab, 2012.
    '''
    r = pearsonr(y1, y2)[0]

    # generate phase-randomized samples using the Theiler & Prichard method
    Y1surr = phaseran(y1, nsim)
    Y2surr = phaseran(y2, nsim)

    # compute correlations
    Y1s = preprocessing.scale(Y1surr)
    Y2s = preprocessing.scale(Y2surr)

    n = np.size(y1)
    C = np.dot(np.transpose(Y1s), Y2s) / (n-1)
    rSim = np.diag(C)

    # compute fraction of values higher than observed
    F = np.sum(np.abs(rSim) >= np.abs(r)) / nsim

    # establish significance
    signif = F < alpha  # significant or not?

    return r, signif, F


def isopersistent_rn(X, p):
    ''' Generates p realization of a red noise [i.e. AR(1)] process
    with same persistence properties as X (Mean and variance are also preserved).

    Args:
        X (array): vector of (real) numbers as a time series, no NaNs allowed
        p (int): number of simulations

    Returns:
        red (matrix) - n rows by p columns matrix of an AR1 process, where n is the size of X \n
        g (real) - lag-1 autocorrelation coefficient

    Remarks:
        (Some Rights Reserved) Hepta Technologies, 2008
    '''
    n = np.size(X)
    sig = np.std(X, ddof=1)

    g = ar1_fit(X)
    red = ar1_sim(n, p, g, sig)

    return red, g


def ar1_fit(ts):
    ''' Return the lag-1 autocorrelation from ar1 fit.

    Args:
        ts (array): vector of (real) numbers as a time series

    Returns:
        g (real): lag-1 autocorrelation coefficient
    '''


    ar1_mod = sm.tsa.AR(ts, missing='drop').fit(maxlag=1)
    g = ar1_mod.params[1]

    return g


def ar1_sim(n, p, g, sig):
    ''' Produce p realizations of an AR1 process of length n with lag-1 autocorrelation g

    Args:
        n, p (int): dimensions as n rows by p columns
        g (real): lag-1 autocorrelation coefficient
        sig (real): the standard deviation of the original time series

    Returns:
        red (matrix): n rows by p columns matrix of an AR1 process
    '''
    # specify model parameters (statsmodel wants lag0 coefficents as unity)
    ar = np.r_[1, -g]  # AR model parameter
    ma = np.r_[1, 0.0] # MA model parameters
    sig_n = sig*np.sqrt(1-g**2) # theoretical noise variance for red to achieve the same variance as X

    red = np.empty(shape=(n, p)) # declare array

    # simulate AR(1) model for each column
    for i in np.arange(p):
        red[:, i] = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=50, sigma=sig_n)

    return red


def phaseran(recblk, nsurr):
    ''' Phaseran by Carlos Gias

    http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m

    Args:
        recblk (2D array): Row: time sample. Column: recording.
            An odd number of time samples (height) is expected.
            If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
            The class must be double and it must be nonsparse.
        nsurr (int): is the number of image block surrogates that you want to generate.

    Returns:
        surrblk: 3D multidimensional array image block with the surrogate datasets along the third dimension

    Reference:
        Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994)
        Physical Review Letters, Vol 73, Number 7
    '''
    # Get parameters
    nfrms = recblk.shape[0]

    if nfrms % 2 == 0:
        nfrms = nfrms-1
        recblk = recblk[0:nfrms]

    len_ser = int((nfrms-1)/2)
    interv1 = np.arange(1, len_ser+1)
    interv2 = np.arange(len_ser+1, nfrms)

    # Fourier transform of the original dataset
    fft_recblk = np.fft.fft(recblk)

    surrblk = np.zeros((nfrms, nsurr))

    #  for k in tqdm(np.arange(nsurr)):
    for k in np.arange(nsurr):
        ph_rnd = np.random.rand(len_ser)

        # Create the random phases for all the time series
        ph_interv1 = np.exp(2*np.pi*1j*ph_rnd)
        ph_interv2 = np.conj(np.flipud(ph_interv1))

        # Randomize all the time series simultaneously
        fft_recblk_surr = np.copy(fft_recblk)
        fft_recblk_surr[interv1] = fft_recblk[interv1] * ph_interv1
        fft_recblk_surr[interv2] = fft_recblk[interv2] * ph_interv2

        # Inverse transform
        surrblk[:, k] = np.real(np.fft.ifft(fft_recblk_surr))

    return surrblk

# -----------------------------------------------
#  filters
# -----------------------------------------------
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.sosfilt(sos, data)
    return y
