from datetime import datetime, timedelta
import cftime
import numpy as np
import pprint
pp = pprint.PrettyPrinter()
from termcolor import cprint
import os
from spharm import Spharmt, regrid
import pandas as pd
from scipy import spatial
from scipy.special import factorial
from scipy.stats.mstats import mquantiles
import scipy.stats as ss
from tqdm import tqdm
import xarray as xr

def p_header(text):
    return cprint(text, 'cyan', attrs=['bold'])

def p_hint(text):
    return cprint(text, 'grey', attrs=['bold'])

def p_success(text):
    return cprint(text, 'green', attrs=['bold'])

def p_fail(text):
    return cprint(text, 'red', attrs=['bold'])

def p_warning(text):
    return cprint(text, 'yellow', attrs=['bold'])

def cfg_abspath_str(cfg_path, path_str_in_cfg):
    ''' Convert a path in a config YAML file to an abs path.
    '''
    if not os.path.isabs(path_str_in_cfg):
        cfg_dirpath = os.path.dirname(cfg_path)
        new_path = os.path.abspath(os.path.join(cfg_dirpath, path_str_in_cfg))
    else:
        new_path = path_str_in_cfg
    return new_path

def cfg_abspath_dict(cfg_path, path_dict_in_cfg):
    ''' Convert a CWD path to an abs path.
    '''
    new_dict = {}
    for k, v in path_dict_in_cfg.items():
        new_dict[k] = cfg_abspath_str(cfg_path, v)

    return new_dict

def cfg_abspath(cfg_path, path_in_cfg):
    if isinstance(path_in_cfg, str):
        new_path = cfg_abspath_str(cfg_path, path_in_cfg)
    elif isinstance(path_in_cfg, dict):
        new_path = cfg_abspath_dict(cfg_path, path_in_cfg)
    else:
        raise ValueError('Wrong type for path_in_cfg: must be {str, dict}')

    return new_path

def cwd_abspath(path):
    cwd = os.getcwd()
    # try to load from given path
    if isinstance(path, str):
        new_path = os.path.abspath(os.path.join(cwd, path))
    elif isinstance(path, dict):
        new_path_dict = {}
        for k, v in path.items():
            new_path_dict[k] = os.path.abspath(os.path.join(cwd, v))

        new_path = new_path_dict

    return new_path

def clean_ts(ts, ys):
    ''' Delete the NaNs in the time series and sort it with time axis ascending

    Parameters
    ----------
    ts : array
        The time axis of the time series, NaNs allowed
    ys : array
        A time series, NaNs allowed

    Returns
    -------
    ts : array
        The time axis of the time series without NaNs
    ys : array
        The time series without NaNs
    '''
    ys = np.asarray(ys, dtype=np.float)
    ts = np.asarray(ts, dtype=np.float)
    assert ys.size == ts.size, 'The size of time axis and data value should be equal!'

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

    # handle duplicated time points
    t_count = {}
    value_at_t = {}
    for i, t in enumerate(ts):
        if t not in t_count:
            t_count[t] = 1
            value_at_t[t] = ys[i]
        else:
            t_count[t] += 1
            value_at_t[t] += ys[i]

    ys = []
    ts = []
    for t, v in value_at_t.items():
        ys.append(v / t_count[t])
        ts.append(t)

    ts = np.array(ts)
    ys = np.array(ys)

    return ts, ys

def dropna_field(time, field):
    ''' Drop the time if the field has all NaNs; field should have dimensions (time, lat, lon)
    '''
    t_keep = []
    fd_keep = []
    for idx, t in enumerate(time):
        if not np.all(np.isnan(field[idx])):
            t_keep.append(t)
            fd_keep.append(field[idx])
    
    t_keep = np.array(t_keep)
    fd_keep = np.array(fd_keep)
    return t_keep, fd_keep


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
    ''' Convert a list of dates to floats in year
    '''
    if isinstance(date[0], np.datetime64):
        date = pd.to_datetime(date)

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

def seasonal_var(year_float, var, resolution='month',
                 avgMonths=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], make_yr_mm_nan=True,
                 verbose=False):
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

    time = year_float2datetime(year_float, resolution=resolution)

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
        if verbose:
            print('nbcyears', nbcyears)
            print('cyears[i]', cyears[i])
            print('indsyrm1', indsyrm1)
            print('indsyr', indsyr)
            print('indsyrp1', indsyrp1)
            print('inds', inds)

        if ndim == 1:
            tmp = np.nanmean(var[inds], axis=0)
            #  nancount = np.isnan(var[inds]).sum(axis=0)
            #  if nancount > 0:
                #  tmp = np.nan
        else:
            tmp = np.nanmean(var[inds, ...], axis=0)
            #  nancount = np.isnan(var[inds, ...]).sum(axis=0)
            #  tmp[nancount > 0] = np.nan

        if make_yr_mm_nan and len(inds) != nbmonths:
            tmp = np.nan

        var_ann[i, ...] = tmp

    return year_ann, var_ann

def regrid_field(field, lat, lon, lat_new, lon_new):
    nlat_old, nlon_old = np.size(lat), np.size(lon)
    nlat_new, nlon_new = np.size(lat_new), np.size(lon_new)
    spec_old = Spharmt(nlon_old, nlat_old, gridtype='regular', legfunc='computed')
    spec_new = Spharmt(nlon_new, nlat_new, gridtype='regular', legfunc='computed')

    field_new = []
    for field_old in field:
        regridded_field =  regrid(spec_old, spec_new, field_old, ntrunc=None, smooth=None)
        field_new.append(regridded_field)

    field_new = np.array(field_new)
    return field_new

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

    if mode == 'latlon':
        # model locations
        mesh = np.meshgrid(lon, lat)

        list_of_grids = list(zip(*(grid.flat for grid in mesh)))
        model_lon, model_lat = zip(*list_of_grids)

    elif mode == 'mesh':
        model_lat = lat.flatten()
        model_lon = lon.flatten()

    elif mode == 'list':
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
    for i, target_loc in (enumerate(tqdm(target_locations, desc='Searching nearest location')) if verbose else enumerate(target_locations)):
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

    # West Pacific index
    lat_mask['wpi'] = (lats >= -10) & (lats <= 10)
    lon_mask['wpi'] = (lons >= lon360(120)) & (lons <= lon360(150))

    for region in lat_mask.keys():
        sst_sub = sst[..., lon_mask[region]]
        sst_sub = sst_sub[..., lat_mask[region], :]
        ma = np.ma.MaskedArray(sst_sub, mask=np.isnan(sst_sub))
        ind[region] = np.average(
            np.average(ma, axis=-1),
            axis=-1,
            weights=np.cos(np.deg2rad(lats[lat_mask[region]])),
        )
    return ind


def calc_tpi(sst, lats, lons):
    ''' Calculate tripole index

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

def global_hemispheric_means(field, lat):
    """ Adapted from LMR_utils.py by Greg Hakim & Robert Tardif | U. of Washington

     compute global and hemispheric mean valuee for all times in the input (i.e. field) array
     Args:
        field[ntime,nlat,nlon] or field[nlat,nlon]
        lat[nlat,nlon] in degrees

     Retruns:
        gm : global mean of "field"
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

def geo_area_mean(field_value, field_lat, field_lon, lat_min, lat_max, lon_min, lon_max):
    def lon360(lon):
        # convert from (-180, 180) to (0, 360)
        return np.mod(lon, 360)

    lats = np.array(field_lat)
    lons = np.array(field_lon)
    if any(np.diff(lons)) < 0:
        field_value, lons = rotate_lon(field_value, lons)

    lat_mask = {}
    lon_mask = {}
    ind = {}

    if lat_max - lat_min == 90:
        lat_mask = np.ones(np.size(lats), dtype=int)
    else:
        lat_mask = (lats >= lat_min) & (lats <= lat_max)

    if lon_max - lon_min == 360:
        lon_mask = np.ones(np.size(lons), dtype=int)
    else:
        lon_mask = (lon360(lons) >= lon360(lon_min)) & (lon360(lons) <= lon360(lon_max))

    field_sub = field_value[..., lon_mask]
    field_sub = field_sub[..., lat_mask, :]
    field_area_mean = np.average(
        np.average(field_sub, axis=-1),
        axis=-1,
        weights=np.cos(np.deg2rad(lats[lat_mask])),
    )

    return field_area_mean

def geo_mean(field_value, field_lat, field_lon, lats, lons):
    ''' Calculate the average value of the given field over a list of lat/lon locations
    '''
    value_list = []
    weight_list = []

    for lat, lon in tqdm(zip(lats, lons), total=len(lats)):
        lat_ind, lon_ind = find_closest_loc(field_lat, field_lon, lat, lon)

        lat_weight = np.cos(np.deg2rad(lat))
        weight_list.append(lat_weight)

        value = field_value[..., lat_ind, lon_ind]
        value_list.append(value)

    value_array = np.array(value_list)
    value_avg = np.average(value_array, axis=0, weights=weight_list)
    return value_avg

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
    # print('dims_test: ', dims_test, ' dims_ref: ', dims_ref)

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

def make_xr(var, year_float):
    var = np.array(var)
    year_float = np.array(year_float)

    ndims = len(np.shape(var))
    dims = ['time']
    for i in range(ndims-1):
        dims.append(f'dim{i+1}')

    time = year_float2datetime(year_float, resolution='day')
    var_da = xr.DataArray(var, dims=dims, coords={'time': time})
    return var_da

def get_anomaly(var, year_float, ref_period=[1951, 1980]):
    var_da = make_xr(var, year_float)

    if ref_period[0] > np.max(year_float) or ref_period[-1] < np.min(year_float):
        print(f'Time axis not overlap with the reference period {ref_period}; use its own time period as reference [{np.min(year_float):.2f}, {np.max(year_float):.2f}].')
        var_ref = var_da
    else:
        var_ref = var_da.loc[str(ref_period[0]):str(ref_period[-1])]

    climatology = var_ref.groupby('time.month').mean('time')
    var_anom = var_da.groupby('time.month') - climatology
    return var_anom.values

def sea_dbl(time, value, events, nonevents=None, preyr=5, postyr=15, seeds=None, nsample=10,
            qs=[0.05, 0.5, 0.95], qs_signif=[0.01, 0.05, 0.10, 0.90, 0.95, 0.99],
            nboot_event=1000, verbose=False, draw_mode='non-events'):
    ''' A double bootstrap approach to Superposed Epoch Analysis to evaluate response uncertainty

    Args:
        time (array): time axis
        value (1-D array or 2-D array): value axis; if 2-D, with time as the 1st dimension
        events (array): event years
        draw_mode ({'all', 'non-events'}): the pool for significance test

    Returns:
        res (dict): result dictionary

    References:
        Rao MP, Cook ER, Cook BI, et al (2019) A double bootstrap approach to Superposed Epoch Analysis to evaluate response uncertainty.
            Dendrochronologia 55:119–124. doi: 10.1016/j.dendro.2019.05.001
    '''
    if type(events) is list:
        events = np.array(events)

    nevents = np.size(events)
    if nsample > nevents:
        print(f'SEA >>> nsample: {nsample} > nevents: {nevents}; setting nsample=nevents: {nevents} ...')
        nsample = nevents

    total_draws = factorial(nevents)/factorial(nsample)/factorial(nevents-nsample)
    nyr = preyr + postyr + 1

    # embed()
    # avoid edges
    time_inner = time[preyr:-postyr]
    events_inner = events[(events>=np.min(time_inner)) & (events<=np.max(time_inner))]
    if draw_mode == 'all':
        signif_pool = list(time_inner)
    elif draw_mode == 'non-events':
        if nonevents is None:
            signif_pool = list(time_inner)
            events_expanded = set()
            for e in events_inner:
                idx = list(time_inner).index(e)
                subset = set(time_inner[idx:idx+postyr+1])
                events_expanded |= subset
            for e in events_expanded:
                signif_pool.remove(e)
        else:
            signif_pool = [e for e in nonevents if e in list(time_inner)]
    else:
        raise ValueError('ERROR: Wrong `draw_mode`; choose between `all` and `non-events`')

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

        draw_tmp = np.random.choice(signif_pool, nsample, replace=False)
        draws_signif.append(np.sort(draw_tmp))

    draws = np.array(draws)
    draws_signif = np.array(draws_signif)

    # generate composite ndarrays
    ndim = len(np.shape(value))
    if ndim == 1:
        value = value[..., np.newaxis]

    nts = np.shape(value)[-1]

    composite_raw = np.ndarray((nboot_event, nsample, nyr, nts))
    composite_raw_signif = np.ndarray((nboot_event, nsample, nyr, nts))

    for i in range(nboot_event):
        sample_yrs = draws[i]
        sample_yrs_signif = draws_signif[i]

        for j in range(nsample):
            center_yr = list(time).index(sample_yrs[j])
            composite_raw[i, j, :, :] = value[center_yr-preyr:center_yr+postyr+1, :]

            center_yr_signif = list(time).index(sample_yrs_signif[j])
            composite_raw_signif[i, j, :, :] = value[center_yr_signif-preyr:center_yr_signif+postyr+1, :]

    # normalization: remove the mean of the pre-years
    composite_norm = composite_raw - np.average(composite_raw[:, :, :preyr, ...], axis=2)[:, :, np.newaxis, :]
    composite_norm_signif = composite_raw_signif - np.average(composite_raw_signif[:, :, :preyr, ...], axis=2)[:, :, np.newaxis, :]

    composite = np.average(composite_norm, axis=1)
    composite = composite.transpose(0, 2, 1).reshape(nboot_event*nts, -1)
    composite_qs = mquantiles(composite, qs, axis=0)

    composite_signif = np.average(composite_norm_signif, axis=1)
    composite_signif = composite_signif.transpose(0, 2, 1).reshape(nboot_event*nts, -1)
    composite_qs_signif = mquantiles(composite_signif, qs_signif, axis=0)

    composite_yr = np.arange(-preyr, postyr+1)

    res = {
        'events': events_inner,
        'draws': draws,
        'composite': composite,
        'composite_norm': composite_norm,
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

def calc_anom(time, value, target_yrs, preyr=5, post_avg_range=[0]):
    ''' Calculate the anomaly for each target_yr (with a post range) with the pre-target average subtracted.
    '''
    def post_avg_func(value, i, post_avg_range):
        if len(post_avg_range) == 1:
            return value[i+post_avg_range[0]]
        else:
            return np.average(value[i+post_avg_range[0]:i+post_avg_range[-1]+1], axis=0)

    anom = []
    for yr in target_yrs:
        i = list(time).index(yr)
        pre_avg = np.average(value[i-preyr:i], axis=0)
        post_avg = post_avg_func(value, i, post_avg_range)
        anom.append(post_avg - pre_avg)

    anom = np.array(anom)
    return anom

def calc_volc_nonvolc_anom(year_all, target_series, year_volc, preyr=3, postyr=6, year_nonvolc=None, post_avg_range=[1],
                           seeds=None, nboot=1000):
    if year_nonvolc is None:
        events_expanded = set()
        year_inner = year_all[preyr:-postyr]
        for e in year_volc:
            idx = list(year_inner).index(e)
            subset = set(year_inner[idx:idx+postyr+1])
            events_expanded |= subset

        year_nonvolc = np.array(list(set(year_inner)-set(events_expanded)))
    else:
        year_inner = year_all[preyr:-postyr]
        year_nonvolc = [e for e in year_nonvolc if e in list(year_inner)]

    ndim = len(np.shape(target_series))

    if ndim == 1:
        anom_volc = calc_anom(year_all, target_series, year_volc, post_avg_range=post_avg_range, preyr=preyr)
        anom_nonvolc = calc_anom(year_all, target_series, year_nonvolc, post_avg_range=post_avg_range, preyr=preyr)
    else:
        anom_volc = []
        anom_nonvolc = []
        for ts in target_series:
            anom_volc.append( calc_anom(year_all, ts, year_volc, post_avg_range=post_avg_range, preyr=preyr) )
            anom_nonvolc.append( calc_anom(year_all, ts, year_nonvolc, post_avg_range=post_avg_range, preyr=preyr) )

        anom_volc = np.array(anom_volc)
        anom_nonvolc = np.array(anom_nonvolc)

    draws = []
    for i in range(nboot):
        if seeds is not None:
            np.random.seed(seeds[i])

        draw_tmp = np.random.choice(year_nonvolc, np.size(year_volc), replace=False)
        draws.append(np.sort(draw_tmp))

    anom_nonvolc_draws = []
    for draw in draws:
        if ndim == 1:
            anom_nonvolc_draws.append(calc_anom(year_all, target_series, draw, post_avg_range=post_avg_range, preyr=preyr))
        else:
            anom_nonvolc_draws.append(calc_anom(year_all, ts, draw, post_avg_range=post_avg_range, preyr=preyr))

    anom_nonvolc_draws = np.array(anom_nonvolc_draws)

    res_dict = {
        'draws': draws,
        'anom_volc': anom_volc,
        'anom_nonvolc': anom_nonvolc,
        'anom_nonvolc_draws': anom_nonvolc_draws,
    }

    return res_dict