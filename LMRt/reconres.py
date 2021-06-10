import pickle
import numpy as np
import random
import glob
import os
from numpy.lib.arraysetops import isin
import pandas as pd
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
import xarray as xr
import inspect
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from . import utils

from .utils import (
    find_closest_loc,
    coefficient_efficiency,
    pp,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)
from .visual import (
    showfig,
    savefig,
    CartopySettings,
)

from .gridded import (
    Field,
)

from .proxy import (
    ProxyDatabase,
    ProxyRecord,
)

from pyleoclim.core.ui import (
    Series,
    EnsembleSeries,
)
class ReconSeries(EnsembleSeries):
    def validate_fd(self, target_item, stat='corr', valid_period=[1880, 2000], corr_method_kws=None, verbose=False):
        ts_array = []
        for ts in self.series_list:
            ts_array.append(ts.value)

        time_lmr = self.series_list[0].time
        value_lmr = np.median(ts_array, axis=0)
        if valid_period[0] < np.min(target_item.time):
            valid_period[0] = np.min(target_item.time)
        if valid_period[1] > np.max(target_item.time):
            valid_period[1] = np.max(target_item.time)
        if verbose: p_header(f'LMRt: res.ReconSeries.validate() >>> valid_period = {valid_period}')

        ts_lmr = Series(time=time_lmr, value=value_lmr).slice(valid_period)

        if corr_method_kws is None: corr_method_kws = {'method': 'isospectral', 'nsim': 100}

        nlat, nlon = np.size(target_item.lat), np.size(target_item.lon)
        stat_array = np.ndarray((1, nlat, nlon))
        for i in tqdm(range(nlat), desc=f'Calculating metric: {stat}'):
            for j in range(nlon):
                ts_ref = Series(time=target_item.time, value=target_item.value[:, i, j]).slice(valid_period)
                if stat == 'corr':
                    corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
                    if type(corr_res) is dict:
                        stat_array[0, i, j] = corr_res['r']
                    else:
                        stat_array[0, i, j] = corr_res.r
                elif stat == 'R2':
                    corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
                    if type(corr_res) is dict:
                        stat_array[0, i, j] = corr_res['r']**2
                    else:
                        stat_array[0, i, j] = corr_res.r**2
                elif stat == 'CE':
                    stat_array[0, i, j] = coefficient_efficiency(ts_ref.value, ts_lmr.value)
                elif stat == 'RMSE':
                    stat_array[0, i, j] = np.sqrt(np.mean(ts_ref.value-ts_lmr.value)**2)
                else:
                    raise ValueError('Wrong input stat; should be one of: "corr", "R2", "CE", "RMSE"')

        stat_fd = Field(stat, [0], target_item.lat, target_item.lon, stat_array)
        return stat_fd

    def validate_ts(self, target_item, stat='corr', valid_period=[1880, 2000], corr_method_kws=None, verbose=False):
        ts_array = []
        for ts in self.series_list:
            ts_array.append(ts.value)

        time_lmr = self.series_list[0].time
        value_lmr = np.median(ts_array, axis=0)
        if valid_period[0] < np.min(target_item.time):
            valid_period[0] = np.min(target_item.time)
        if valid_period[1] > np.max(target_item.time):
            valid_period[1] = np.max(target_item.time)
        if verbose: p_header(f'LMRt: res.ReconSeries.validate() >>> valid_period = {valid_period}')

        ts_lmr = Series(time=time_lmr, value=value_lmr).slice(valid_period)

        if corr_method_kws is None: corr_method_kws = {'method': 'isospectral', 'nsim': 100}

        new = self.copy()
        ts_ref = target_item.slice(valid_period)
        corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
        res_dict = {}
        if type(corr_res) is dict:
            res_dict['corr'] = corr_res['r']
        else:
            res_dict['corr'] = corr_res.r
        res_dict['CE'] = coefficient_efficiency(ts_ref.value, ts_lmr.value)
        new.valid_dict = res_dict
        new.valid_target = target_item
        return  new

    def validate(self, target_item, stat='corr', valid_period=[1880, 2000], corr_method_kws=None, verbose=False):
        if isinstance(target_item, Field):
            return self.validate_fd(target_item, stat=stat, valid_period=valid_period, corr_method_kws=corr_method_kws, verbose=verbose)

        elif isinstance(target_item, Series):
            return self.validate_ts(target_item, stat=stat, valid_period=valid_period, corr_method_kws=corr_method_kws, verbose=verbose)

        else:
            raise ValueError('Wrong type of target_item; should be either Series or Field.')

    def plot(self, **kws):
        ts_array = []
        for ts in self.series_list:
            ts_array.append(ts.value)

        time_lmr = self.series_list[0].time
        value_lmr = np.median(ts_array, axis=0)
        ts_lmr = Series(time=time_lmr, value=value_lmr)

        fig, ax = self.plot_envelope(mute=True, **kws)
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if hasattr(self, 'valid_target'):
            self.valid_target.plot(
                ax=ax, color=sns.xkcd_rgb['dark grey'], zorder=100,
                label=f'{self.valid_target.label} (corr={self.valid_dict["corr"]:.2f}, CE={self.valid_dict["CE"]:.2f})',
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        savefig_settings = kws['savefig_settings'] if 'savefig_settings' in kws else {}
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        else:
            mute = kws['mute'] if 'mute' in kws else None
            if not mute:
                showfig(fig)
        return fig, ax



# class ReconSeries:
    # def __init__(self, time, value, varname=None, verbose=False):
    #     self.time = time
    #     self.value = value
    #     self.varname = varname

    # def reshape(self):
    #     ''' Reshape self.value to be nt x nEns
    #     '''
    #     new = self.copy()
    #     ndim = len(np.shape(self.value))
    #     if ndim == 3:
    #         # full ens
    #         new.nt, new.nEns, new.nMC = np.shape(new.value)
    #         new.value = np.reshape(new.value, [new.nt, new.nMC*new.nEns])
    #     elif ndim == 2:
    #         # ens mean
    #         new.nt, new.nMC = np.shape(new.value)
    #         new.value = np.reshape(new.value, [new.nt, new.nMC])
    #     elif ndim == 1:
    #         # single series
    #         new.nt = np.size(new.value)
    #         new.value = np.reshape(new.value, [new.nt, 1])
    #     else:
    #         raise ValueError('Wrong number of dimensions.')

    #     return new

    # def median(self, axis=-1):
    #     new = self.copy()
    #     new.value = np.median(new.value, axis=axis)
    #     return new

    # def copy(self):
    #     '''Make a copy of the Series object

    #     Returns
    #     -------
    #     Series
    #         A copy of the Series object

    #     '''
    #     return deepcopy(self)

    # def slice(self, timespan=None):
    #     ''' Slicing the timeseries with a timespan (tuple or list)

    #     Parameters
    #     ----------

    #     timespan : tuple or list
    #         The list of time points for slicing, whose length must be even.
    #         When there are n time points, the output Series includes n/2 segments.
    #         For example, if timespan = [a, b], then the sliced output includes one segment [a, b];
    #         if timespan = [a, b, c, d], then the sliced output includes segment [a, b] and segment [c, d].

    #     Returns
    #     -------

    #     new : ReconSeries
    #         The sliced ReconSeries object.

    #     '''
    #     new = self.copy()
    #     if timespan is not None:
    #         n_elements = len(timespan)
    #         if n_elements % 2 == 1:
    #             raise ValueError('The number of elements in timespan must be even!')

    #         n_segments = int(n_elements / 2)
    #         mask = [False for i in range(np.size(self.time))]
    #         for i in range(n_segments):
    #             mask |= (self.time >= timespan[i*2]) & (self.time <= timespan[i*2+1])

    #         new.time = self.time[mask]
    #         new.value = self.value[mask]

    #     return new

    # def CE(self, ref_ts, timespan=None):
    #     ref = ref_ts.slice(timespan).reshape().median().value
    #     test = self.slice(timespan).reshape().median().value
    #     res = utils.coefficient_efficiency(ref, test)

    #     return res

    # def corr(self, ref_ts, timespan=None):
    #     ref = ref_ts.slice(timespan).reshape().median().value
    #     test = self.slice(timespan).reshape().median().value
    #     res = np.corrcoef(ref, test)[0, 1]

    #     return res


    # def plot(self, qs=[0.025, 0.25, 0.5, 0.75, 0.975], ax=None, savefig_settings=None, figsize=[8, 4], mute=False,
    #          xlim=None, ylim=None, xlabel='Year (CE)', ylabel=None, color=sns.xkcd_rgb['pale red'], **plt_kws):
    #     plt.ioff()
    #     savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)

    #     if 'alpha' not in plt_kws:
    #         plt_kws['alpha'] = 1
    #     if 'lw' not in plt_kws and 'linewidth' not in plt_kws:
    #         plt_kws['linewidth'] = 1


    #     ndim = len(np.shape(self.value))
    #     if ndim > 1:
    #         ts_qs = np.quantile(self.reshape().value, qs, axis=-1)
    #         nqs = len(qs)
    #         idx_mid = int(np.floor(nqs/2))

    #         if qs[idx_mid] == 0.5:
    #             label = 'median'
    #         else:
    #             label = f'{qs[2]*100}%'

    #         ax.plot(self.time, ts_qs[idx_mid], label=label, color=color, **plt_kws)
    #         ax.fill_between(self.time, ts_qs[-2], ts_qs[1], color=color, edgecolor='none', alpha=0.5, label=f'{qs[1]*100}% to {qs[-2]*100}%')
    #         ax.fill_between(self.time, ts_qs[-1], ts_qs[0], color=color, edgecolor='none',alpha=0.1, label=f'{qs[0]*100}% to {qs[-1]*100}%')
    #         ax.legend()
    #     else:
    #         ax.plot(self.time, self.value, color=color, **plt_kws)

    #     ax.set_xlabel(xlabel)
    #     if ylabel is None: ylabel = self.varname
    #     ax.set_ylabel(ylabel)

    #     if xlim is not None:
    #         ax.set_xlim(xlim)
    #     if ylim is not None:
    #         ax.set_ylim(ylim)

    #     if 'fig' in locals():
    #         if 'path' in savefig_settings:
    #             savefig(fig, settings=savefig_settings)
    #         else:
    #             if not mute:
    #                 showfig(fig)
    #         return fig, ax
    #     else:
    #         return ax

class ValidStat:
    def __init__(self, proxydb, stat_name, stat_dict):
        self.proxydb = proxydb
        self.stat_name = stat_name
        self.stat_dict = stat_dict

    def plot(self, figsize=[8, 5], proj_args=None,  projection='Robinson', central_longitude=180, markersize=50, cmap=None,
            transform=ccrs.PlateCarree(), savefig_settings=None, mute=False,
            cbar_labels=None, cbar_pad=0.05, cbar_orientation='vertical', cbar_aspect=10,
            cbar_fraction=0.15, cbar_shrink=0.5, cbar_title=None, cbar_extend=None):
        plt.ioff()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        fig = plt.figure(figsize=figsize)
        proj_args = {} if proj_args is None else proj_args
        proj_args_default = {'central_longitude': central_longitude}
        proj_args_default.update(proj_args)
        projection = CartopySettings.projection_dict[projection](**proj_args_default)
        ax = {}
        ax['map'] = plt.subplot(projection=projection)
        ax['map'].set_global()
        ax['map'].add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

        s_lats = []
        s_lons = []
        s_vals = []
        for pid, pobj in self.proxydb.records.items():
            s_lats.append(pobj.lat)
            s_lons.append(pobj.lon)
            s_vals.append(self.stat_dict[pid])


        if cmap is None:
            if self.stat_name == 'corr':
                cmap = 'RdBu_r'
                vmin, vmax = -1, 1
                cbar_extend = 'neither'
            elif self.stat_name in ['RMSE', 'CE']:
                cmap = 'RdBu_r'
                cbar_extend = 'both'
            elif self.stat_name == 'R2':
                vmin, vmax = 0, 1
                cmap = 'Reds'
                cbar_extend = 'neither'
            else:
                raise ValueError('Wrong stat_name.')

        clr_norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax['map'].scatter(s_lons, s_lats, c=s_vals, s=markersize, cmap=cmap, transform=transform, norm=clr_norm)
        cbar = fig.colorbar(
            im, ax=ax['map'],
            orientation=cbar_orientation,
            pad=cbar_pad, aspect=cbar_aspect,
            extend=cbar_extend, fraction=cbar_fraction,
            shrink=cbar_shrink,
        )
        if cbar_title is None:
            cbar_title = self.stat_name

        cbar.ax.set_title(cbar_title)
        if cbar_labels is not None:
            cbar.set_ticks(cbar_labels)


        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        else:
            if not mute:
                showfig(fig)

        return fig, ax

class ReconField:
    def __init__(self, time, lat, lon, field_list, varname=None, verbose=False):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.field_list = field_list
        self.varname = varname
        self.nt = np.size(time)
        self.nlat = np.size(lat)
        self.nlon = np.size(lon)
        self.nEns = len(field_list)

    def copy(self):
        '''Make a copy of the Series object

        Returns
        -------
        Series
            A copy of the Series object

        '''
        return deepcopy(self)

    def __str__(self):
        msg = inspect.cleandoc(f'''
            varname: {self.varname}
            nt: {self.nt}
            nlat: {self.nlat}
            nlon: {self.nlon}
            nEns: {self.nEns}
            ''')

        return msg

    def validate_fd(self, target_item, stat='corr', valid_period=[1880, 2000], corr_method_kws=None, verbose=False):
        fd_array = []
        for fd in self.field_list:
            fd_array.append(fd.value)

        fd = np.mean(fd_array, axis=0)

        target_field = target_item
        if valid_period[0] < np.min(target_field.time):
            valid_period[0] = np.min(target_field.time)
        if valid_period[1] > np.max(target_field.time):
            valid_period[1] = np.max(target_field.time)
        if verbose: p_header(f'LMRt: res.ReconField.validate() >>> valid_period = {valid_period}')

        target_field = target_field.regrid(self.nlat)

        if corr_method_kws is None: corr_method_kws = {'method': 'isospectral', 'nsim': 100}

        stat_array = np.ndarray((1, self.nlat, self.nlon))
        for i in tqdm(range(self.nlat), desc=f'Calculating metric: {stat}'):
            for j in range(self.nlon):
                ts_lmr = Series(time=self.time, value=fd[:, i, j]).slice(valid_period)
                ts_ref = Series(time=target_field.time, value=target_field.value[:, i, j]).slice(valid_period)
                if stat == 'corr':
                    corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
                    if type(corr_res) is dict:
                        stat_array[0, i, j] = corr_res['r']
                    else:
                        stat_array[0, i, j] = corr_res.r
                elif stat == 'R2':
                    corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
                    if type(corr_res) is dict:
                        stat_array[0, i, j] = corr_res['r']**2
                    else:
                        stat_array[0, i, j] = corr_res.r**2
                elif stat == 'CE':
                    stat_array[0, i, j] = coefficient_efficiency(ts_ref.value, ts_lmr.value)
                elif stat == 'RMSE':
                    stat_array[0, i, j] = np.sqrt(np.mean(ts_ref.value-ts_lmr.value)**2)
                else:
                    raise ValueError('Wrong input stat; should be one of: "corr", "R2", "CE", "RMSE"')

        stat_fd = Field(stat, [0], self.lat, self.lon, stat_array)

        return stat_fd

    def validate_proxydb(self, target_item, stat='corr', corr_method_kws=None, verbose=False):
        if isinstance(target_item, ProxyRecord):
            pdb = ProxyDatabase()
            pdb.add(target_item, inplace=True)
            pdb.refresh()
            target_item = pdb.copy()

        fd_array = []
        for fd in self.field_list:
            fd_array.append(fd.value)

        fd = np.mean(fd_array, axis=0)

        stat_dict = {}
        if corr_method_kws is None: corr_method_kws = {'method': 'isospectral', 'nsim': 100}
        for pid, pobj in target_item.records.items():
            lat_idx, lon_idx = find_closest_loc(self.lat, self.lon, pobj.lat, pobj.lon)
            t1 = self.time
            t2 = pobj.time
            overlap_yrs = np.intersect1d(t1, t2)
            idx1 = np.searchsorted(t1, overlap_yrs)
            idx2 = np.searchsorted(t2, overlap_yrs)
            ts_lmr = Series(time=self.time[idx1], value=fd[idx1, lat_idx, lon_idx])
            ts_ref = Series(time=pobj.time[idx2], value=pobj.value[idx2])
            if stat == 'corr':
                corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
                if type(corr_res) is dict:
                    stat_dict[pid] = corr_res['r']
                else:
                    stat_dict[pid] = corr_res.r
            elif stat == 'R2':
                corr_res = ts_lmr.correlation(ts_ref, settings=corr_method_kws)
                if type(corr_res) is dict:
                    stat_dict[pid] = corr_res['r']**2
                else:
                    stat_dict[pid] = corr_res.r**2
            elif stat == 'CE':
                stat_dict[pid] = coefficient_efficiency(ts_ref.value, ts_lmr.value)
            elif stat == 'RMSE':
                stat_dict[pid] = np.sqrt(np.mean(ts_ref.value-ts_lmr.value)**2)
            else:
                raise ValueError('Wrong input stat; should be one of: "corr", "R2", "CE", "RMSE"')

        valid_stat = ValidStat(target_item, stat, stat_dict)

        return valid_stat



    def validate(self, target_item, stat='corr', valid_period=[1880, 2000], corr_method_kws=None, verbose=False):

        if isinstance(target_item, Field):
            return self.validate_fd(target_item, stat=stat, valid_period=valid_period, corr_method_kws=corr_method_kws, verbose=verbose)

        elif isinstance(target_item, ProxyDatabase) or isinstance(target_item, ProxyRecord):
            return self.validate_proxydb(target_item, stat=stat, corr_method_kws=corr_method_kws, verbose=verbose)

        else:
            raise ValueError('Wrong type of target_item; should be either Field or ProxyDatabase.')
class ReconRes:
    def __init__(self, job_dirpath, load_num=None, verbose=False):
        try:
            recon_paths = sorted(glob.glob(os.path.join(job_dirpath, 'job_r*_recon.nc')))
            if load_num is not None:
                recon_paths = recon_paths[:load_num]
            self.recon_paths = recon_paths
        except:
            self.recon_paths = None

        try:
            idx_paths = sorted(glob.glob(os.path.join(job_dirpath, 'job_r*_idx.pkl')))
            if load_num is not None:
                idx_paths = idx_paths[:load_num]
            self.idx_paths = idx_paths
        except:
            self.idx_paths = None

        try:
            self.job_path = os.path.join(job_dirpath, 'job.pkl')
        except:
            self.job_path = None

        self.vars = {}

        if verbose:
            print(self)

    def copy(self):
        '''Make a copy of the Series object

        Returns
        -------
        Series
            A copy of the Series object

        '''
        return deepcopy(self)

    def __str__(self):
        msg = inspect.cleandoc(f'''
            recon_paths: {self.recon_paths}
            idx_paths: {self.idx_paths}
            job_path: {self.job_path}
            ''')

        return msg


    def get_vars(self, varnames, field_varnames=['tas', 'pr', 'psl', 'u', 'v'],
                series_varnames=[
                  'tas_gm_ens', 'tas_nhm_ens', 'tas_shm_ens',
                  'pr_gm_ens', 'pr_nhm_ens', 'pr_shm_ens',
                  'nino1+2', 'nino3', 'nino3.4', 'nino4', 'wpi', 'tpi', 'geo_mean',
                ], verbose=False):
        if type(varnames) is str:
            varnames = [varnames]

        for varname in varnames:
            if verbose: p_hint(f'LMRt: res.get_var() >>> loading variable: {varname}')
            var_list = []
            for idx, path in enumerate(self.recon_paths):
                with xr.open_dataset(path) as ds:
                    var_list.append(ds[varname])
                    if idx == 0:
                        year = ds['year'].values
                        lat = ds['lat'].values
                        lon = ds['lon'].values

            var_array = np.array(var_list)
            var_array = np.moveaxis(var_array, 0, -1)  # the axis of iteration to the end 

            if varname in field_varnames:
                ndim = len(np.shape(var_array))
                if ndim == 5:
                    # full ens
                    nt, nlat, nlon, nEns, nMC = np.shape(var_array)
                    value = np.reshape(var_array, [nt, nlat, nlon, nMC*nEns])
                elif ndim == 4:
                    # ens mean
                    nt, nlat, nlon, nMC = np.shape(var_array)
                    nEns = None
                    value = np.reshape(var_array, [nt, nlat, nlon, nMC])
                else:
                    raise ValueError('Wrong number of dimensions.')

                field_list = []
                if nEns is not None:
                    for j in range(nMC):
                        for i in range(nEns):
                            field_list.append(
                                Field(varname, year, lat, lon, value[..., i, j])
                            )
                else:
                    for j in range(nMC):
                        field_list.append(
                            Field(varname, year, lat, lon, value[..., j])
                        )

                self.vars[varname] = ReconField(year, lat, lon, field_list, varname=varname)
            elif varname in series_varnames:
                if 'tas' in varname or 'nino' in varname or 'wpi' in varname or 'tpi' in varname:
                    value_unit = 'K'
                else:
                    value_unit = None

                nt, nEns, nMC = np.shape(var_array)
                series_list = []
                for j in range(nMC):
                    for i in range(nEns):
                        series_list.append(
                            Series(
                                time=year, value=var_array[:,i,j],
                                time_name='Time', time_unit='yr',
                                value_name=varname, value_unit=value_unit,
                            )
                        )

                # self.vars[varname] = ReconSeries(year, var_array, varname=varname)
                self.vars[varname] = ReconSeries(series_list)

        if verbose: p_success(f"LMRt: res.get_var() >>> res.vars filled w/ varnames: {varnames} and ['year', 'lat', 'lon']")