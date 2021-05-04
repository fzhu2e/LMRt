''' For proxies
'''
import numpy as np
import pandas as pd
import pickle
import inspect
import copy
import os
from collections import OrderedDict
from tqdm import tqdm
import random

from pyleoclim.core.ui import (
    Series,
)

from .utils import (
    pp,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
    clean_ts,
    seasonal_var,
    find_closest_loc,
)

from .psm import (
    Linear,
    Bilinear,
)

import matplotlib.pyplot as plt
from .visual import (
    plot_proxies,
    showfig,
    savefig,
    PAGES2k,
)


def get_ptype(archive_type, proxy_type):
    ptype_dict = {
        ('tree', 'delta Density'): 'tree.MXD',
        ('tree', 'MXD'): 'tree.MXD',
        ('tree', 'TRW'): 'tree.TRW',
        ('tree', 'ENSO'): 'tree.ENSO',
        ('coral', 'Sr/Ca'): 'coral.SrCa',
        ('coral', 'Coral Sr/Ca'): 'coral.SrCa',
        ('coral', 'd18O'): 'coral.d18O',
        ('coral', 'calcification'): 'coral.calc',
        ('coral', 'calcification rate'): 'coral.calc',
        ('sclerosponge', 'd18O'): 'coral.d18O',
        ('sclerosponge', 'Sr/Ca'): 'coral.SrCa',
        ('glacier ice', 'melt'): 'ice.melt',
        ('glacier ice', 'd18O'): 'ice.d18O',
        ('glacier ice', 'dD'): 'ice.dD',
        ('speleothem', 'd18O'): 'speleothem.d18O',
        ('marine sediment', 'TEX86'): 'marine.TEX86',
        ('marine sediment', 'foram Mg/Ca'): 'marine.MgCa',
        ('marine sediment', 'd18O'): 'marine.d18O',
        ('marine sediment', 'dynocist MAT'): 'marine.MAT',
        ('marine sediment', 'alkenone'): 'marine.alkenone',
        ('marine sediment', 'planktonic foraminifera'): 'marine.foram',
        ('marine sediment', 'foraminifera'): 'marine.foram',
        ('marine sediment', 'foram d18O'): 'marine.foram',
        ('marine sediment', 'diatom'): 'marine.diatom',
        ('lake sediment', 'varve thickness'): 'lake.varve_thickness',
        ('lake sediment', 'varve property'): 'lake.varve_property',
        ('lake sediment', 'sed accumulation'): 'lake.accumulation',
        ('lake sediment', 'chironomid'): 'lake.chironomid',
        ('lake sediment', 'midge'): 'lake.midge',
        ('lake sediment', 'TEX86'): 'lake.TEX86',
        ('lake sediment', 'BSi'): 'lake.BSi',
        ('lake sediment', 'chrysophyte'): 'lake.chrysophyte',
        ('lake sediment', 'reflectance'): 'lake.reflectance',
        ('lake sediment', 'pollen'): 'lake.pollen',
        ('lake sediment', 'alkenone'): 'lake.alkenone',
        ('borehole', 'borehole'): 'borehole',
        ('hybrid', 'hybrid'): 'hybrid',
        ('bivalve', 'd18O'): 'bivalve.d18O',
        ('documents', 'Documentary'): 'documents',
        ('documents', 'historic'): 'documents',
    }

    return ptype_dict[(archive_type, proxy_type)]


class ProxyRecord(Series):
    def __init__(self, pid, lat, lon, time, value, ptype, psm_name=None, psm=None, time_name='Time', value_name=None, time_unit='yr', value_unit=None,
        label=None, prior_value=None, prior_time=None, obs_value=None, obs_time=None, seasonality=None, ye_time=None, ye_value=None):
        '''
        Parameters
        ----------
        pid : str
            the unique proxy ID

        lat : float
            latitude

        lon : float
            longitude

        time : np.array
            time axis in unit of year CE 

        value : np.array
            proxy value axis

        ptype : str
            the label of proxy type according to archive and proxy information;
            some examples:
            - 'tree.trw' : TRW
            - 'tree.mxd' : MXD
            - 'coral.d18O' : Coral d18O isotopes
            - 'coral.SrCa' : Coral Sr/Ca ratios
            - 'ice.d18O' : Ice d18O isotopes

        psm_name : str
            specified PSM type name

        psm : PSM
            specified PSM object

        prior_value : dict
            a dict of environmental variable values

        prior_time : dict
            a dict of environmental variable time

        seasonality : dict
            a dict of seasonality for multiple environmental variables

        ye_time : np.array
            time axis for the simulated pseudoproxy

        ye_value : np.array
            value axis for the simulated pseudoproxy

        '''
        self.pid = pid
        self.lat = lat
        self.lon = lon
        self.time = time
        self.value = value
        self.time_name = time_name
        self.value_name = value_name
        self.time_unit = time_unit
        self.value_unit = value_unit
        self.label = label
        self.dt = np.median(np.diff(time))
        self.nt = np.size(time)
        self.ptype = ptype
        self.psm_name = psm_name
        self.psm = psm
        self.prior_value = prior_value
        self.prior_time = prior_time
        self.obs_value = obs_value
        self.obs_time = obs_time
        self.seasonality = seasonality
        self.ye_time = ye_time
        self.ye_value = ye_value

    def plot(self, plot_Ye=True, **kws):
        if 'color' not in kws and 'c' not in kws:
            kws['color'] = PAGES2k.colors_dict[self.ptype]

        fig, ax = super().plot(mute=True, **kws)

        if hasattr(self.psm, 'calib_details') and self.psm.calib_details is not None:
            calibed_season = self.psm.calib_details['seasonality']
            if type(calibed_season) is str:
                season_tas = [int(m) for m in calibed_season.split('_')]
                season_pr = None
            elif type(calibed_season) is tuple:
                season_tas = [int(m) for m in calibed_season[0].split('_')]
                season_pr = [int(m) for m in calibed_season[1].split('_')]

            season_str = f'Seasonality: tas {season_tas}'
            if season_pr is not None:
                season_str += f'; pr {season_pr}'
        else:
            season_str = ''

        ax.set_title(f'{self.pid} @ (lat:{self.lat}, lon:{self.lon}) | Proxy type: {self.ptype} | PSM: {self.psm_name}\n{season_str}')
        if self.ye_value is not None and plot_Ye:
            if 'plot_kwargs' in kws:
                ax.plot(self.ye_time, self.ye_value, label='Ye', **kws['plot_kwargs'])
            else:
                ax.plot(self.ye_time, self.ye_value, label='Ye')

        if 'lgd_kwargs' in kws:
            ax.legend(**kws['lgd_kwargs'])
        else:
            ax.legend()

        savefig_settings = kws['savefig_settings'] if 'savefig_settings' in kws else {}
        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        else:
            mute = kws['mute'] if 'mute' in kws else None
            if not mute:
                showfig(fig)
        return fig, ax

    # def plot(self, ax=None, savefig_settings=None, figsize=[8, 4], mute=False, xlim=None, ylim=None, **plt_kws):
    #     plt.ioff()
    #     savefig_settings = {} if savefig_settings is None else savefig_settings.copy()
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)

    #     if hasattr(self.psm, 'calib_details') and self.psm.calib_details is not None:
    #         calibed_season = self.psm.calib_details['seasonality']
    #         if type(calibed_season) is str:
    #             season_tas = [int(m) for m in calibed_season.split('_')]
    #             season_pr = None
    #         elif type(calibed_season) is tuple:
    #             season_tas = [int(m) for m in calibed_season[0].split('_')]
    #             season_pr = [int(m) for m in calibed_season[1].split('_')]

    #         season_str = f'Seasonality: tas {season_tas}'
    #         if season_pr is not None:
    #             season_str += f'; pr {season_pr}'
    #     else:
    #         season_str = ''

    #     ax.set_title(f'{self.pid} @ (lat:{self.lat}, lon:{self.lon}) | Proxy type: {self.ptype} | PSM: {self.psm_name}\n{season_str}')
    #     ax.plot(self.time, self.value, label='Proxy', **plt_kws)
    #     if self.ye_value is not None:
    #         ax.plot(self.ye_time, self.ye_value, label='Ye', **plt_kws)

    #     ax.legend()
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

    def copy(self):
        return copy.deepcopy(self)

    def seasonalize(self, season=list(range(1, 13)), make_year_mm_nan=False, inplace=False):
        ''' Seasonalize the proxy record
        '''

        old_time = np.copy(self.time)
        new_time = self.time[old_time >= 1]
        new_value = self.value[old_time >= 1]
        new_time, new_value = seasonal_var(new_time, new_value, avgMonths=season, make_yr_mm_nan=make_year_mm_nan)
        new_time, new_value = clean_ts(new_time, new_value)

        if inplace:
            self.time, self.value = new_time, new_value
        else:
            new_record = self.copy()
            new_record.time = new_time
            new_record.value = new_value
            new_record.dt = np.median(np.diff(new_time))
            new_record.nt = np.size(new_time)
            return new_record

    def init_psm(self):
        if self.psm_name == 'linear':
            self.psm = Linear(
                self.time, self.value,
                self.obs_time['tas'], self.obs_value['tas'],
                prior_tas_time=self.prior_time['tas'],
                prior_tas_value=self.prior_value['tas'],
            )
        elif self.psm_name == 'bilinear':
            self.psm = Bilinear(
                self.time, self.value,
                self.obs_time['tas'], self.obs_value['tas'],
                self.obs_time['pr'], self.obs_value['pr'],
                prior_tas_time=self.prior_time['tas'],
                prior_tas_value=self.prior_value['tas'],
                prior_pr_time=self.prior_time['pr'],
                prior_pr_value=self.prior_value['pr'],
            )

    def calib_psm(self, calib_period=None, calib_kws=None):
        calib_kws = {} if calib_kws is None else calib_kws.copy()
        self.psm.calibrate(calib_period=calib_period, **calib_kws)
        if self.psm.calib_details is not None:
            self.seasonality_opt = self.psm.calib_details['seasonality']
            self.SNR = self.psm.calib_details['SNR']
            self.R = self.psm.calib_details['PSMmse']
        else:
            self.seasonality_opt = None
            self.SNR = None
            self.R = None


    def forward_psm(self):
        ''' Forward modeling: calculate ye using the PSM according to self.psm
        '''
        if self.psm.calib_details is not None:
            self.psm.forward()
            self.ye_time = self.psm.ye_time
            self.ye_value = self.psm.ye_value
        else:
            self.ye_time = None
            self.ye_value = None


    def __str__(self):
        preview_num=10
        msg = inspect.cleandoc(f'''
            {self.pid} @ (lat:{self.lat}, lon:{self.lon}) | length: {np.size(self.value)} | proxy type: {self.ptype} | PSM: {self.psm_name} | Seasonality: {self.seasonality}
             Time[:{preview_num}]: {self.time[:preview_num]}
            Value[:{preview_num}]: {self.value[:preview_num]}
            ''')

        return msg


class ProxyDatabase:
    def __init__(self, records=None, source=None):
        '''
        Parameters
        ----------
        records : dict
            a dict of the ProxyRecord objects with proxy ID as keys

        source : str
            a path to the original source file

        '''
        records = {} if records is None else records
        self.records = records
        self.source = source
        if records is not None:
            self.refresh()

    def copy(self):
        return copy.deepcopy(self)

    def refresh(self):
        self.nrec = len(self.records)
        self.pids = [pobj.pid for pid, pobj in self.records.items()]
        self.lats = [pobj.lat for pid, pobj in self.records.items()]
        self.lons = [pobj.lon for pid, pobj in self.records.items()]
        self.type_list = [pobj.ptype for pid, pobj in self.records.items()]
        self.type_dict = {}
        for t in self.type_list:
            if t not in self.type_dict:
                self.type_dict[t] = 1
            else:
                self.type_dict[t] += 1

    def load_df(self, df, pid_column='paleoData_pages2kID', lat_column='geo_meanLat', lon_column='geo_meanLon',
                time_column='year', value_column='paleoData_values', proxy_type_column='paleoData_proxy', archive_type_column='archiveType',
                value_name_column='paleoData_variableName', value_unit_column='paleoData_units',
                ptype_season=None, ptype_psm=None, verbose=False):
        ''' Load database from a Pandas DataFrame

        Parameters
        ----------
        df : Pandas DataFrame
            a Pandas DataFrame include at least lat, lon, time, value, proxy_type
        
        ptype_psm : dict
            a mapping from ptype to psm
        '''
        if not isinstance(df, pd.DataFrame):
            err_msg = 'the input df should be a Pandas DataFrame.'
            if verbose:
                p_fail(f'LMRt: job.proxydb.load_df() >>> {err_msg}')
            raise TypeError(err_msg)

        records = OrderedDict()

        for idx, row in df.iterrows():
            proxy_type = row[proxy_type_column]
            archive_type = row[archive_type_column]
            ptype = get_ptype(archive_type, proxy_type)
            if ptype_psm is not None and ptype in ptype_psm:
                psm_name = ptype_psm[ptype]
            else:
                psm_name = None

            if ptype_season is not None and ptype in ptype_season:
                seasonality = ptype_season[ptype]
            else:
                seasonality = list(range(1, 13))  # annual by default

            pid = row[pid_column]
            lat = row[lat_column]
            lon = np.mod(row[lon_column], 360)
            time = np.array(row[time_column])
            value = np.array(row[value_column])
            time, value = clean_ts(time, value)
            value_name=row[value_name_column] if value_name_column in row else None
            value_unit=row[value_unit_column] if value_name_column in row else None

            record = ProxyRecord(
                pid=pid, lat=lat, lon=lon, time=time, value=value,
                ptype=ptype, psm_name=psm_name, psm=None,
                value_name=value_name, value_unit=value_unit, label=pid,
                prior_value={}, prior_time={}, obs_value={}, obs_time={},
                seasonality=seasonality,
            )
            records[pid] = record

        # update the attributes
        self.records = records
        self.refresh()

    def filter_ptype(self, ptype_list, inplace=False):
        ''' Filter the proxy database according to given ptype list

        Parameters
        ----------

        ptype_list : list
            a list of ptype's
        '''
        new_records = {}
        for pid, pobj in self.records.items():
            if pobj.ptype in ptype_list:
                new_records[pid] = pobj

        if inplace:
            # update the attributes
            self.records = new_records
            self.refresh()
        else:
            new_db = self.copy()
            new_db.records = new_records
            new_db.refresh()
            return new_db

    def filter_dt(self, dt=1, inplace=False):
        ''' Filter the proxy database according to temporal resolution
        '''
        new_records = {}
        for pid, pobj in self.records.items():
            if pobj.dt <= dt:
                new_records[pid] = pobj

        if inplace:
            # update the attributes
            self.records = new_records
            self.refresh()
        else:
            new_db = self.copy()
            new_db.records = new_records
            new_db.refresh()
            return new_db

    def filter_pids(self, pids, inplace=False):
        ''' Filter the proxy database according to given pids
        '''
        new_records = {}
        for pid, pobj in self.records.items():
            if pid in pids:
                new_records[pid] = pobj

        if inplace:
            # update the attributes
            self.records = new_records
            self.refresh()
        else:
            new_db = self.copy()
            new_db.records = new_records
            new_db.refresh()
            return new_db

    def seasonalize(self, ptype_season, inplace=False):
        ''' Filter the proxy database according to given ptype list

        Parameters
        ----------

        ptype_season : dict
            a dict of seasonality for each proxy type
        '''

        ptypes_to_seasonalize = ptype_season.keys()

        new_records = {}
        for pid, pobj in self.records.items():
            if pobj.ptype in ptypes_to_seasonalize:
                if pobj.dt >= 1:
                    pobj.time = np.array([int(t) for t in pobj.time])
                    new_records[pid] = pobj
                    continue

                season = ptype_season[pobj.ptype]
                if isinstance(season[0], list):
                    # when ptype_season[pobj.ptype] contains multiple seasonality possibilities
                    new_pobj = pobj
                else:
                    # when ptype_season[pobj.ptype] contains only one seasonality possibility
                    new_pobj = pobj.seasonalize(season=season, inplace=False)

                if np.size(new_pobj.value) >= 1:
                    new_records[pid] = new_pobj
            else:
                new_records[pid] = pobj

        if inplace:
            # update the attributes
            self.records = new_records
            self.refresh()
        else:
            new_db = self.copy()
            new_db.records = new_records
            new_db.refresh()
            return new_db

    def find_nearest_loc(self, var_names, ds=None, ds_type=None, ds_loc_path=None, save_path=None, verbose=False):
        if ds_loc_path is not None and os.path.exists(ds_loc_path):
            with open(ds_loc_path, 'rb') as f:
                if ds_type == 'prior':
                    self.prior_lat_idx, self.prior_lon_idx = pickle.load(f)
                elif ds_type == 'obs':
                    self.obs_lat_idx, self.obs_lon_idx = pickle.load(f)
                else:
                    raise ValueError('Wrong ds_type')
        else:
            lat_idx = {}
            lon_idx = {}
            for vn in var_names:
                field = ds.fields[vn]
                lat_idx[vn], lon_idx[vn] = find_closest_loc(field.lat, field.lon, self.lats, self.lons, mode='latlon', verbose=verbose)

            if ds_type == 'prior':
                self.prior_lat_idx = lat_idx
                self.prior_lon_idx = lon_idx
            elif ds_type == 'obs':
                self.obs_lat_idx = lat_idx
                self.obs_lon_idx = lon_idx
            else:
                raise ValueError('Wrong ds_type')

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump([lat_idx, lon_idx], f)

        if verbose:
            p_success(f'LMRt: job.proxydb.find_nearest_loc() >>> job.proxydb.{ds_type}_lat_idx & job.proxydb.{ds_type}_lon_idx created')

    def get_var_from_ds(self, seasonalized_ds, ptype_season, ds_type=None, verbose=False):
        ''' Get environmental variables from prior
        '''
        for i, pid in enumerate(self.pids):
            for vn in self.prior_lat_idx.keys():
                season = ptype_season[self.records[pid].ptype]
                if isinstance(season[0], list):
                    pass
                else:
                    # when ptype_season[pobj.ptype] contains only one seasonality possibility
                    season = [season]

                if ds_type == 'prior':
                    i_lat, i_lon = self.prior_lat_idx[vn][i], self.prior_lon_idx[vn][i]
                    tmp_value = self.records[pid].prior_value
                    tmp_time = self.records[pid].prior_time
                elif ds_type == 'obs':
                    i_lat, i_lon = self.obs_lat_idx[vn][i], self.obs_lon_idx[vn][i]
                    tmp_value = self.records[pid].obs_value
                    tmp_time = self.records[pid].obs_time
                else:
                    raise ValueError('Wrong ds_type')

                tmp_value[vn] =  {}
                tmp_time[vn] = {}

                for sn in season:
                    season_tag = '_'.join(str(s) for s in sn)
                    tmp_time[vn][season_tag] = seasonalized_ds[season_tag].fields[vn].time
                    tmp_value[vn][season_tag] = seasonalized_ds[season_tag].fields[vn].value[:, i_lat, i_lon]

        if verbose: p_success(f'LMRt: job.proxydb.get_var_from_ds() >>> job.proxydb.records[pid].{ds_type}_time & job.proxydb.records[pid].{ds_type}_value created')

    def add(self, records, inplace=False, verbose=False):
        ''' Add a list of records into the database
        '''
        newdb = self.copy()
        if isinstance(records, ProxyRecord):
            # if only one record
            records = [records]

        for record in records:
            newdb.records[record.pid] = record
            if verbose: p_success(f'LMRt: job.proxydb.add() >>> Record {record.pid} added.')

        if inplace:
            self.records = newdb.records
            self.refresh()
        else:
            newdb.refresh()
            return newdb

    def remove(self, pids, inplace=False, verbose=False):
        ''' Remove a list of records from the database regardless existing or not
        '''
        newdb = self.copy()
        if type(pids) is str:
            # contains only one id
            pids = [pids]

        for pid in pids:
            newdb.records.pop(pid, None)
            if verbose: p_success(f'LMRt: job.proxydb.remove() >>> Record {pid} removed.')

        if inplace:
            self.records = newdb.records
            self.refresh()
        else:
            newdb.refresh()
            return newdb


    def split(self, assim_frac, seed=0, verbose=False):
        random.seed(seed)
        targetdb = self.calibed

        nsites_assim = int(targetdb.nrec * assim_frac)
        idx_assim = random.sample(range(targetdb.nrec), nsites_assim)
        idx_assim.sort()
        idx_eval = list(set(range(targetdb.nrec)) - set(idx_assim))
        idx_eval.sort()
        proxydb_assim = ProxyDatabase(source=targetdb.source)
        proxydb_eval = ProxyDatabase(source=targetdb.source)
        pobjs_assim = []
        pobjs_eval = []
        for pid, pobj in targetdb.records.items():
            idx = list(targetdb.pids).index(pid)
            if idx in idx_assim:
                pobjs_assim.append(pobj)
            elif idx in idx_eval:
                pobjs_eval.append(pobj)

        proxydb_assim.add(pobjs_assim, inplace=True)
        proxydb_assim.refresh()
        proxydb_eval.add(pobjs_eval, inplace=True)
        proxydb_eval.refresh()

        self.calibed_idx_assim = idx_assim
        self.calibed_idx_eval = idx_eval
        self.assim = proxydb_assim
        self.eval = proxydb_eval
        if verbose:
            p_success(f'LMRt: job.proxydb.split() >>> job.proxydb.assim created')
            p_success(f'LMRt: job.proxydb.split() >>> job.proxydb.eval created')
            p_success(f'LMRt: job.proxydb.split() >>> job.proxydb.calibed_idx_assim created')
            p_success(f'LMRt: job.proxydb.split() >>> job.proxydb.calibed_idx_eval created')


    def init_psm(self, verbose=False):
        for pid, pobj in self.records.items():
            pobj.init_psm()

        if verbose: p_success(f'LMRt: job.proxydb.init_psm() >>> job.proxydb.records[pid].psm initialized')


    def calib_psm(self, calib_period=None, calib_kws=None, save_path=None, calibed_psm_path=None, verbose=False):
        if calibed_psm_path is not None and os.path.exists(calibed_psm_path):
            with open(calibed_psm_path, 'rb') as f:
                psm_model_dict, calib_details_dict = pickle.load(f)
                calibed_pobjs = []
                calibed = ProxyDatabase(source=self.source)
                for pid, pobj in self.records.items():
                    pobj.psm.model = psm_model_dict[pid]
                    pobj.psm.calib_details = calib_details_dict[pid]
                    if calib_details_dict[pid] is not None:
                        pobj.seasonality_opt = calib_details_dict[pid]['seasonality']
                        pobj.SNR = calib_details_dict[pid]['SNR']
                        pobj.R = calib_details_dict[pid]['PSMmse']
                    else:
                        pobj.seasonality_opt = None
                        pobj.SNR = None
                        pobj.R = None

                    if calib_details_dict[pid] is not None:
                        calibed_pobjs.append(pobj)

            calibed.add(calibed_pobjs, inplace=True)
            calibed.refresh()
            self.calibed = calibed
        else:
            calib_kws = {} if calib_kws is None else calib_kws.copy()
            psm_model_dict = {}
            calib_details_dict = {}
            calibed_pobjs = []
            calibed = ProxyDatabase(source=self.source)
            for pid, pobj in tqdm(self.records.items(), desc='Calibrating PSM'):
                pobj.calib_psm(calib_period=calib_period, calib_kws=calib_kws)
                psm_model_dict[pid] = pobj.psm.model
                calib_details_dict[pid] = pobj.psm.calib_details
                if pobj.psm.calib_details is not None:
                    calibed_pobjs.append(pobj)

            calibed.add(calibed_pobjs, inplace=True)
            calibed.refresh()
            self.calibed = calibed

            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump([psm_model_dict, calib_details_dict], f)

        if verbose:
            p_success(f'LMRt: job.proxydb.calib_psm() >>> job.proxydb.records[pid].psm calibrated')
            p_success(f'LMRt: job.proxydb.calib_psm() >>> job.proxydb.calibed created')

    def forward_psm(self, verbose=False):
        for pid, pobj in tqdm(self.records.items(), desc='Forwarding PSM'):
            if pobj.psm.calib_details is not None:
                pobj.forward_psm()

        if verbose: p_success(f'LMRt: job.proxydb.forward_psm() >>> job.proxydb.records[pid].psm forwarded')

    def plot(self, savefig_settings=None, mute=False, **kws):
        plt.ioff()
        savefig_settings = {} if savefig_settings is None else savefig_settings.copy()

        time_list = []
        for pid, pobj in self.records.items():
            time_list.append(pobj.time)

        df = pd.DataFrame({'lat': self.lats, 'lon': self.lons, 'type': self.type_list, 'time': time_list})
        fig, ax = plot_proxies(df, **kws)

        if 'path' in savefig_settings:
            savefig(fig, settings=savefig_settings)
        else:
            if not mute:
                showfig(fig)
        return fig, ax


    def __str__(self):
        msg = inspect.cleandoc(f'''
            Proxy Database Overview
            -----------------------
                 Source:\t{self.source}
                   Size:\t{self.nrec}
            Proxy types:\t{self.type_dict}
            ''')
        return msg