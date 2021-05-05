''' For model prior and instrumental observations
'''
import numpy as np
import inspect
import xarray as xr
from spharm import Spharmt, regrid
import copy
import matplotlib.pyplot as plt
from .visual import (
    plot_field_map,
    showfig,
    savefig,
)

from .utils import (
    pp,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
    datetime2year_float,
    generate_latlon,
    seasonal_var,
    rotate_lon,
    get_anomaly,
    dropna_field,
)

class Field:
    def __init__(self, name, time, lat, lon, value, source=None):
        self.name = name
        self.time = time
        self.value = value
        self.lat = lat
        self.lon = np.mod(lon, 360)
        if any(np.diff(self.lon) < 0):
            self.value, self.lon = rotate_lon(self.value, self.lon)

        self.source = source
        self.nt = np.size(time)
        self.dt = np.median(np.diff(time))
        self.nlat = np.size(lat)
        self.nlon = np.size(lon)

    def copy(self):
        return copy.deepcopy(self)

    def seasonalize(self, season=list(range(1, 13)), make_year_mm_nan=True, inplace=False):
        ''' Seasonalize the field
        '''

        new_time, new_value = seasonal_var(self.time, self.value, avgMonths=season, make_yr_mm_nan=make_year_mm_nan)
        new_time, new_value = dropna_field(new_time, new_value)
        if inplace:
            self.time = new_time
            self.value = new_value
            self.nt = np.size(time)
            self.dt = np.median(np.diff(time))
        else:
            new_field = self.copy()
            new_field.value = new_value
            new_field.time = new_time
            new_field.nt = np.size(new_time)
            new_field.dt = np.median(np.diff(new_time))
            return new_field

    def regrid(self, ntrunc, inplace=False):
        old_spec = Spharmt(self.nlon, self.nlat, gridtype='regular', legfunc='computed')
        ifix = ntrunc % 2
        new_nlat = ntrunc + ifix
        new_nlon = int(new_nlat*1.5)
        new_spec = Spharmt(new_nlon, new_nlat, gridtype='regular', legfunc='computed')
        include_poles = False if new_nlat % 2 == 0 else True
        new_lat_2d, new_lon_2d, _, _ = generate_latlon(new_nlat, new_nlon, include_endpts=include_poles)
        new_lat = new_lat_2d[:, 0]
        new_lon = new_lon_2d[0, :]

        # new_value = []
        # for old_value in self.value:
        old_value = np.moveaxis(self.value, 0, -1)
        regridded_value = regrid(old_spec, new_spec, old_value, ntrunc=new_nlat-1, smooth=None)
        new_value = np.moveaxis(regridded_value, -1, 0)
            # new_value.append(regridded_value)

        new_value = np.array(new_value)

        if inplace:
            self.value = new_value
            self.lat = new_lat
            self.lon = new_lon
            self.nlat = np.size(new_lat)
            self.nlon = np.size(new_lon)
            self.ntrunc = ntrunc
        else:
            new_field = self.copy()
            new_field.value = new_value
            new_field.lat = new_lat
            new_field.lon = new_lon
            new_field.nlat = np.size(new_lat)
            new_field.nlon = np.size(new_lon)
            new_field.ntrunc = ntrunc
            return new_field

    def plot(self, idx_t=0, mute=False, **kwargs):
        plt.ioff()
        fig, ax =  plot_field_map(self.value[idx_t], self.lat, self.lon, **kwargs)
        if not mute:
            showfig(fig)
        return fig, ax



class Dataset:
    def __init__(self, fields=None):
        ''' Gridded data that includes multiple fields

        This type can be used for both model prior and instrumental observations that are stored in gridded data format

        Parameters
        ----------
        fields : dict
            a dict with variable names as keys
        '''
        self.fields = fields

    def copy(self):
        return copy.deepcopy(self)
    
    def load_nc(self, path_dict, anom_period=None, varname_dict=None, inplace=False):

        ''' Load a NetCDF file that assumed to include time, lat, lon, value

        Parameters
        ----------

        varname_dict: dict
            a dict to map variable names, e.g. {'tas': 'tempanomaly'} means 'tas' is named 'tempanomaly' in the input NetCDF file

        '''
        varname_dict = {} if varname_dict is None else varname_dict.copy()
        # the default names for variables time, lat, lon
        vn_dict = {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon',
        }
        vn_dict.update(varname_dict)
        
        fields = {}
        for name, path in path_dict.items():
            with xr.open_dataset(path) as ds:
                if vn_dict['time'] in ds:
                    date = ds[vn_dict['time']].values
                    time = datetime2year_float(date)
                else:
                    raise ValueError('time not available')

                if vn_dict['lat'] in ds.keys():
                    lat = ds[vn_dict['lat']].values
                else:
                    raise ValueError('lat not available')

                if vn_dict['lon'] in ds.keys():
                    lon = ds[vn_dict['lon']].values
                else:
                    raise ValueError('lon not available')

                vn = vn_dict[name]  # variable name in the NetCDF file
                if anom_period is None:
                    # load the original data
                    value = ds[vn].values
                else:
                    # calculate the anomaly
                    value = get_anomaly(ds[vn].values, time, ref_period=anom_period)

                fields[name] = Field(name=name, time=time, lat=lat, lon=lon, value=value, source=path)

        if inplace:
            self.fields = fields
        else:
            new_ds = self.copy()
            new_ds.fields = fields

            return new_ds

    def seasonalize(self, season, inplace=False):
        new_fields = {}
        for name, field in self.fields.items():
            new_fields[name] = field.seasonalize(season=season, inplace=False)

        if inplace:
            self.fields = new_fields
        else:
            new_ds = self.copy()
            new_ds.fields = new_fields

            return new_ds

    def regrid(self, ntrunc, inplace=False):
        new_fields = {}
        for name, field in self.fields.items():
            new_fields[name] = field.regrid(ntrunc=ntrunc, inplace=False)

        if inplace:
            self.fields = new_fields
        else:
            new_ds = self.copy()
            new_ds.fields = new_fields

            return new_ds

    def __str__(self):
        msg_block = ''
        for k, field in self.fields.items():
            msg_block += f'''
                 Name:\t{field.name}
               Source:\t{field.source}
                Shape:\ttime:{field.nt}, lat:{field.nlat}, lon:{field.nlon}
            '''

        msg = inspect.cleandoc(f'''
            Dataset Overview
            -----------------------
            {msg_block}
            ''')
        return msg

