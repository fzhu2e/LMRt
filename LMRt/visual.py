''' Visualization and post processing
'''
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import cm
import pickle

import matplotlib as mpl
import numpy as np
import os
from scipy import stats
from scipy.stats.mstats import mquantiles
from cartopy import util as cutil

from . import utils

import pandas as pd
import statsmodels as sm
from statsmodels.graphics.gofplots import ProbPlot
from pandas.plotting import autocorrelation_plot
from tqdm import tqdm

from scipy.stats import cumfreq
from scipy.integrate import cumtrapz
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerLine2D

class PAGES2k(object):
    colors_dict = {
        'Bivalve_d18O': sns.xkcd_rgb['gold'],
        'Corals and Sclerosponges_Rates': sns.xkcd_rgb['yellow'],
        'Corals and Sclerosponges_SrCa': sns.xkcd_rgb['orange'],
        'Corals and Sclerosponges_d18O': sns.xkcd_rgb['amber'],
        'Ice Cores_MeltFeature': sns.xkcd_rgb['pale blue'],
        'Ice Cores_d18O': sns.xkcd_rgb['light blue'],
        'Ice Cores_dD': sns.xkcd_rgb['sky blue'],
        'Lake Cores_Misc': sns.xkcd_rgb['blue'],
        'Lake Cores_Varve': sns.xkcd_rgb['dark blue'],
        'Tree Rings_WidthPages2': sns.xkcd_rgb['green'],
        'Tree Rings_WoodDensity': sns.xkcd_rgb['forest green'],
        'Tree Rings_WidthBreit': sns.xkcd_rgb['sea green'],
        'tas': sns.xkcd_rgb['pale red'],
        'pr': sns.xkcd_rgb['aqua'],
        'Tree Rings_PDSI': sns.xkcd_rgb['sea green'],
    }

    markers_dict = {
        'Bivalve_d18O': 'p',
        'Corals and Sclerosponges_Rates': 'P',
        'Corals and Sclerosponges_SrCa': 'X',
        'Corals and Sclerosponges_d18O': 'o',
        'Ice Cores_MeltFeature': '<',
        'Ice Cores_d18O': 'd',
        'Ice Cores_dD': '>',
        'Lake Cores_Misc': 'D',
        'Lake Cores_Varve': 's',
        'Tree Rings_WidthPages2': '^',
        'Tree Rings_WoodDensity': 'v',
        'Tree Rings_WidthBreit': 'h',
        'tas': '^',
        'pr': 'o',
        'Tree Rings_PDSI': '^',
    }



class PAGES2k_ptype(object):
    ptypes = [
        # tree
        'tree_TRW',
        'tree_MXD',
        'tree_dDensity',
        # glacier ice
        'ice_melt',
        'ice_d18O',
        'ice_dD',
        # sclerosponge
        'sclerosponge_Sr/Ca',
        'sclerosponge_d18O',
        # bivalve
        'bivalve_d18O',
        # marine sediment
        'marine_d18O',
        'marine_foram Mg/Ca',
        'marine_alkenone',
        'marine_dynocist MAT',
        'marine_foram d18O',
        'marine_foraminifera',
        'marine_diatom',
        'marine_TEX86',
        # documents
        'documents_historic',
        'documents_Documentary',
        # hybrid
        'hybrid_hybrid',
        # coral
        'coral',
        'coral_d18O',
        'coral_Sr/Ca',
        'coral_calcification',
        # speleothem
        'speleothem_d18O',
        'lake sediment',
        'lake_midge',
        'lake_reflectance',
        'lake_varve thickness',
        'lake_varve property',
        'lake_sed accumulation',
        'lake_TEX86',
        'lake_BSi',
        'lake_pollen',
        'lake_chironomid',
        'lake_chrysophyte',
        'lake_alkenone',
        # borehole
        'borehole_borehole',
    ]

    markers_dict = {}
    for ptype in ptypes:
        if 'tree' in ptype:
            markers_dict[ptype] = '^'
        elif 'ice' in ptype:
            markers_dict[ptype] = 'd'
        elif 'sclerosponge' in ptype:
            markers_dict[ptype] = '8'
        elif 'bivalve' in ptype:
            markers_dict[ptype] = 'p'
        elif 'marine' in ptype:
            markers_dict[ptype] = 'x'
        elif 'documents' in ptype:
            markers_dict[ptype] = 'v'
        elif 'hybrid' in ptype:
            markers_dict[ptype] = '*'
        elif 'coral' in ptype:
            markers_dict[ptype] = 'o'
        elif 'speleothem' in ptype:
            markers_dict[ptype] = 'D'
        elif 'lake' in ptype:
            markers_dict[ptype] = 's'
        elif 'borehole' in ptype:
            markers_dict[ptype] = 'P'

    colors_dict = {}
    for ptype in ptypes:
        if 'TRW' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['green']
        elif 'MXD' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['forest green']
        elif 'dDensity' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['dark green']
        elif 'd18O' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['light blue']
        elif 'dD' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['baby blue']
        elif 'melt' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['pale blue']
        elif 'Sr/Ca' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['orange']
        elif 'Mg/Ca' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['mauve']
        elif 'alkenone' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['gold']
        elif 'MAT' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['salmon']
        elif 'diatom' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['teal']
        elif 'TEX86' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['beige']
        elif 'foraminifera' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['turquoise']
        elif 'calcification' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['amber']
        elif 'midge' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['bright blue']
        elif 'reflectance' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['periwinkle']
        elif 'thickness' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['navy blue']
        elif 'property' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['aquamarine']
        elif 'sed' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['cyan']
        elif 'BSi' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['olive']
        elif 'pollen' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['lime']
        elif 'chironomid' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['light pink']
        elif 'chrysophyte' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['lilac']
        elif 'hybrid' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['ochre']
        elif 'documents' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['puke']
        elif 'borehole' in ptype:
            colors_dict[ptype] = sns.xkcd_rgb['light brown']



class CartopySettings:
    projection_dict = {
        'Robinson': ccrs.Robinson,
        'NorthPolarStereo': ccrs.NorthPolarStereo,
        'SouthPolarStereo': ccrs.SouthPolarStereo,
        'PlateCarree': ccrs.PlateCarree,
        'AlbersEqualArea': ccrs.AlbersEqualArea,
        'AzimuthalEquidistant': ccrs.AzimuthalEquidistant,
        'EquidistantConic': ccrs.EquidistantConic,
        'LambertConformal': ccrs.LambertConformal,
        'LambertCylindrical': ccrs.LambertCylindrical,
        'Mercator': ccrs.Mercator,
        'Miller': ccrs.Miller,
        'Mollweide': ccrs.Mollweide,
        'Orthographic': ccrs.Orthographic,
        'Sinusoidal': ccrs.Sinusoidal,
        'Stereographic': ccrs.Stereographic,
        'TransverseMercator': ccrs.TransverseMercator,
        'UTM': ccrs.UTM,
        'InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine,
        'RotatedPole': ccrs.RotatedPole,
        'OSGB': ccrs.OSGB,
        'EuroPP': ccrs.EuroPP,
        'Geostationary': ccrs.Geostationary,
        'NearsidePerspective': ccrs.NearsidePerspective,
        'EckertI': ccrs.EckertI,
        'EckertII': ccrs.EckertII,
        'EckertIII': ccrs.EckertIII,
        'EckertIV': ccrs.EckertIV,
        'EckertV': ccrs.EckertV,
        'EckertVI': ccrs.EckertVI,
        'EqualEarth': ccrs.EqualEarth,
        'Gnomonic': ccrs.Gnomonic,
        'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea,
        'OSNI': ccrs.OSNI,
    }

def in_notebook():
    ''' Check if the code is executed in a Jupyter notebook
    '''
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


def showfig(fig):
    if in_notebook:
        try:
            from IPython.display import display
        except ImportError:
            pass

        plt.close()
        display(fig)

    else:
        plt.show()


def savefig(fig, path, settings={}, verbose=True):
    ''' Save a figure to a path
    Args
    ----
    fig : figure
        the figure to save
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified; it can be any existed or non-existed path,
          with or without a suffix; if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
    '''
    savefig_args = {'bbox_inches': 'tight'}
    savefig_args.update(settings)

    path = pathlib.Path(savefig_args['path'])
    savefig_args.pop('path')

    dirpath = path.parent
    if not dirpath.exists():
        dirpath.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f'Directory created at: "{dirpath}"')

    path_str = str(path)
    if path.suffix not in ['.eps', '.pdf', '.png', '.ps']:
        path = pathlib.Path(f'{path_str}.pdf')

    fig.savefig(path_str, **savefig_args)
    plt.close()

    if verbose:
        print(f'Figure saved at: "{str(path)}"')


def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
    ''' Enumerate plots

    Reference: https://stackoverflow.com/questions/22508590/enumerate-plots-in-matplotlib-figure
    '''
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.NaN, np.NaN,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()


def plot_proxies(df, year=np.arange(2001), lon_col='lon', lat_col='lat', type_col='type', time_col='time',
                 title=None, title_weight='normal', font_scale=1.5, markers_dict=None, colors_dict=None,
                 plot_timespan=None,  plot_xticks=[850, 1000, 1200, 1400, 1600, 1800, 2000],
                 figsize=[8, 10], projection='Robinson', proj_args={}, central_longitude=0, markersize=50,
                 plot_count=True, nrow=2, ncol=1, wspace=0.5, hspace=0.1,
                 lgd_ncol=1, lgd_anchor_upper=(1, -0.1), lgd_anchor_lower=(1, -0.05),lgd_frameon=False,
                 enumerate_ax=False, enumerate_prop={'weight': 'bold', 'size': 30}, p=PAGES2k,
                 enumerate_anchor_map=[0, 1], enumerate_anchor_count=[0, 1], map_grid_idx=0, count_grid_idx=-1):

    sns.set(style='darkgrid', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    if not plot_count:
        nrow = 1
        ncol = 1

    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=wspace, hspace=hspace)

    projection = CartopySettings.projection_dict[projection](**proj_args)
    ax = {}
    ax['map'] = plt.subplot(gs[map_grid_idx], projection=projection)

    if title:
        ax['map'].set_title(title, fontweight=title_weight)

    ax['map'].set_global()
    ax['map'].add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

    # plot markers by archive types
    if markers_dict is None:
        markers_dict = p.markers_dict
    if colors_dict is None:
        colors_dict = p.colors_dict

    s_plots = []
    type_names = []
    type_set = np.unique(df[type_col])
    max_count = []
    for ptype in type_set:
        selector = df[type_col] == ptype
        max_count.append(len(df[selector]))
        type_names.append(f'{ptype} (n={max_count[-1]})')
        lons = list(df[selector][lon_col])
        lats = list(df[selector][lat_col])
        s_plots.append(
            ax['map'].scatter(
                lons, lats, marker=markers_dict[ptype],
                c=colors_dict[ptype], edgecolor='k', s=markersize, transform=ccrs.Geodetic()
            )
        )

    ax['map'].legend(
        s_plots, type_names,
        scatterpoints=1,
        bbox_to_anchor=lgd_anchor_upper,
        loc='lower left',
        ncol=lgd_ncol,
        frameon=lgd_frameon,
    )

    if plot_count:
        ax['count'] = plt.subplot(gs[count_grid_idx])
        proxy_count = {}
        for index, row in df.iterrows():
            ptype = row[type_col]
            time = row[time_col]
            time = np.array([int(t) for t in time])
            time = time[~np.isnan(time)]
            time = np.sort(list(set(time)))  # remove the duplicates for monthly data

            if ptype not in proxy_count.keys():
                proxy_count[ptype] = np.zeros(np.size(year))

            for k in time:
                if k < np.max(year):
                    proxy_count[ptype][k] += 1

        cumu_count = np.zeros(np.size(year))
        cumu_last = np.copy(cumu_count)
        idx = np.argsort(max_count)
        for ptype in type_set[idx]:
            cumu_count += proxy_count[ptype]
            ax['count'].fill_between(
                year, cumu_last, cumu_count,
                color=colors_dict[ptype],
                label=f'{ptype}',
                alpha=0.8,
            )
            cumu_last = np.copy(cumu_count)

        ax['count'].set_xlabel('Year (AD)')
        ax['count'].set_ylabel('number of proxies')
        if plot_timespan is not None:
            ax['count'].set_xlim(plot_timespan)
            ax['count'].set_xticks(plot_xticks)
        handles, labels = ax['count'].get_legend_handles_labels()
        ax['count'].legend(handles[::-1], labels[::-1], frameon=lgd_frameon, bbox_to_anchor=lgd_anchor_lower, loc='lower left')

        if enumerate_ax:
            setlabel(ax['map'], '(a)', prop=enumerate_prop, bbox_to_anchor=enumerate_anchor_map)
            setlabel(ax['count'], '(b)', prop=enumerate_prop, bbox_to_anchor=enumerate_anchor_count)

    return fig, ax


def plot_proxy_age_map(df, lon_col='lon', lat_col='lat', type_col='type', time_col='time',
                       title=None, title_weight='normal', font_scale=1.5,
                       figsize=[12, 10], projection=ccrs.Robinson, central_longitude=0, markersize=150,
                       plot_cbar=True, marker_color=None, transform=ccrs.PlateCarree(), p=PAGES2k,
                       add_nino34_box=False, add_nino12_box=False, add_box=False, add_box_lf=None, add_box_ur=None):

    sns.set(style='ticks', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    projection = projection(central_longitude=central_longitude)
    ax_map = plt.subplot(projection=projection)

    if title:
        ax_map.set_title(title, fontweight=title_weight)

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

    if add_nino12_box:
        x, y = [-90, -90, -80, -80, -90], [0, -10, -10, 0, 0]
        ax_map.plot(x, y, '--', transform=transform, color='gray')

    if add_nino34_box:
        x, y = [-170, -170, -120, -120, -170], [5, -5, -5, 5, 5]
        ax_map.plot(x, y, '--', transform=transform, color='gray')

    if add_box:
        lf_lat, lf_lon = add_box_lf
        ur_lat, ur_lon = add_box_ur
        x, y = [lf_lon, lf_lon, ur_lon, ur_lon, lf_lon], [ur_lat, lf_lat, lf_lat, ur_lat, ur_lat]
        ax_map.plot(x, y, '--', transform=transform, color='gray')

    color_norm = Normalize(vmin=0, vmax=1000)

    cmap = cm.get_cmap('viridis_r', 10)
    cmap.set_under(sns.xkcd_rgb['cream'])
    cmap.set_over('black')

    # plot markers by archive types
    s_plots = []
    type_names = []
    type_set = np.unique(df[type_col])
    max_count = []
    for ptype in type_set:
        selector = df[type_col] == ptype
        max_count.append(len(df[selector]))
        type_names.append(f'{ptype} (n={max_count[-1]})')
        lons = list(df[selector][lon_col])
        lats = list(df[selector][lat_col])
        ages = []
        for idx, row in df[selector].iterrows():
            ages.append(1950-np.min(row['time']))

        if marker_color is None:
            s_plots.append(
                ax_map.scatter(
                    lons, lats, marker=p.markers_dict[ptype], cmap=cmap, norm=color_norm,
                    c=ages, edgecolor='k', s=markersize, transform=ccrs.Geodetic()
                )
            )
        else:
            s_plots.append(
                ax_map.scatter(
                    lons, lats, marker=p.markers_dict[ptype], cmap=cmap, norm=color_norm,
                    c=marker_color, edgecolor='k', s=markersize, transform=ccrs.Geodetic()
                )
            )

    if plot_cbar:
        cbar_lm = plt.colorbar(s_plots[0], orientation='vertical',
                               pad=0.05, aspect=10, extend='min',
                               ax=ax_map, fraction=0.05, shrink=0.5)

        cbar_lm.ax.set_title(r'age [yrs]', y=1.05)
        cbar_lm.set_ticks([0, 200, 400, 600, 800, 1000])

    return fig


def plot_field_map(field_var, lat, lon, levels=50, add_cyclic_point=True,
                   title=None, title_size=20, title_weight='normal', figsize=[10, 8],
                   site_lats=None, site_lons=None, site_marker='o',
                   site_markersize=50, site_color=sns.xkcd_rgb['amber'],
                   projection='Robinson', transform=ccrs.PlateCarree(),
                   proj_args={}, latlon_range=None, central_longitude=0,
                   lon_ticks=[60, 120, 180, 240, 300], lat_ticks=[-90, -45, 0, 45, 90],
                   land_color=sns.xkcd_rgb['light grey'], ocean_color=sns.xkcd_rgb['light grey'],
                   land_zorder=None, ocean_zorder=None, signif_values=None, signif_range=[0.05, 9999], hatch='..',
                   clim=None, cmap=None, cmap_under=None, cmap_over=None, extend=None, mode='latlon', add_gridlines=False,
                   make_cbar=True, cbar_labels=None, cbar_pad=0.05, cbar_orientation='vertical', cbar_aspect=10,
                   cbar_fraction=0.15, cbar_shrink=0.5, cbar_title=None, font_scale=1.5, plot_type=None,
                   fig=None, ax=None):

    if plot_type == 'corr':
        clim = [-1, 1] if clim is None else clim
        levels = np.linspace(-1, 1, 21) if levels is None else levels
        cbar_labels = np.linspace(-1, 1, 11) if cbar_labels is None else cbar_labels
        cbar_title = 'r' if cbar_title is None else cbar_title
        extend = 'neither' if extend is None else extend
        cmap = 'RdBu_r' if cmap is None else cmap
    elif plot_type == 'R2':
        clim = [0, 1] if clim is None else clim
        levels = np.linspace(0, 1, 21) if levels is None else levels
        cbar_labels = np.linspace(0, 1, 11) if cbar_labels is None else cbar_labels
        cbar_title = r'R$^2$' if cbar_title is None else cbar_title
        extend = 'neither' if extend is None else extend
        cmap = 'Reds' if cmap is None else cmap
    else:
        extend = 'both' if extend is None else extend
        cmap = 'RdBu_r' if cmap is None else cmap

    if add_cyclic_point:
        if mode == 'latlon':
            field_var_c, lon_c = cutil.add_cyclic_point(field_var, lon)
            if signif_values is not None:
                signif_values_c, lon_c = cutil.add_cyclic_point(signif_values, lon)
            lat_c = lat
        elif mode == 'mesh':
            if len(np.shape(lat)) == 1:
                lon, lat = np.meshgrid(lon, lat, sparse=False, indexing='xy')
            if central_longitude == 180:
                lon = np.mod(lon+180, 360) - 180

            nx, ny = np.shape(field_var)

            lon_c = np.ndarray((nx, ny+1))
            lat_c = np.ndarray((nx, ny+1))
            field_var_c = np.ndarray((nx, ny+1))
            if signif_values is not None:
                signif_values_c = np.ndarray((nx, ny+1))

            lon_c[:, :-1] = lon
            lon_c[:, -1] = lon[:, 0]

            lat_c[:, :-1] = lat
            lat_c[:, -1] = lat[:, 0]

            field_var_c[:, :-1] = field_var
            field_var_c[:, -1] = field_var[:, 0]

            if signif_values is not None:
                signif_values_c[:, :-1] = signif_values
                signif_values_c[:, -1] = signif_values[:, 0]
    else:
        field_var_c, lat_c, lon_c = field_var, lat, lon
        if signif_values is not None:
            signif_values_c = signif_values

    if ax is None or fig is None:
        sns.set(style='ticks', font_scale=font_scale)
        fig = plt.figure(figsize=figsize)

        projection = CartopySettings.projection_dict[projection](**proj_args)
        ax = plt.subplot(projection=projection)

    if title:
        plt.title(title, fontsize=title_size, fontweight=title_weight)

    if latlon_range:
        lon_min, lon_max, lat_min, lat_max = latlon_range
        ax.set_extent(latlon_range, crs=transform)
        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        lon_ticks = np.array(lon_ticks)
        lat_ticks = np.array(lat_ticks)
        mask_lon = (lon_ticks >= lon_min) & (lon_ticks <= lon_max)
        mask_lat = (lat_ticks >= lat_min) & (lat_ticks <= lat_max)
        ax.set_xticks(lon_ticks[mask_lon], crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks[mask_lat], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor=land_color, zorder=land_zorder)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor=ocean_color, zorder=ocean_zorder)
    ax.coastlines()

    if add_gridlines:
        ax.gridlines(edgecolor='gray', linestyle=':', crs=transform)

    cmap = plt.get_cmap(cmap)
    if cmap_under is not None:
        cmap.set_under(cmap_under)
    if cmap_over is not None:
        cmap.set_over(cmap_over)

    if mode == 'latlon':
        im = ax.contourf(lon_c, lat_c, field_var_c, levels, transform=transform, cmap=cmap, extend=extend)

    elif mode == 'mesh':
        if type(levels) is int:
            levels = MaxNLocator(nbins=levels).tick_values(np.nanmax(field_var_c), np.nanmin(field_var_c))
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = ax.pcolormesh(lon_c, lat_c, field_var_c, transform=transform, cmap=cmap, norm=norm)

    if clim:
        im.set_clim(clim)

    if signif_values is not None:
        ax.contourf(lon_c, lat_c, signif_values_c, signif_range, transform=transform, hatches=[hatch], colors='none')

    if make_cbar:
        cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
                        fraction=cbar_fraction, shrink=cbar_shrink)

        if cbar_labels is not None:
            cbar.set_ticks(cbar_labels)

        if cbar_title:
            cbar.ax.set_title(cbar_title)

    if site_lats is not None and site_lons is not None:
        if type(site_lats) is not dict:
            ax.scatter(site_lons, site_lats, s=site_markersize, c=site_color, marker=site_marker, edgecolors='k',
                       zorder=99, transform=transform)
        else:
            for name in site_lats.keys():
                ax.scatter(site_lons[name], site_lats[name], s=site_markersize[name], c=site_color[name], marker=site_marker[name], edgecolors='k',
                           zorder=99, transform=transform)

    return fig, ax


def plot_scatter_map(df, lon_col='lon', lat_col='lat', type_col='type', value_col='value',
                     vmin=None, vmax=None, cmap_str='viridis_r', edge_lw=1,
                     bad_clr='grey', under_clr=sns.xkcd_rgb['cream'], over_clr='black',
                     title=None, title_weight='normal', font_scale=1.5,
                     projection='Robinson', transform=ccrs.PlateCarree(), proj_args={},
                     figsize=[12, 10], markersize=150, num_color=10,
                     plot_cbar=True, cbar_title=None, cbar_ticks=None, cbar_ticklabels=None,
                     cbar_extend_mode='neither', add_legend=True, lgd_args={'frameon': False}):

    sns.set(style='ticks', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    projection = CartopySettings.projection_dict[projection](**proj_args)
    ax_map = plt.subplot(projection=projection)

    if title:
        ax_map.set_title(title, fontweight=title_weight)

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

    if vmin is None:
        vmin = np.min(df[value_col].values)
    if vmax is None:
        vmax = np.min(df[value_col].values)

    color_norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = cm.get_cmap(cmap_str, num_color)
    cmap.set_under(under_clr)
    cmap.set_over(over_clr)
    cmap.set_bad(bad_clr)

    # plot markers by archive types
    s_plots = []
    type_names = []
    type_set = np.unique(df[type_col])
    max_count = []
    for ptype in type_set:
        selector = df[type_col] == ptype
        max_count.append(len(df[selector]))
        type_names.append(f'{ptype} (n={max_count[-1]})')
        lons = list(df[selector][lon_col])
        lats = list(df[selector][lat_col])
        values = list(df[selector][value_col])
        s_plots.append(
            ax_map.scatter(
                lons, lats, marker=PAGES2k.markers_dict[ptype], cmap=cmap, norm=color_norm, linewidths=edge_lw,
                c=values, edgecolor='k', s=markersize, transform=ccrs.Geodetic(), label=f'{ptype} (n={np.size(lats)})',
            )
        )

    if add_legend:
        ax_map.legend(**lgd_args)
        leg = ax_map.get_legend()
        for l in leg.legendHandles:
            l.set_color('grey')

    if plot_cbar:
        cbar_lm = plt.colorbar(s_plots[0], orientation='vertical',
                               pad=0.05, aspect=10, extend=cbar_extend_mode,
                               ax=ax_map, fraction=0.05, shrink=0.5)

        cbar_lm.ax.set_title(cbar_title, y=1.05)
        cbar_lm.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar_lm.set_ticklabels(cbar_ticklabels)

    return fig, ax_map


def plot_gmt_vs_inst(gmt_qs, year, ana_pathdict,
                     verif_yrs=np.arange(1880, 2001), ref_period=[1951, 1980],
                     var='gmt', lmr_label='LMR', style='ticks', ylim=[-0.7, 0.8]):
    if np.shape(gmt_qs)[-1] == 1:
        nt = np.size(gmt_qs)
        gmt_qs_new = np.ndarray((nt, 3))
        for i in range(3):
            gmt_qs_new[:, i] = gmt_qs

        gmt_qs = gmt_qs_new

    syear, eyear = verif_yrs[0], verif_yrs[-1]
    mask = (year >= syear) & (year <= eyear)
    mask_ref = (year >= ref_period[0]) & (year <= ref_period[-1])
    lmr_gmt = gmt_qs[mask] - np.nanmean(gmt_qs[mask_ref, 1])  # remove the mean w.r.t. the ref_period

    var_str = {
        'gmt': 'gm',
        'nhmt': 'nhm',
        'shmt': 'shm',
    }
    inst_gmt, inst_time = utils.load_inst_analyses(
        ana_pathdict, var=var_str[var], verif_yrs=verif_yrs, ref_period=ref_period)

    consensus_yrs = np.copy(verif_yrs)
    for name in ana_pathdict.keys():
        mask_ref = (inst_time[name] >= ref_period[0]) & (inst_time[name] <= ref_period[-1])
        inst_gmt[name] -= np.nanmean(inst_gmt[name][mask_ref])  # remove the mean w.r.t. the ref_period

        overlap_yrs = np.intersect1d(consensus_yrs, inst_time[name])
        ind_inst = np.searchsorted(inst_time[name], overlap_yrs)
        consensus_yrs = inst_time[name][ind_inst]

    consensus_gmt = np.zeros(np.size(consensus_yrs))
    for name in ana_pathdict.keys():
        overlap_yrs = np.intersect1d(consensus_yrs, inst_time[name])
        ind_inst = np.searchsorted(inst_time[name], overlap_yrs)
        consensus_gmt += inst_gmt[name][ind_inst]/len(ana_pathdict.keys())

    inst_gmt['consensus'] = consensus_gmt
    inst_time['consensus'] = consensus_yrs

    # stats
    corr_vs_lmr = {}
    ce_vs_lmr = {}
    for name in inst_gmt.keys():
        print(f'Calculating corr and CE against LMR for {name}')
        overlap_yrs = np.intersect1d(verif_yrs, inst_time[name])
        ind_lmr = np.searchsorted(verif_yrs, overlap_yrs)
        ind_inst = np.searchsorted(inst_time[name], overlap_yrs)
        ts_inst = inst_gmt[name][ind_inst]

        ts_lmr = lmr_gmt[ind_lmr, 1]
        corr_vs_lmr[name] = np.corrcoef(ts_inst, ts_lmr)[1, 0]
        ce_vs_lmr[name] = utils.coefficient_efficiency(ts_inst, ts_lmr)

    sns.set(style=style, font_scale=2)
    fig, ax = plt.subplots(figsize=[16, 10])

    ax.plot(verif_yrs, lmr_gmt[:, 1], '-', lw=3, color=sns.xkcd_rgb['black'], alpha=1, label=lmr_label)
    ax.fill_between(verif_yrs, lmr_gmt[:, 0], lmr_gmt[:, -1], color=sns.xkcd_rgb['black'], alpha=0.1)
    for name in inst_gmt.keys():
        ax.plot(inst_time[name], inst_gmt[name], '-', alpha=1,
                label=f'{name} (corr={corr_vs_lmr[name]:.2f}; CE={ce_vs_lmr[name]:.2f})')

    ax.set_xlim([syear, eyear])
    ax.set_ylim(ylim)
    ax.set_ylabel('Temperature anomaly (K)')
    ax.set_xlabel('Year (AD)')

    ax_title = {
        'gmt': 'Global mean temperature',
        'nhmt': 'NH mean temperature',
        'shmt': 'SH mean temperature',
    }

    ax.set_title(ax_title[var])
    ax.legend(frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, corr_vs_lmr, ce_vs_lmr


def plot_corr_ce(corr_dict, ce_dict, lw=3, ms=10,
                 colors=[
                     sns.xkcd_rgb['denim blue'],
                     sns.xkcd_rgb['pale red'],
                     sns.xkcd_rgb['medium green'],
                     sns.xkcd_rgb['amber'],
                     sns.xkcd_rgb['purpleish'],
                 ],
                 corr_ls='-o',
                 ce_ls='--o',
                 lgd_ncol=1, lgd_loc='upper right',
                 lgd_bbox_to_anchor=(1.3, 1., 0, 0),
                 ylim=[0, 1],
                 ):
    exp_names = list(corr_dict.keys())
    inst_names = list(corr_dict[exp_names[0]].keys())

    sns.set(style="darkgrid", font_scale=2)
    fig, ax = plt.subplots(figsize=[16, 10])

    inst_corr = {}
    inst_ce = {}
    for exp_name in exp_names:
        inst_corr[exp_name] = []
        inst_ce[exp_name] = []

    inst_cat = []
    for inst_name in inst_names:
        inst_cat.append(inst_name)
        for exp_name in exp_names:
            inst_corr[exp_name].append(corr_dict[exp_name][inst_name])
            inst_ce[exp_name].append(ce_dict[exp_name][inst_name])

    for i, exp_name in enumerate(exp_names):
        ax.plot(inst_cat, inst_corr[exp_name], corr_ls, lw=lw, ms=ms, color=colors[i % len(colors)],
                alpha=1, label=f'corr ({exp_name})')
        ax.plot(inst_cat, inst_ce[exp_name], ce_ls, lw=lw, ms=ms, color=colors[i % len(colors)],
                alpha=1, label=f'CE ({exp_name})')

    ax.set_ylim(ylim)
    ax.set_title('corr and CE against LMR')
    ax.set_ylabel('coefficient')
    ax.legend(frameon=False, ncol=lgd_ncol, loc=lgd_loc, bbox_to_anchor=lgd_bbox_to_anchor)

    return fig


def plot_rolling_phase(time, value, window, factor=1, lw=1, ms=3,
                       clr_value=sns.xkcd_rgb['dark grey'], clr_pos=sns.xkcd_rgb['pale red'], clr_neg=sns.xkcd_rgb['denim blue'],
                       ax=None, figsize=[8, 3], xlabel=None, ylabel=None, xlim=None, ylim=None, title=None, mute=False,
                       xticks=None, yticks=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    rolling_std = pd.Series(value).rolling(window).std().values
    rolling_mean = pd.Series(value).rolling(window).mean().values

    ax.plot(time, value, color=clr_value, lw=lw, marker='o', markersize=ms)
    ax.plot(time-window//2, rolling_mean+factor*rolling_std, color=clr_pos, lw=lw)
    ax.plot(time-window//2, rolling_mean-factor*rolling_std, color=clr_neg, lw=lw)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)

    if 'fig' in locals():
        if not mute:
            showfig(fig)
        return fig, ax
    else:
        return ax

def plot_rolling_phase_comparison(time1, value1, time2, value2, window, factor=1, lw=1,
                                  clr_value=sns.xkcd_rgb['dark grey'], clr_norm=sns.xkcd_rgb['grey'],
                                  clr_pos=sns.xkcd_rgb['pale red'], clr_neg=sns.xkcd_rgb['denim blue'],
                                  figsize=[12, 6], xlabel1=None, xlabel2=None, ylabel1=None, ylabel2=None,
                                  xlim=None, ylim=None, title=None, xticks=None, yticks=None, shade_kws=None,
                                  signif_method='isospec', events=None):

    shade_kwargs = {'alpha':0.5, 'width': 1}
    if shade_kws is not None:
        shade_kwargs.update(shade_kws)

    fig, ax = plt.subplots(2, figsize=figsize, sharex=True)
    plot_rolling_phase(time1, value1, window, clr_value=clr_value, clr_pos=clr_pos, clr_neg=clr_neg, xlabel=xlabel1, ylabel=ylabel1, ax=ax[0], lw=lw, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    plot_rolling_phase(time2, value2, window, clr_value=clr_value, clr_pos=clr_pos, clr_neg=clr_neg, xlabel=xlabel2, ylabel=ylabel2, ax=ax[1], lw=lw, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    ax[0].tick_params(labelbottom=True)

    res_dict = utils.calc_phase_consistent_rate(time1, value1, time2, value2, window, factor=factor)
    idx_pos = res_dict['idx_pos']
    idx_neg = res_dict['idx_neg']
    idx_norm = res_dict['idx_norm']
    idx_pos_1 = res_dict['idx_pos_1']
    idx_neg_1 = res_dict['idx_neg_1']
    idx_norm_1 = res_dict['idx_norm_1']
    idx_pos_2 = res_dict['idx_pos_2']
    idx_neg_2 = res_dict['idx_neg_2']
    idx_norm_2 = res_dict['idx_norm_2']
    cons_rate = res_dict['cons_rate']
    t = res_dict['t']

    ax[0].bar(t[idx_pos], 1, color=clr_pos, transform=ax[0].get_xaxis_transform(), **shade_kwargs)
    ax[0].bar(t[idx_neg], 1, color=clr_neg, transform=ax[0].get_xaxis_transform(), **shade_kwargs)
    ax[0].bar(t[idx_norm], 1, color=clr_norm, transform=ax[0].get_xaxis_transform(), **shade_kwargs)
    ax[1].bar(t[idx_pos], 1, color=clr_pos, transform=ax[1].get_xaxis_transform(), **shade_kwargs)
    ax[1].bar(t[idx_neg], 1, color=clr_neg, transform=ax[1].get_xaxis_transform(), **shade_kwargs)
    ax[1].bar(t[idx_norm], 1, color=clr_norm, transform=ax[1].get_xaxis_transform(), **shade_kwargs)

    if events is not None:
        events = np.array(events)
        mask = (events>=t[0]) & (events<=t[-1])
        for event in events[mask]:
            if event in t[idx_pos_1]:
                clr_event = clr_pos
            elif event in t[idx_neg_1]:
                clr_event = clr_neg
            elif event in t[idx_norm_1]:
                clr_event = clr_norm
            ax_ylim = ax[0].get_ylim()
            ax[0].axvline(x=event, color=clr_event, ls='-', zorder=99, lw=1)
            ax[0].text(event, 1.05*ax_ylim[-1], event, horizontalalignment='center', color=clr_event)

            if event in t[idx_pos_2]:
                clr_event = clr_pos
            elif event in t[idx_neg_2]:
                clr_event = clr_neg
            elif event in t[idx_norm_2]:
                clr_event = clr_norm
            ax_ylim = ax[1].get_ylim()
            ax[1].axvline(x=event, color=clr_event, ls='-', zorder=99, lw=1)
            ax[1].text(event, 1.05*ax_ylim[-1], event, horizontalalignment='center', color=clr_event)

    signif_test = utils.signif_test_consistent_rate(time1, value1, time2, value2, window, factor=factor, qs=[0.95], method=signif_method)
    signif_q95 = signif_test['cons_rate_qs'][0]
    if title is None:
        fig.suptitle(f'Timespan: {int(t[0])}-{int(t[-1])}; Rolling window: {window}; Consistency rate: {cons_rate:.2f} ({signif_method} 95% = {signif_q95:.2f})')

    return fig, ax

def plot_candlesticks(time, value, upcolor=sns.xkcd_rgb['medium green'], downcolor=sns.xkcd_rgb['pale red'],
                      ax=None, bar_kws=None, figsize=[8, 3], xlabel=None, ylabel=None, xlim=None, ylim=None, title=None,
                      mute=False, xticks=None, yticks=None):
    bar_kwargs = {'alpha':1, 'width': 1}
    if bar_kws is not None:
        bar_kwargs.update(bar_kws)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    value_left = value[:-1]
    value_right = value[1:]
    idx_up = []
    idx_down = []
    for idx in range(np.size(value_right)):
        if value_right[idx] > value_left[idx]:
            idx_up.append(idx)
        else:
            idx_down.append(idx)

    ax.bar(time[1:][idx_up], value_right[idx_up]-value_left[idx_up], bottom=value_left[idx_up], color=upcolor, **bar_kwargs)
    ax.bar(time[1:][idx_down], value_left[idx_down]-value_right[idx_down], bottom=value_right[idx_down], color=downcolor, **bar_kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title)

    if 'fig' in locals():
        if not mute:
            showfig(fig)
        return fig, ax
    else:
        return ax

def plot_candlesticks_comparison(
        time1, value1, time2, value2, signif_method='isospec',
        upcolor=sns.xkcd_rgb['medium green'], downcolor=sns.xkcd_rgb['pale red'],
        bar_kws=None, shade_kws=None, figsize=[14, 6], xlabel1=None, ylabel1=None, xlabel2=None, ylabel2=None,
        xlim=None, ylim=None, xticks=None, yticks=None, title=None, mute=False, savefig_path=None, savefig_settigns=None,
        events=None):
    shade_kwargs = {'alpha':0.3, 'width': 1}
    if shade_kws is not None:
        shade_kwargs.update(shade_kws)

    fig, ax = plt.subplots(2, figsize=figsize, sharex=True)

    plot_candlesticks(time1, value1, upcolor=upcolor, downcolor=downcolor, bar_kws=bar_kws, xlabel=xlabel1, ylabel=ylabel1, xlim=xlim, ylim=ylim, ax=ax[0], xticks=xticks, yticks=yticks)
    plot_candlesticks(time2, value2, upcolor=upcolor, downcolor=downcolor, bar_kws=bar_kws, xlabel=xlabel2, ylabel=ylabel2, xlim=xlim, ylim=ylim, ax=ax[1], xticks=xticks, yticks=yticks)
    ax[0].tick_params(labelbottom=True)

    res_dict = utils.calc_sync_rate(value1, value2)
    consistent_up = res_dict['consistent_up']
    consistent_down = res_dict['consistent_down']
    sync_rate = res_dict['sync_rate']
    sync_up_rate_1 = res_dict['sync_up_rate_1']
    sync_up_rate_2 = res_dict['sync_up_rate_2']
    sync_down_rate_1 = res_dict['sync_down_rate_1']
    sync_down_rate_2 = res_dict['sync_down_rate_2']
    tot_length = res_dict['tot_length']
    idx_up = res_dict['idx_up']
    idx_down = res_dict['idx_down']
    idx_up_1 = res_dict['idx_up_1']
    idx_down_1 = res_dict['idx_down_1']
    idx_up_2 = res_dict['idx_up_2']
    idx_down_2 = res_dict['idx_down_2']

    ax[0].bar(time1[1:][idx_up], 1, color=upcolor, transform=ax[0].get_xaxis_transform(), **shade_kwargs)
    ax[0].bar(time1[1:][idx_down], 1, color=downcolor, transform=ax[0].get_xaxis_transform(), **shade_kwargs)
    ax[1].bar(time2[1:][idx_up], 1, color=upcolor, transform=ax[1].get_xaxis_transform(), **shade_kwargs)
    ax[1].bar(time2[1:][idx_down], 1, color=downcolor, transform=ax[1].get_xaxis_transform(), **shade_kwargs)
    
    if events is not None:
        events = np.array(events)
        mask = (events>=time1[0]) & (events<=time1[-1])
        for event in events[mask]:
            if event in time1[1:][idx_up_1]:
                clr_event = upcolor
            elif event in time1[1:][idx_down_1]:
                clr_event = downcolor
            ax_ylim = ax[0].get_ylim()
            ax[0].axvline(x=event, color=clr_event, ls='-', zorder=99, lw=1)
            ax[0].text(event, 1.05*ax_ylim[-1], event, horizontalalignment='center', color=clr_event)

            if event in time1[1:][idx_up_2]:
                clr_event = upcolor
            elif event in time1[1:][idx_down_2]:
                clr_event = downcolor
            ax_ylim = ax[1].get_ylim()
            ax[1].axvline(x=event, color=clr_event, ls='-', zorder=99, lw=1)
            ax[1].text(event, 1.05*ax_ylim[-1], event, horizontalalignment='center', color=clr_event)

    signif_test = utils.signif_test_sync_rate(value1, value2, qs=[0.95], method=signif_method)
    signif_q95 = signif_test['sync_rate_qs'][0]
    if title is None:
        fig.suptitle(
            f'Timespan: {int(time1[0])}-{int(time1[-1])}; Synchronization rate: {sync_rate:.2f} ({signif_method} 95% = {signif_q95:.2f})'
        )
    
    if not mute:
        showfig(fig)

    return fig, ax

def plot_ts_from_jobs(
    exp_dir, time_span=(0, 2000), savefig_path=None,
    plot_vars=['tas_sfc_Amon_gm_ens', 'tas_sfc_Amon_nhm_ens', 'tas_sfc_Amon_shm_ens'],
    qs=[0.025, 0.25, 0.5, 0.75, 0.975], pannel_size=[10, 4], ylabel='T anom. (K)',
    font_scale=1.5, hspace=0.5, ylim=[-1, 1], color=sns.xkcd_rgb['pale red'],
    title=None, plot_title=True, title_y=1,
    plot_lgd=True, lgd_ncol=3, lgd_bbox_to_anchor=None, lgd_order=[0, 2, 3, 1], style='ticks',
    bias_correction=False,
    ref_value=None, ref_time=None, ref_color='k', ref_ls='-', ref_label='reference', ref_alpha=1,
    plot_proxies=False, count_proxies=None, clr_proxies=None, label_proxies=None, time_proxies=np.arange(2001),
    anchor_proxies=(1.3, 0.05), count_min=0.7, count_max=100, load_num=None, proxy_count_style='log',
):
    ''' Plot timeseries

    Args:
        exp_dir (str): the path of the results directory that contains subdirs r0, r1, ...

    Returns:
        fig (figure): the output figure
    '''
    def make_list(item, nvars=1):
        if type(item) is not list:
            item = [item]
            if len(item) == 1:
                for i in range(1, nvars):
                    item.append(item[0])

        return item

    plot_vars = make_list(plot_vars)
    nvars = len(plot_vars)

    time_span = make_list(time_span, nvars)
    plot_title = make_list(plot_title, nvars)
    ylabel = make_list(ylabel, nvars)

    # load data
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    nvar = len(plot_vars)

    sns.set(style=style, font_scale=font_scale)
    fig = plt.figure(figsize=[pannel_size[0], pannel_size[1]*nvar])

    ax_title = {
        'tas_sfc_Amon_gm_ens': 'Global mean temperature',
        'tas_sfc_Amon_nhm_ens': 'NH mean temperature',
        'tas_sfc_Amon_shm_ens': 'SH mean temperature',
        'nino1+2': 'Annual Niño 1+2 Index',
        'nino3': 'Annual Niño 3 Index',
        'nino3.4': 'Annual Niño 3.4 Index',
        'nino4': 'Annual Niño 4 Index',
        'tpi': 'Annual Tripole Index',
    }

    ax = {}
    for plot_i, var in enumerate(plot_vars):

        ts_qs, year = utils.load_ts_from_jobs(exp_dir, qs, var=var, load_num=load_num)

        mask = (year >= time_span[plot_i][0]) & (year <= time_span[plot_i][-1])
        ts_qs = ts_qs[mask, :]
        year = year[mask]

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        ax[plot_i] = plt.subplot(gs[plot_i, 0])
        if qs[2] == 0.5:
            label = 'median'
        else:
            label = f'{qs[2]*100}%'

        if title is None and var in ax_title.keys():
            title_str = ax_title[var]
        else:
            title_str = title

        ax[plot_i].plot(year, ts_qs[:, 2], '-', color=color, alpha=1, label=f'{label}')
        ax[plot_i].fill_between(year, ts_qs[:, -2], ts_qs[:, 1], color=color, alpha=0.5,
                        label=f'{qs[1]*100}% to {qs[-2]*100}%')
        ax[plot_i].fill_between(year, ts_qs[:, -1], ts_qs[:, 0], color=color, alpha=0.1,
                        label=f'{qs[0]*100}% to {qs[-1]*100}%')
        ax[plot_i].set_ylabel(ylabel[plot_i])
        ax[plot_i].set_xlabel('Year (AD)')
        ax[plot_i].set_ylim(ylim)

        if type(ref_value) is list and len(ref_value) > 1:
            ref_v = ref_value[plot_i]
        else:
            ref_v = ref_value

        if type(ref_time) is list and len(ref_time) > 1:
            ref_t = ref_time[plot_i]
        else:
            ref_t = ref_time

        if type(ref_label) is list and len(ref_label) > 1:
            ref_l = ref_label[plot_i]
        else:
            ref_l = ref_label

        if ref_v is not None:
            mask = (ref_t >= time_span[plot_i][0]) & (ref_t <= time_span[plot_i][-1])
            ref_v = ref_v[mask]
            ref_t = ref_t[mask]

            if bias_correction:
                ts_qs_mean = np.nanmean(ts_qs, axis=0)
                ts_qs -= ts_qs_mean

                ref_mean = np.nanmean(ref_v)
                ref_v -= ref_mean

            overlap_yrs = np.intersect1d(ref_t, year)
            ind_ref = np.searchsorted(ref_t, overlap_yrs)
            ind_ts = np.searchsorted(year, overlap_yrs)
            corr = np.corrcoef(ts_qs[ind_ts, 2], ref_v[ind_ref])[1, 0]
            ce = utils.coefficient_efficiency(ref_v[ind_ref], ts_qs[ind_ts, 2])

            ax[plot_i].plot(ref_t, ref_v, ls=ref_ls, color=ref_color, alpha=ref_alpha, label=f'{ref_l}')
            if plot_title[plot_i]:
                ax[plot_i].set_title(f'{title_str} (corr={corr:.2f}; CE={ce:.2f})', y=title_y)

            if plot_i == 0:
                if plot_lgd:
                    if lgd_order:
                        handles, labels = ax[plot_i].get_legend_handles_labels()
                        ax[plot_i].legend(
                            [handles[idx] for idx in lgd_order], [labels[idx] for idx in lgd_order],
                            loc='upper center', ncol=lgd_ncol, frameon=False, bbox_to_anchor=lgd_bbox_to_anchor,
                        )
                    else:
                        ax[plot_i].legend(loc='upper center', ncol=lgd_ncol, frameon=False, bbox_to_anchor=lgd_bbox_to_anchor)
        else:
            ax[plot_i].set_title(title_str, y=title_y)
            if plot_lgd:
                ax[plot_i].legend(loc='upper center', ncol=lgd_ncol, frameon=False, bbox_to_anchor=lgd_bbox_to_anchor)

        ax[plot_i].spines['right'].set_visible(False)
        ax[plot_i].spines['top'].set_visible(False)

    if plot_proxies:
        # extend ylim but keep the old tick & ticklabels
        ax[nvars-1].set_ylim([ylim[0]-2, ylim[-1]])
        yticks = ax[nvars-1].get_yticks()
        mask = (yticks>=ylim[0]) & (yticks<=ylim[-1])
        ax[nvars-1].set_yticks(yticks[mask])

        ax[nvars] = ax[nvars-1].twinx()
        ct_last = np.zeros(np.size(count_proxies[0]))
        ct_total = np.zeros(np.size(count_proxies[0]))
        for p_idx, ct in enumerate(count_proxies):
            ct_total = ct_total + ct
            ax[nvars].fill_between(time_proxies, ct_total, ct_last, color=clr_proxies[p_idx], label=label_proxies[p_idx])
            ct_last = np.copy(ct_total)

        max_proxy_num = np.max(ct_total)
        if proxy_count_style == 'log':
            yticklabels_default = np.array([1, 10, 100, 1000, 10000])
            ax[nvars].set_yscale('log')
            count_max = 1e10
        else:
            yticklabels_default = np.array([0, 50, 100, 200, 500, 1000])

        ax[nvars].set_ylim(count_min, count_max)

        upper_bd = yticklabels_default[yticklabels_default>max_proxy_num][0]
        mask = yticklabels_default <= upper_bd
        ax[nvars].set_yticks(yticklabels_default[mask])
        ax[nvars].set_yticklabels(yticklabels_default[mask], fontsize=15)
        ax[nvars].spines['left'].set_visible(False)
        ax[nvars].spines['right'].set_visible(False)
        ax[nvars].spines['top'].set_visible(False)
        ax[nvars].set_ylabel('# of records', y=0.2, fontsize=15)
        ax[nvars].set_xlim(*time_span[nvars-1])
        ax[nvars].legend(frameon=False, loc='lower right', bbox_to_anchor=anchor_proxies, fontsize=15)

    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


def plot_vslite_params(lat_obs, lon_obs, T1, T2, M1, M2,
                       T1_ticks=None, T2_ticks=None, M1_ticks=None, M2_ticks=None, save_path=None):
    sns.set(style='ticks', font_scale=2)
    fig = plt.figure(figsize=[18, 12])
    gs = mpl.gridspec.GridSpec(2, 2)
    gs.update(wspace=0.1, hspace=0.1)

    map_proj = ccrs.Robinson()
    # map_proj = ccrs.PlateCarree()
    # map_proj = ccrs.Mercator()
    # map_proj = ccrs.Miller()
    # map_proj = ccrs.Mollweide()
    # map_proj = ccrs.EqualEarth()

    pad = 0.05
    fraction = 0.05

    # T1
    ax1 = plt.subplot(gs[0, 0], projection=map_proj)
    ax1.set_title('T1')
    ax1.set_global()
    ax1.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax1.gridlines(edgecolor='gray', linestyle=':')
    z = T1
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax1.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Reds', s=20, transform=ccrs.Geodetic()
    )
    cbar1 = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=pad, fraction=fraction, extend='both')
    if T1_ticks:
        cbar1.set_ticks(T1_ticks)

    # T2
    ax2 = plt.subplot(gs[0, 1], projection=map_proj)
    ax2.set_title('T2')
    ax2.set_global()
    ax2.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax2.gridlines(edgecolor='gray', linestyle=':')
    z = T2
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax2.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Reds', s=20, transform=ccrs.Geodetic()
    )
    cbar2 = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=pad, fraction=fraction, extend='both')
    if T2_ticks:
        cbar2.set_ticks(T2_ticks)

    # M1
    ax3 = plt.subplot(gs[1, 0], projection=map_proj)
    ax3.set_title('M1')
    ax3.set_global()
    ax3.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax3.gridlines(edgecolor='gray', linestyle=':')
    z = M1
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax3.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Blues', s=20, transform=ccrs.Geodetic()
    )
    cbar3 = fig.colorbar(im, ax=ax3, orientation='horizontal', pad=pad, fraction=fraction, extend='both')
    if M1_ticks:
        cbar3.set_ticks(M1_ticks)

    # M2
    ax4 = plt.subplot(gs[1, 1], projection=map_proj)
    ax4.set_title('M2')
    ax4.set_global()
    ax4.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax4.gridlines(edgecolor='gray', linestyle=':')
    z = M2
    norm = mpl.colors.Normalize(vmin=np.min(z), vmax=np.max(z))
    im = ax4.scatter(
        lon_obs, lat_obs, marker='o', norm=norm,
        c=z, cmap='Blues', s=20, transform=ccrs.Geodetic()
    )
    cbar4 = fig.colorbar(im, ax=ax4, orientation='horizontal', pad=pad, fraction=fraction, extend='both')
    if M2_ticks:
        cbar4.set_ticks(M2_ticks)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig


def plot_volc_events(gmt, event_yrs, start_yr=0, before=3, after=10, highpass=False, qs=[0.05, 0.5, 0.95],
                     pannel_size=[4, 4], grid_nrow=5, grid_ncol=5, clr=sns.xkcd_rgb['pale red'],
                     ylim=[-0.6, 0.3], ylabel='T anom. (K)'):
    ''' plot gmt over the periors of volcanic eruption events

    Args:
        gmt (2d array): the mean temperature timeseries in shape of (time, ensemble)
    '''
    # Superposed Epoch Analysis
    Xevents, Xcomp, tcomp = utils.sea(gmt, event_yrs, start_yr=start_yr, before=before, after=after, highpass=highpass)

    n_events = len(event_yrs)
    if grid_nrow*grid_ncol <= n_events:
        grid_nrow = n_events // grid_ncol + 1

    gs = gridspec.GridSpec(grid_nrow, grid_ncol)
    gs.update(wspace=0.5, hspace=1)

    sns.set(style="darkgrid", font_scale=2)
    fig = plt.figure(figsize=[pannel_size[1]*grid_ncol, pannel_size[0]*grid_nrow])

    ax = {}
    for i in range(n_events):
        ax[i] = plt.subplot(gs[i])
        ax[i].set_title(f'{event_yrs[i]} (AD)')
        if len(np.shape(Xevents)) == 3:
            Xevents_qs = mquantiles(Xevents[:, :, i], qs, axis=-1)
            ax[i].plot(tcomp, Xevents_qs[:, 1], '-', lw=3, color=clr)
            ax[i].fill_between(tcomp, Xevents_qs[:, 0], Xevents_qs[:, -1], alpha=0.3, color=clr)
        else:
            ax[i].plot(tcomp, Xevents[:, i], '-', lw=3, color=clr)

        ax[i].axvline(x=0, ls=':', color='grey')
        ax[i].axhline(y=0, ls=':', color='grey')
        ax[i].set_xlabel('Year')
        ax[i].set_ylim(ylim)
        ax[i].set_yticks(np.arange(-0.6, 0.4, 0.2))
        if i % grid_ncol == 0:
            ax[i].set_ylabel(ylabel)

    return fig


def plot_volc_composites(gmt, event_yrs, start_yr=0, before=3, after=10, highpass=False, qs=[0.05, 0.5, 0.95],
                         figsize=[12, 8], title_str=None,
                         clr=sns.xkcd_rgb['pale red'], ylim=[-0.6, 0.3], ylabel='T anom. (K)'):
    ''' plot gmt over the periors of volcanic eruption events

    Args:
        gmt (2d array): the mean temperature timeseries in shape of (time, ensemble)
    '''
    # Superposed Epoch Analysis
    Xevents, Xcomp, tcomp = utils.sea(gmt, event_yrs, start_yr=start_yr, before=before, after=after, highpass=highpass)
    ndim = len(np.shape(Xcomp))
    if ndim > 1:
        Xevents_qs = mquantiles(Xcomp, qs, axis=-1)

    sns.set(style="darkgrid", font_scale=2)
    fig, ax = plt.subplots(figsize=figsize)

    if title_str:
        ax.set_title(title_str)

    if ndim == 1:
        ax.plot(tcomp, Xcomp, '-', lw=3, color=clr)
    else:
        ax.plot(tcomp, Xevents_qs[:, 1], '-', lw=3, color=clr)
        ax.fill_between(tcomp, Xevents_qs[:, 0], Xevents_qs[:, -1], alpha=0.3, color=clr)

    ax.axvline(x=0, ls=':', color='grey')
    ax.axhline(y=0, ls=':', color='grey')
    ax.set_xlabel('Year')
    ax.set_ylim(ylim)
    ax.set_yticks(np.arange(-0.6, 0.4, 0.2))
    ax.set_ylabel(ylabel)

    return fig

def plot_volc_ranking_ana(year_volc, anom_volc, anom_nonvolc_draws, xlim=None,
                  figsize=[5, 5], xlabel=None, ylabel=None,
                  title=None, clr_title='k', show_ratio_in_title=True, style='ridge',
                  clr_volc_signif=sns.xkcd_rgb['pale red'], clr_volc=sns.xkcd_rgb['black'],
                  clr_nonvolc=sns.xkcd_rgb['grey'], clr_nonvolc_light=sns.xkcd_rgb['light grey'],
                  title_fs=15, dist_height=0.05, yticks=None, xticks=None,
                  lgd_style=None, lgd_fs=15, volc_ms_large=40, volc_ms_small=20, signif_qs=[0.8, 0.9, 0.95],
                  label_nonvolc_qs='Distribution of randomly selected non-volcanic years', plot_lgd=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if len(np.shape(anom_volc)) == 1:
        anom_volc = np.expand_dims(anom_volc, axis=0)
        single_curve = True

    if len(np.shape(anom_nonvolc_draws)) == 1:
        anom_nonvolc_draws = np.expand_dims(anom_nonvolc_draws, axis=0)

    n_member, n_volc = np.shape(anom_volc)
    cdf_levels_volc = np.linspace(1/n_volc, 1, n_volc) - 1/n_volc/2
    sorted_volc = np.array([sorted(anom_volc[i]) for i in range(n_member)])

    if single_curve:
        sorted_volc_idx = np.argsort(anom_volc[0])
        for i, level in enumerate(cdf_levels_volc):
            ax.text(
                sorted_volc[0, i], level-0.03, year_volc[sorted_volc_idx[i]],
                color=clr_volc, zorder=100, fontsize=10,  verticalalignment='center', horizontalalignment='center',
            )

    n_draw_member, _ = np.shape(anom_nonvolc_draws)
    sorted_nonvolc_draws = np.array([sorted(anom_nonvolc_draws[i]) for i in range(n_draw_member)])

    nonvolc_draws_qs = np.percentile(sorted_nonvolc_draws, signif_qs, axis=0)
    violin_plot = ax.violinplot(
        sorted_nonvolc_draws, positions=cdf_levels_volc,
        vert=False, widths=dist_height*2,
        showextrema=False,
        # showmedians=True,
        # quantiles=quantiles,
    )

    for pc in violin_plot['bodies']:
        pc.set_edgecolor(clr_nonvolc)
        pc.set_facecolor(clr_nonvolc_light)
        pc.set_alpha(1)
        if style == 'ridge':
            m = np.mean(pc.get_paths()[0].vertices[:, 1])
            # clip the lower half of the violin to make it a ridge
            pc.get_paths()[0].vertices[:, 1] = np.clip(pc.get_paths()[0].vertices[:, 1], m, np.inf)
        elif style == 'violin':
            pass
        else:
            raise ValueError('Wong `style` value. Should be `ridge` or `violin`.')

    # for partname in ['cquantiles']:
    #     vp = violin_plot[partname]
    #     vp.set_edgecolor(clr_nonvolc)

    signif_1st = True
    insignif_1st = True
    nsig_dict = {}
    clr_dict = {}
    for i in range(np.size(signif_qs)+1):
        nsig_dict[i] = 0
        clr_dict[i] = clr_volc if i==0 else clr_volc_signif

    signif_markers = {
        0: 'o',
        1: 'v',
        2: '^',
        3: 'd',
    }
    for i, cdf_level in enumerate(cdf_levels_volc):
        xs = sorted_volc[:, i]
        # kde = sorted_nonvolc_draws[:, i]
        # cdf = cumtrapz(kde, x=np.arange(0, 100), initial=0)
        nonvolc_draws_median = mquantiles(sorted_nonvolc_draws[:, i], [0.5])[0]
        ax.vlines(nonvolc_draws_median, cdf_level, cdf_level+dist_height, linestyle='--', color=clr_nonvolc, lw=2)

        nonvolc_draws_qs = mquantiles(sorted_nonvolc_draws[:, i], signif_qs)
        for nonvolc_draws_q in nonvolc_draws_qs:
            ax.vlines(nonvolc_draws_q, cdf_level, cdf_level+dist_height, linestyle='--', color=clr_nonvolc, lw=2)

        for x in xs:
            found_loc = False
            for j, q in enumerate(nonvolc_draws_qs[::-1]):
                if x > q and not found_loc:
                    level = np.size(signif_qs)-j
                    found_loc = True
                    volc_ms = volc_ms_large
                    break
                
            if not found_loc:
                volc_ms = volc_ms_small
                level = 0

            # print(x, nonvolc_draws_qs, level)
            nsig_dict[level] += 1
            ax.scatter(x, cdf_level, color=clr_dict[level], marker=signif_markers[level], zorder=102, s=volc_ms)

    ax.set_ylim([0, 1.05])
    if yticks is None:
        ax.set_yticks(np.linspace(0, 1, 6))
    else:
        ax.set_yticks(yticks)

    if xticks is not None:
        ax.set_xticks(xticks)

    if ylabel is not None:    
        ax.set_ylabel(ylabel)

    if xlabel is not None:    
        ax.set_xlabel(xlabel)

    if title is not None:    
        if show_ratio_in_title:
            nsig_dict_cum = {}
            for i in range(np.size(signif_qs)+1):
                nsig_dict_cum[i] = 0
                for j in range(i, np.size(signif_qs)+1):
                    nsig_dict_cum[i] += nsig_dict[j]

            nsig_str_list = []
            for i, q in enumerate(signif_qs):
                nsig_str_list.append(str(nsig_dict_cum[i+1]))

            nsig_str = ','.join(s for s in nsig_str_list)

            ratio_str = f'[{nsig_str}]/{n_volc*n_member}'
            title = f'{title} (Signif. ratio: {ratio_str})'

        ax.set_title(title, color=clr_title, fontsize=title_fs)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(True, which='major', zorder=-1)

    if plot_lgd:
        lgd_kwargs = {'loc': 'lower left', 'bbox_to_anchor': (0, -0.5), 'fontsize': lgd_fs, 'ncol': 1}
        lgd_style = {} if lgd_style is None else lgd_style.copy()
        lgd_kwargs.update(lgd_style)

        legend_elements = [
            Line2D([], [], marker='o', markersize=8, color=clr_volc,
            label=f'Volcanic events (< {signif_qs[0]*100:g}% randomly selected non-volcanic years)', linestyle='None'),
        ]

        for i, q in enumerate(signif_qs):
            if i < np.size(signif_qs)-1:
                legend_elements.append(
                    Line2D([], [], marker=signif_markers[i+1], markersize=8, color=clr_volc_signif,
                    label=f'Volcanic events (between {signif_qs[i]*100:g}-{signif_qs[i+1]*100:g}% randomly selected non-volcanic years)', linestyle='None'),
                )
            else:
                legend_elements.append(
                    Line2D([], [], marker=signif_markers[i+1], markersize=8, color=clr_volc_signif,
                    label=f'Volcanic events (>{signif_qs[i]*100:g}% randomly selected non-volcanic years)', linestyle='None'),
                )

        signif_str_list = [f'{q*100:g}%' for q in signif_qs]
        signif_str = ','.join(signif_str_list)

        legend_elements.append(
            Patch(facecolor=clr_nonvolc_light, edgecolor=clr_nonvolc, label=f'{label_nonvolc_qs} (dashed bars: 50%,{signif_str})')
        )

        ax.legend(handles=legend_elements, **lgd_kwargs)

    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_volc_ranking(year_volc, anom_volc, anom_nonvolc, anom_nonvolc_draws, xlim=None,
                  qs=[0.05, 0.95], figsize=[5, 5], xlabel=None, ylabel='CDF',
                  title=None, clr_title='k', show_ratio_in_title=True,
                  clr_volc_signif=sns.xkcd_rgb['pale red'], clr_volc=sns.xkcd_rgb['black'],
                  clr_nonvolc=sns.xkcd_rgb['light grey'], clr_nonvolc_qs=sns.xkcd_rgb['grey'],
                  fs=15, ms=100, plot_qs=True, yticks=None,
                  label_volc_insignif='Volcanic events (insignificant)',
                  label_volc_signif='Volcanic events (significant)',
                  label_nonvolc='Non-volcanic years',
                  label_nonvolc_qs='Randomly selected\nnon-volcanic years', plot_lgd=True, ax=None, lgd_style=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_member, n_volc = np.shape(anom_volc)
    cdf_levels_volc = np.linspace(1/n_volc, 1, n_volc)

    _, n_nonvolc = np.shape(anom_nonvolc)
    cdf_levels_nonvolc = np.linspace(1/n_nonvolc, 1, n_nonvolc)

    sorted_volc = np.array([sorted(anom_volc[i]) for i in range(n_member)])
    sorted_nonvolc = np.array([sorted(anom_nonvolc[i]) for i in range(n_member)])

    n_draw_member, _ = np.shape(anom_nonvolc_draws)
    sorted_nonvolc_draws = np.array([sorted(anom_nonvolc_draws[i]) for i in range(n_draw_member)])

    draws_qs = []
    for i, cdf_level in enumerate(cdf_levels_volc):
        value_nonvolc_draws = sorted_nonvolc_draws[:, i]
        value_nonvolc_draws_qs = mquantiles(value_nonvolc_draws, qs)
        label_nonvolc_qs = f'{label_nonvolc_qs} ({qs[0]*100:g}%-{qs[-1]*100:g}%)'
        lb = label_nonvolc_qs if i == 0 else None
        draws_qs.append(value_nonvolc_draws_qs)
        ax.plot(value_nonvolc_draws_qs, [cdf_level, cdf_level], color=clr_nonvolc_qs, marker='.', zorder=98, label=lb)

    sorted_volc_median = np.median(sorted_volc, axis=0)
    sorted_nonvolc_median = np.median(sorted_nonvolc, axis=0)

    signif_1st = True
    insignif_1st = True
    nsig = 0
    for i, cdf_level in enumerate(cdf_levels_volc):
        volc_median = sorted_volc_median[i]
        if volc_median > draws_qs[i][-1]:
            nsig += 1
            tmp_clr = clr_volc_signif
            lb = label_volc_signif if signif_1st else None
            signif_1st = False
        else:
            tmp_clr = clr_volc
            lb = label_volc_insignif if insignif_1st else None
            insignif_1st = False

        ax.scatter(volc_median, cdf_level, color=tmp_clr, marker='o', zorder=100, label=lb)

    ax.scatter(sorted_nonvolc_median, cdf_levels_nonvolc, color=clr_nonvolc, marker='o', label=label_nonvolc)

    if plot_qs:
        for i, cdf_level in enumerate(cdf_levels_volc):
            volc_median = sorted_volc_median[i]
            if volc_median > draws_qs[i][-1]:
                tmp_clr = clr_volc_signif
            else:
                tmp_clr = clr_volc

            value_volc = sorted_volc[:, i]
            value_volc_qs = mquantiles(value_volc, qs)
            ax.plot(value_volc_qs, [cdf_level, cdf_level], color=tmp_clr, marker='.', zorder=99)

        for i, cdf_level in enumerate(cdf_levels_nonvolc):
            value_nonvolc = sorted_nonvolc[:, i]
            value_nonvolc_qs = mquantiles(value_nonvolc, qs)
            ax.plot(value_nonvolc_qs, [cdf_level, cdf_level], color=clr_nonvolc, marker='.')


    ax.set_ylim([0, 1.02])
    if yticks is None:
        ax.set_yticks(np.linspace(0, 1, 6))
    else:
        ax.set_yticks(yticks)

    if plot_lgd:
        handles, labels = ax.get_legend_handles_labels()
        n_handles = len(handles)
        order = list(range(n_handles))
        order.append(order.pop(order.index(0)))  # put the 1st to the end

        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]

        lgd_kwargs = {'loc': 'lower right', 'bbox_to_anchor': (2, 0), 'fontsize': fs}
        lgd_style = {} if lgd_style is None else lgd_style.copy()
        lgd_kwargs.update(lgd_style)

        new_handler = HandlerLine2D(numpoints=2)
        ax.legend(handles, labels, **lgd_kwargs, handler_map={Line2D: new_handler})

    if ylabel is not None:    
        ax.set_ylabel(ylabel)

    if xlabel is not None:    
        ax.set_xlabel(xlabel)

    if title is not None:    
        if show_ratio_in_title:
            nevents = np.size(year_volc)
            ratio_str = f'{nsig}/{nevents}'
            title = f'{title} (Signif. ratio: {ratio_str})'

        ax.set_title(title, color=clr_title, fontsize=fs)

    if xlim is not None:
        ax.set_xlim(xlim)


    if 'fig' in locals():
        return fig, ax
    else:
        return ax

def plot_volc_cdf(year_volc, anom_volc, anom_nonvolc, anom_nonvolc_draws, value_range,
                  nbin=2000, qs=[0.05, 0.95], figsize=[5, 5], xlabel=None, ylabel='Cumulative distribution function',
                  lw_nonvolc_qs=2, lw_volc_nonvolc=2, xlim=None, title=None, show_ratio_in_title=True,
                  clr_volc_signif=sns.xkcd_rgb['pale red'], clr_volc=sns.xkcd_rgb['black'],
                  clr_nonvolc=sns.xkcd_rgb['grey'], clr_nonvolc_qs=sns.xkcd_rgb['light grey'],
                  fs=15, ms=100, yr_base=2001,
                  label_volc='Volcanic years', label_nonvolc='Non-volcanic years',
                  label_nonvolc_qs='Randomly selected\nnon-volcanic years', plot_lgd=True, ax=None, lgd_style=None):

        kws = {'cumulative': True, 'density': True, 'histtype': 'step', 'range': value_range, 'bins': nbin, 'lw': lw_volc_nonvolc}
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        n, bins, patches = {}, {}, {}
        n['volc'], bins['volc'], patches['volc'] = ax.hist(anom_volc, label=label_volc, color=clr_volc, **kws, zorder=99)
        n['nonvolc'], bins['nonvolc'], patches['nonvolc'] = ax.hist(anom_nonvolc, label=label_nonvolc, color=clr_nonvolc, zorder=98, **kws)

        cdf_draw = []
        for anom_draw in anom_nonvolc_draws:
            cdf_anom_draw = cumfreq(anom_draw, numbins=nbin, defaultreallimits=value_range)
            cdf = cdf_anom_draw.cumcount / np.shape(anom_nonvolc_draws)[-1]
            cdf_draw.append(cdf)

        cdf_qs = utils.calc_cdf_qs(cdf_draw, qs)
        cdf_lb = utils.recover_cdf_from_locs(cdf_qs[qs[0]], nbin)
        cdf_ub = utils.recover_cdf_from_locs(cdf_qs[qs[-1]], nbin)

        x_values = np.linspace(value_range[0], value_range[1], nbin)
        ub_loc_dict = {}
        for k, v in cdf_qs[qs[-1]].items():
            ub_loc_dict[f'{k:.2f}'] = x_values[int(v)]

        for k in n.keys():
            patches[k][0].set_xy(patches[k][0].get_xy()[:-1])

        nsig = 0
        for i, yr in enumerate(year_volc):
            for j, b in enumerate(bins['volc']):
                if anom_volc[i] >= b and anom_volc[i] < bins['volc'][j+1]:
                    n_tmp = n['volc'][j]

            loc = f'{n_tmp:.2f}'
            if anom_volc[i] < ub_loc_dict[loc]:
                signif = False
            else:
                signif = True
                nsig += 1

            clr = clr_volc_signif if signif else clr_volc
            ax.scatter(anom_volc[i], n_tmp, marker='^', color=clr, zorder=100, s=ms)
            ax.text(anom_volc[i]-0.2, n_tmp+0.02, yr%yr_base, color=clr, zorder=100, fontsize=fs)

        ax.scatter(None, None, marker='^', color=clr_volc, label='Insignificant events')
        ax.scatter(None, None, marker='^', color=clr_volc_signif, label='Significant events')

        label_nonvolc_qs = f'{label_nonvolc_qs} ({qs[0]*100:g}%-{qs[-1]*100:g}%)'
        ax.fill_between(np.linspace(value_range[0], value_range[-1], nbin), cdf_lb, cdf_ub, color=clr_nonvolc_qs, label=label_nonvolc_qs, lw=lw_nonvolc_qs)

        ax.set_ylim(0, 1.05)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(-1, 1)

        ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if plot_lgd:
            handles, labels = ax.get_legend_handles_labels()
            order = [0, 1, 4, 2, 3]
            handles = [handles[idx] for idx in order]
            labels = [labels[idx] for idx in order]

            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles[0:2]]
            handles[0:2] = new_handles

            lgd_kwargs = {'loc': 'lower right', 'bbox_to_anchor': (2., 0), 'fontsize': 15}
            lgd_style = {} if lgd_style is None else lgd_style.copy()
            lgd_kwargs.update(lgd_style)

            ax.legend(handles, labels, **lgd_kwargs)

        if title is not None:
            ax.set_title(f'{title}', y=1.05)

        if show_ratio_in_title:
            nevents = np.size(year_volc)
            ratio_str = f'{nsig}/{nevents}'
            ax.text(0.02, 0.9, f'Signif. ratio: {ratio_str}', transform=ax.transAxes)

        if 'fig' in locals():
            return fig, ax
        else:
            return ax


def plot_volc_timeseries(timeseries_dict, event_yrs, before=3, after=10, main_alpha=1, event_alpha=1,
                         clr_dict=None, ls_dict=None, lw_dict=None, xlabel=None, ylabel=None, figsize=[20, 12],
                         xlim=None, ylim=None, event_ylim=None, ncol=4, lgd_ncol=1, lgd_loc=(0, 1)):
    ''' Plot timeseires around volcanic events

    Args
    ----
    timeseries_dict : dict
        the nested dictionary with key as labels for the timeseries and value as timeseries dictionaries,
        each timeseries dictionary is with keys as 'time' and 'value'
    event_yrs : list
        the list of volcanic events
    before : int
        the years before volcanic events for plotting
    after : int
        the years after volcanic events for plotting
    clr_dict : dict
        the dictionary that assigns colors for each timeseries dictionary
    ls_dict : dict
        the dictionary that assigns linestyles for each timeseries dictionary
    lw_dict : dict
        the dictionary that assigns linewidths for each timeseries dictionary

    '''
    sns.set(style="ticks", font_scale=1.5)
    fig = plt.figure(figsize=figsize)

    nevents = np.size(event_yrs)
    nrow = int(1 + np.ceil(nevents/ncol))
    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=0.2, hspace=0.5)

    ax = {}
    ax['main'] = plt.subplot(gs[0, :])
    for ts_name, ts_dict in timeseries_dict.items():
        ts_time = ts_dict['time']
        ts_value = ts_dict['value']

        plot_kwargs = {}
        if clr_dict is not None:
            plot_kwargs['color'] = clr_dict[ts_name]
        if ls_dict is not None:
            plot_kwargs['linestyle'] = ls_dict[ts_name]
        if lw_dict is not None:
            plot_kwargs['linewidth'] = lw_dict[ts_name]

        ax['main'].plot(ts_time, ts_value, label=ts_name, alpha=main_alpha, **plot_kwargs)

    if xlabel is not None:
        ax['main'].set_xlabel(xlabel)

    if ylabel is not None:
        ax['main'].set_ylabel(ylabel)

    if xlim is not None:
        ax['main'].set_xlim(xlim)

    if ylim is not None:
        ax['main'].set_ylim(ylim)

    ax['main'].legend(frameon=False, ncol=lgd_ncol, bbox_to_anchor=lgd_loc, loc='upper left')
    ax['main'].spines['right'].set_visible(False)
    ax['main'].spines['top'].set_visible(False)

    i_row = 1
    for idx, event in enumerate(event_yrs):
        ax[event] = plt.subplot(gs[i_row+idx//ncol, idx%ncol])

        x_start = event-before
        x_end = event+after

        if ylabel is not None and idx%ncol == 0:
            ax[event].set_ylabel(ylabel)

        if event_ylim is None:
            ax[event].set_ylim(ylim)
        else:
            ax[event].set_ylim(event_ylim)

        ax[event].set_xlabel('Years relative to event year')

        for ts_name, ts_dict in timeseries_dict.items():
            ts_time = ts_dict['time']
            ts_value = ts_dict['value']

            plot_kwargs = {}
            if clr_dict is not None:
                plot_kwargs['color'] = clr_dict[ts_name]
            if ls_dict is not None:
                plot_kwargs['linestyle'] = ls_dict[ts_name]
            if lw_dict is not None:
                plot_kwargs['linewidth'] = lw_dict[ts_name]

            if event in list(ts_time):
                i_start = list(ts_time).index(x_start)
                i_end = list(ts_time).index(x_end)
                ax[event].plot(ts_time[i_start:i_end+1], ts_value[i_start:i_end+1], alpha=event_alpha, **plot_kwargs)
            else:
                empty_time = np.arange(x_start, x_end+1)
                empty_value = np.empty(np.size(empty_time))
                empty_value[:] = np.nan
                ax[event].plot(empty_time, empty_value)

            ax_ylim = ax[event].get_ylim()
            ax[event].axvline(x=event, color=sns.xkcd_rgb['grey'], ls='--')
            ax[event].axhline(y=0, color=sns.xkcd_rgb['grey'], ls='--')
            ax[event].text(event, 1.05*ax_ylim[-1], event, horizontalalignment='center', color=sns.xkcd_rgb['grey'])
            
            ax[event].set_xlim([x_start, x_end])
            if before % 2 == 0:
                xticklabels = np.arange(-before, after+1, 2)
                xticks = np.arange(x_start, x_end+1, 2)
            else:
                xticklabels = np.arange(-before+1, after+1, 2)
                xticks = np.arange(x_start+1, x_end+1, 2)

            ax[event].set_xticks(xticks)
            ax[event].set_xticklabels(xticklabels)

        ax[event].spines['right'].set_visible(False)
        ax[event].spines['top'].set_visible(False)

    return fig, ax

def plot_sea_ensemble(res, figsize=[6, 6],
                      clr_volc=sns.xkcd_rgb['pale red'],
                      ylim=None, xlim=None, plot_lgd=False, lgd_kws=None,
                      signif_alpha=0.3, signif_color='k', signif_text_loc_fix=(0.1, -0.01), signif_fontsize=15,
                      xlabel='Years relative to event year', ylabel='T anom. (K)',
                      xticks=None, yticks=None, title=None, plot_signif=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    year_window = res['year_window']
    volc_composite_qs = res['volc_composite_qs']
    nonvolc_draw_composite_qs = res['nonvolc_draw_composite_qs']
    qs = res['qs']
    qs_signif = res['qs_signif']
    nqs = np.size(qs)
    ax.plot(year_window, volc_composite_qs[int((nqs-1)/2)], '-o', color=clr_volc, label='median')
    ax.fill_between(year_window, volc_composite_qs[0], volc_composite_qs[-1], facecolor=clr_volc, alpha=signif_alpha,
                    label=f'{qs[0]*100:g}%-{qs[-1]*100:g}%')

    for i, qs_v in enumerate(res['qs_signif']):
        ax.plot(year_window, nonvolc_draw_composite_qs[i], '--', color=signif_color, alpha=signif_alpha)
        ax.text(year_window[-1]+signif_text_loc_fix[0], nonvolc_draw_composite_qs[i][-1]+signif_text_loc_fix[-1],
                f'{qs_v*100:g}%', color=signif_color, alpha=signif_alpha, fontsize=signif_fontsize)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.axvline(x=0, ls=':', color='grey')
    ax.axhline(y=0, ls=':', color='grey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if title is not None:
        ax.set_title(title)

    if plot_lgd:
        lgd_kws = {} if lgd_kws is None else lgd_kws.copy()
        lgd_style = {'frameon': False, 'loc': 'upper left'}
        lgd_style.update(lgd_kws)
        ax.legend(**lgd_style)

    if 'fig' in locals():
        return fig, ax
    else:
        return ax


def plot_sea_res(res, style='ticks', font_scale=2, figsize=[6, 6],
                 ls='-o', lw=3, color='k', label=None, label_shade=None, alpha=1, shade_alpha=0.3,
                 ylim=None, xlim=None, plot_mode='composite_qs', lgd_individual_yrs=False,
                 signif_alpha=0.5, signif_color=sns.xkcd_rgb['grey'], signif_text_loc_fix=(0.1, -0.01),
                 signif_fontsize=10, signif_lw=1,
                 xlabel='Years relative to event year', ylabel='T anom. (K)', plot_lgd=False,
                 xticks=None, yticks=None, title=None, plot_signif=True, ax=None):
    ''' Plot SEA results
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if plot_mode in res.keys():
        if plot_mode == 'composite_qs':
            ax.plot(res['composite_yr'], res['composite_qs'][1], ls, color=color, label=label, lw=lw, alpha=alpha)
            ax.fill_between(res['composite_yr'], res['composite_qs'][0], res['composite_qs'][-1], facecolor=color, alpha=shade_alpha, label=label_shade)
        elif plot_mode == 'composite':
            ax.plot(res['composite_yr'], res['composite'], ls, color=color, label=label, lw=lw, alpha=alpha)
        elif plot_mode == 'composite_norm':
            ax.plot(res['composite_yr'], res['composite_qs'][1], ls, color=color, label=label, lw=lw, alpha=1)
            for i, individual_curve in enumerate(res['composite_norm'][0, :, :, 0]):
                if lgd_individual_yrs:
                    lb = res['events'][i]
                    clr = None
                else:
                    lb = 'individual events' if i==0 else None
                    clr = color

                ax.plot(res['composite_yr'], individual_curve, '--', label=lb, lw=1, alpha=alpha, color=clr)
    else:
        raise KeyError('Wrong plot_mode!')

    if 'qs_signif' not in res.keys():
        plot_signif = False

    if plot_signif:
        for i, qs_v in enumerate(res['qs_signif']):
            ax.plot(res['composite_yr'], res['composite_qs_signif'][i], '-.', color=signif_color, alpha=signif_alpha, lw=signif_lw)
            ax.text(res['composite_yr'][-1]+signif_text_loc_fix[0], res['composite_qs_signif'][i][-1]+signif_text_loc_fix[-1],
                    f'{qs_v*100:g}%', color=signif_color, alpha=signif_alpha, fontsize=signif_fontsize)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.axvline(x=0, ls=':', color='grey')
    ax.axhline(y=0, ls=':', color='grey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if title is not None:
        ax.set_title(title)

    if plot_lgd:
        ax.legend(frameon=False, loc='upper left')

    if 'fig' in locals():
        return fig, ax
    else:
        return ax


def plot_sea_field_map(field_var, field_signif_lb, field_signif_ub, lat, lon,
                       levels=50, add_cyclic_point=True,
                       title=None, title_size=20, title_weight='normal', figsize=[10, 8],
                       projection='Robinson', transform=ccrs.PlateCarree(),
                       proj_args={}, latlon_range=None,
                       land_alpha=1, ocean_alpha=1,
                       land_color=sns.xkcd_rgb['silver'], ocean_color=sns.xkcd_rgb['silver'],
                       land_zorder=None, ocean_zorder=None, hatch_lb='..', hatch_ub='///',
                       clim=None, cmap='RdBu_r', extend='both', add_gridlines=False,
                       cbar_labels=None, cbar_pad=0.05, cbar_orientation='vertical', cbar_aspect=10,
                       cbar_fraction=0.15, cbar_shrink=0.5, cbar_title=None, font_scale=1.5):

    if add_cyclic_point:
        field_var_c, lon_c = cutil.add_cyclic_point(field_var, lon)
        field_signif_lb_c, lon_c = cutil.add_cyclic_point(field_signif_lb, lon)
        field_signif_ub_c, lon_c = cutil.add_cyclic_point(field_signif_ub, lon)
        lat_c = lat
    else:
        field_var_c, lat_c, lon_c = field_var, lat, lon
        field_signif_lb_c, lat_c, lon_c = field_signif_lb, lat, lon
        field_signif_ub_c, lat_c, lon_c = field_signif_ub, lat, lon

    sns.set(style='ticks', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    projection_dict = {
        'Robinson': ccrs.Robinson,
        'Orthographic': ccrs.Orthographic,
        'NorthPolarStereo': ccrs.NorthPolarStereo,
        'SouthPolarStereo': ccrs.SouthPolarStereo,
        'PlateCarree': ccrs.PlateCarree,
        'AlbersEqualArea': ccrs.AlbersEqualArea,
        'AzimuthalEquidistant': ccrs.AzimuthalEquidistant,
        'EquidistantConic': ccrs.EquidistantConic,
        'LambertConformal': ccrs.LambertConformal,
        'LambertCylindrical': ccrs.LambertCylindrical,
        'Mercator': ccrs.Mercator,
        'Miller': ccrs.Miller,
        'Mollweide': ccrs.Mollweide,
        'Sinusoidal': ccrs.Sinusoidal,
        'Stereographic': ccrs.Stereographic,
        'TransverseMercator': ccrs.TransverseMercator,
        'UTM': ccrs.UTM,
        'InterruptedGoodeHomolosine': ccrs.InterruptedGoodeHomolosine,
        'RotatedPole': ccrs.RotatedPole,
        'OSGB': ccrs.OSGB,
        'EuroPP': ccrs.EuroPP,
        'Geostationary': ccrs.Geostationary,
        'NearsidePerspective': ccrs.NearsidePerspective,
        'EckertI': ccrs.EckertI,
        'EckertII': ccrs.EckertII,
        'EckertIII': ccrs.EckertIII,
        'EckertIV': ccrs.EckertIV,
        'EckertV': ccrs.EckertV,
        'EckertVI': ccrs.EckertVI,
        'EqualEarth': ccrs.EqualEarth,
        'Gnomonic': ccrs.Gnomonic,
        'LambertAzimuthalEqualArea': ccrs.LambertAzimuthalEqualArea,
        'OSNI': ccrs.OSNI,
    }

    projection = projection_dict[projection](**proj_args)
    ax = plt.subplot(projection=projection)

    if title:
        plt.title(title, fontsize=title_size, fontweight=title_weight)

    if latlon_range:
        lon_min, lon_max, lat_min, lat_max = latlon_range
        ax.set_extent(latlon_range, crs=transform)
        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        lon_ticks = np.array(lon_ticks)
        lat_ticks = np.array(lat_ticks)
        mask_lon = (lon_ticks >= lon_min) & (lon_ticks <= lon_max)
        mask_lat = (lat_ticks >= lat_min) & (lat_ticks <= lat_max)
        ax.set_xticks(lon_ticks[mask_lon], crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks[mask_lat], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor='k', alpha=land_alpha, zorder=land_zorder)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor='k', alpha=ocean_alpha, zorder=ocean_zorder)
    ax.coastlines()

    if add_gridlines:
        ax.gridlines(edgecolor='gray', linestyle=':', crs=transform)

    cmap = plt.get_cmap(cmap)

    im = ax.contourf(lon_c, lat_c, field_var_c, levels, transform=transform, cmap=cmap, extend=extend)

    field_tmp = np.copy(field_var_c)
    field_tmp[field_tmp>0] = np.nan
    diff = field_tmp-field_signif_lb_c
    ax.contourf(lon_c, lat_c, diff, [0, 9999], transform=transform, hatches=[hatch_lb], colors='none')

    field_tmp = np.copy(field_var_c)
    field_tmp[field_tmp<0] = np.nan
    diff = field_tmp-field_signif_ub_c
    ax.contourf(lon_c, lat_c, diff, [-9999, 0], transform=transform, hatches=[hatch_ub], colors='none')

    if clim:
        im.set_clim(clim)

    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
                        fraction=cbar_fraction, shrink=cbar_shrink)

    if cbar_labels is not None:
        cbar.set_ticks(cbar_labels)

    if cbar_title:
        cbar.ax.set_title(cbar_title)

    return fig, ax


def plot_vsl_dashboard(pid, vsl_res, vsl_params,
                       tas_model, pr_model,
                       lat_model, lon_model, time_model, elev_model=None,
                       tas_corrected=None, pr_corrected=None, xlim=[850, 2005],
                       ls_pseudoproxy='-', ls_proxy='-',
                       calc_corr=True, text_x_fix=0, corr_loc=[1.01, 0.1],
                       fix_T=False, T1_quantile=0.7, T2_quantile=0.7,
                       lat_ind_dict=None, lon_ind_dict=None,
                       beta_params=np.array([
                            [9, 5, 0, 9],
                            [3.5, 3.5, 10, 24],
                            [1.5, 2.8, 0, 0.1],
                            [1.5, 2.5, 0.1, 0.5],
                        ]), seed=0):
    ''' Plot the dashboard to check VSL results

    Args:
        pid (str): the proxy ID
        vsl_res (dict): the detailed result from VSL, including
            - vsl_res[pid]['trw_org']: the original TRW output without normalization
            - vsl_res[pid]['gT']: the growth response related to temperature
            - vsl_res[pid]['gM']: the growth response related to moisture
            - vsl_res[pid]['gE']: the growth response related to latitude
            - vsl_res[pid]['M']: the soil moisutre from the leaky bucket model
        vsl_params (dict): the dict for the parameters for VSL, including
            - vsl_params['pid_obs']:
            - vsl_params['lat_obs']:
            - vsl_params['lon_obs']:
            - vsl_params['elev_obs']:
            - vsl_params['values_obs']:
            - vsl_params['T1']:
            - vsl_params['T2']:
            - vsl_params['M1']:
            - vsl_params['M2']:
        tas_model (3-D array): surface air temperature in (time, lat, lon) [K]
        pr_model (3-D array): precipitation rate in (time, lat, lon) [kg/m2/s]

    Returns:
        fig (figure)
    '''
    #===========================================================
    # preprocessing
    #-----------------------------------------------------------
    import p2k

    idx = list(vsl_params['pid_obs']).index(pid)
    lat_obs = vsl_params['lat_obs'][idx]
    lon_obs = vsl_params['lon_obs'][idx]
    elev_obs = vsl_params['elev_obs'][idx]
    trw_data = vsl_params['values_obs'][idx]
    trw_value = np.array(trw_data.values)
    trw_time = np.array(trw_data.index)

    T1 = vsl_params['T1'][idx]
    T2 = vsl_params['T2'][idx]
    M1 = vsl_params['M1'][idx]
    M2 = vsl_params['M2'][idx]

    T1_dist = vsl_params['params_est'][idx][4]
    T2_dist = vsl_params['params_est'][idx][5]
    M1_dist = vsl_params['params_est'][idx][6]
    M2_dist = vsl_params['params_est'][idx][7]

    gT = vsl_res[pid]['gT']
    gM = vsl_res[pid]['gM']
    gE = vsl_res[pid]['gE']
    M = vsl_res[pid]['M']

    if lat_ind_dict is None:
        lat_ind, lon_ind = utils.find_closest_loc(lat_model, lon_model, lat_obs, lon_obs)
    else:
        lat_ind, lon_ind = lat_ind_dict[pid], lon_ind_dict[pid]

    if tas_corrected is not None:
        tas_sub = tas_corrected[pid]
        pr_sub = pr_corrected[pid]
    else:
        tas_sub = tas_model[:, lat_ind, lon_ind]
        pr_sub = pr_model[:, lat_ind, lon_ind]

    if fix_T:
        tas_qs = mquantiles(tas_sub, T1_quantile)[0]
        if T1 > tas_qs:
            diff = T1 - tas_qs
            T1 -= diff
            T2 -= diff

        tas_qs = mquantiles(tas_sub, T2_quantile)[0]
        if T2 < tas_qs:
            diff = tas_qs - T2
            T1 += diff
            T2 += diff

    tas_ann, year_ann = utils.annualize_var(tas_sub, time_model)
    pr_ann, year_ann = utils.annualize_var(pr_sub, time_model)
    M_ann, year_ann = utils.annualize_var(M, time_model)
    gT_ann, year_ann = utils.annualize_var(gT, time_model)
    gM_ann, year_ann = utils.annualize_var(gM, time_model)

    # pseudo value with bias correction and vairance match
    trw_pseudo = vsl_res[pid]['trw_org']
    trw_pseudo = LMRt.utils.ts_matching(year_ann, trw_pseudo, trw_time, trw_value)['value_target']

    #===========================================================
    # plot
    #-----------------------------------------------------------
    tas_color = sns.xkcd_rgb['pale red']
    pr_color = sns.xkcd_rgb['denim blue']
    M_color = 'gray'
    gM_color = M_color
    gT_color = sns.xkcd_rgb['pale red']
    gE_color = sns.xkcd_rgb['amber']
    trw_color = sns.xkcd_rgb['medium green']

    sns.set(style="ticks", font_scale=1.5)
    fig = plt.figure(figsize=[20, 12])
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=3, hspace=0.5)

    ax_tas = plt.subplot(gs[0, 0:3])
    ax_tas.plot(time_model, tas_sub, '-', color=tas_color, alpha=0.1)
    ax_tas.plot(year_ann, tas_ann, '-', color=tas_color, label=f'mean={np.nanmean(tas_sub):.2f}, max={np.nanmax(tas_sub):.2f}, min={np.nanmin(tas_sub):.2f}, std={np.nanstd(tas_sub):.2f}')
    ax_tas.spines['right'].set_visible(False)
    ax_tas.spines['top'].set_visible(False)
    ax_tas.spines['bottom'].set_visible(False)
    ax_tas.set_ylabel(r'tas. ($^\circ$C)', color=tas_color)
    ax_tas.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_tas.tick_params('y', colors=tas_color)
    ax_tas.spines['left'].set_color(tas_color)
    ax_tas.axhline(y=T1, ls='--', color=tas_color)
    ax_tas.axhline(y=T2, ls='--', color=tas_color)

    ax_tas.text(year_ann[-1]+text_x_fix, T1, f'T1={T1:.2f}', color=tas_color, fontsize=15)
    ax_tas.text(year_ann[-1]+text_x_fix, T2, f'T2={T2:.2f}', color=tas_color, fontsize=15)
    ax_tas.set_xlim(xlim)
    ax_tas.legend(loc='lower left', frameon=False, bbox_to_anchor=(0, -0.2))

    #-----------------------------------------------------------
    ax_pr = plt.subplot(gs[1, 0:3], sharex=ax_tas)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['bottom'].set_visible(False)
    ax_pr.plot(time_model, pr_sub, '-', color=pr_color, alpha=0.1)
    ax_pr.plot(year_ann, pr_ann, '-', color=pr_color)
    ax_pr.set_ylabel('acc. pr. (mm)', color=pr_color)
    ax_pr.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_pr.tick_params('y', colors=pr_color)
    ax_pr.spines['left'].set_color(pr_color)

    #-----------------------------------------------------------
    ax_soil = plt.subplot(gs[2, 0:3], sharex=ax_tas)
    ax_soil.plot(time_model, M, '-', color=M_color, alpha=0.1)
    ax_soil.plot(year_ann, M_ann, '-', color=M_color, label=f'mean={np.nanmean(M):.2f}, max={np.nanmax(M):.2f}, min={np.nanmin(M):.2f}, std={np.nanstd(M):.2f}')
    ax_soil.tick_params('y', colors=M_color)
    ax_soil.spines['right'].set_color(M_color)
    ax_soil.spines['right'].set_visible(False)
    ax_soil.spines['top'].set_visible(False)
    ax_soil.spines['bottom'].set_visible(False)
    ax_soil.set_ylabel('soil moisture (v/v)', color=M_color)
    ax_soil.axhline(y=M1, ls='--', color=M_color)
    ax_soil.axhline(y=M2, ls='--', color=M_color)
    ax_soil.text(year_ann[-1]+text_x_fix, M1, f'M1={M1:.2f}', color=M_color, fontsize=15)
    ax_soil.text(year_ann[-1]+text_x_fix, M2, f'M2={M2:.2f}', color=M_color, fontsize=15)
    ax_soil.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 1.2))
    ax_soil.spines['left'].set_color(M_color)
    ax_soil.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    #-----------------------------------------------------------
    ax_map = plt.subplot(gs[0:2, 3:], projection=ccrs.Robinson())
    if elev_model is not None:
        ax_map.set_title(f'{pid}\nTarget: (lat: {lat_obs:.2f}, lon: {lon_obs:.2f}, elev: {elev_obs:.2f})\nFound: (lat: {lat_model[lat_ind]:.2f}, lon: {lon_model[lon_ind]:.2f}, elev: {elev_model:.2f})')
    else:
        ax_map.set_title(f'{pid}\nTarget: (lat: {lat_obs:.2f}, lon: {lon_obs:.2f})\nFound: (lat: {lat_model[lat_ind]:.2f}, lon: {lon_model[lon_ind]:.2f})')

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax_map.gridlines(edgecolor='gray', linestyle=':')
    p = PAGES2k()
    ax_map.scatter(
        lon_obs, lat_obs, marker=p.markers_dict['Tree Rings_WidthPages2'],
        c=p.colors_dict['Tree Rings_WidthPages2'], edgecolor='k', s=50, transform=ccrs.Geodetic()
    )

    #-----------------------------------------------------------
    ax_growth = plt.subplot(gs[3, 0:3], sharex=ax_tas)
    ax_growth.plot(time_model, gT, '-', color=gT_color, alpha=0.1)
    ax_growth.plot(year_ann, gT_ann, '-', color=gT_color, label='gT')
    ax_growth.plot(time_model, gM, '-', color=gM_color, alpha=0.1)
    ax_growth.plot(year_ann, gM_ann, '-', color=gM_color, label=f'gM')

    ax_growth.spines['right'].set_visible(False)
    ax_growth.spines['top'].set_visible(False)
    ax_growth.spines['bottom'].set_visible(False)
    ax_growth.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_growth.set_ylabel('growth')
    ax_growth.set_ylim(0, 1)
    ax_growth.legend(fontsize=15, bbox_to_anchor=(1.15, 1), loc='upper right', ncol=1, frameon=False)

    if calc_corr:
        res = utils.compare_ts(year_ann, trw_pseudo, trw_time, trw_value)
        corr = res['corr']

    #-----------------------------------------------------------
    ax_trw = plt.subplot(gs[4, 0:3], sharex=ax_tas)
    ax_trw.plot(year_ann, trw_pseudo, ls_pseudoproxy, color=trw_color, label='pseudoproxy')
    ax_trw.plot(trw_time, trw_value, ls_proxy, color='gray', ms=2, label='proxy')
    ax_trw.spines['right'].set_visible(False)
    ax_trw.spines['top'].set_visible(False)
    ax_trw.set_ylabel('TRW')
    ax_trw.set_xlabel('Year (AD)')
    ax_trw.legend(fontsize=15, bbox_to_anchor=(1.25, 1), loc='upper right', ncol=1, frameon=False)
    ax_trw.set_xlim(xlim)

    if calc_corr:
        ax_trw.text(corr_loc[0], corr_loc[-1], f'corr={corr:.2f}', transform=ax_trw.transAxes, fontsize=15)

    period_ticks=[2, 5, 10, 20, 50, 100, 200, 500, 1000]

    #-----------------------------------------------------------
    T1_a, T1_b, T1_lb, T1_ub = beta_params[0]
    T2_a, T2_b, T2_lb, T2_ub = beta_params[1]
    M1_a, M1_b, M1_lb, M1_ub = beta_params[2]
    M2_a, M2_b, M2_lb, M2_ub = beta_params[3]

    T1_prior = stats.beta.rvs(T1_a, T1_b, loc=T1_lb, scale=T1_ub-T1_lb, size=1001, random_state=seed)
    T2_prior = stats.beta.rvs(T2_a, T2_b, loc=T2_lb, scale=T2_ub-T2_lb, size=1001, random_state=seed)
    M1_prior = stats.beta.rvs(M1_a, M1_b, loc=M1_lb, scale=M1_ub-M1_lb, size=1001, random_state=seed)
    M2_prior = stats.beta.rvs(M2_a, M2_b, loc=M2_lb, scale=M2_ub-M2_lb, size=1001, random_state=seed)

    ax_Tdist = plt.subplot(gs[2, 3:])
    sns.distplot(T2_prior, hist=False, color=M_color, kde_kws={'ls': '-'}, label=f'T2 (prior: {np.nanmedian(T2_prior):.2f})')
    sns.distplot(T1_prior, hist=False, color=M_color, kde_kws={'ls': '--'}, label=f'T1 (prior: {np.nanmedian(T1_prior):.2f})')
    ax_Tdist.axvline(x=np.nanmedian(T1_prior), ls='--', ymax=0.1, color=M_color)
    ax_Tdist.axvline(x=np.nanmedian(T2_prior), ls='-', ymax=0.1, color=M_color)

    sns.distplot(T2_dist, hist=False, color=tas_color, kde_kws={'ls': '-'}, label=f'T2 (posterior: {np.nanmedian(T2_dist):.2f})')
    sns.distplot(T1_dist, hist=False, color=tas_color, kde_kws={'ls': '--'}, label=f'T1 (posterior: {np.nanmedian(T1_dist):.2f})')
    ax_Tdist.axvline(x=np.nanmedian(T1_dist), ls='--', ymax=0.1, color=tas_color)
    ax_Tdist.axvline(x=np.nanmedian(T2_dist), ls='-', ymax=0.1, color=tas_color)

    ax_Tdist.spines['right'].set_visible(False)
    ax_Tdist.spines['top'].set_visible(False)
    ax_Tdist.set_title('Prior/Posterior of parameters')
    ax_Tdist.set_ylabel('KDE')
    ax_Tdist.set_xlabel(r'tas. ($^\circ$C)')
    ax_Tdist.legend(frameon=False, loc='upper right', fontsize=11, ncol=2)

    #-----------------------------------------------------------
    ax_Mdist = plt.subplot(gs[3, 3:])
    sns.distplot(M2_prior, hist=False, color=M_color, kde_kws={'ls': '-'}, label=f'M2 (prior: {np.nanmedian(M2_prior):.2f})')
    sns.distplot(M1_prior, hist=False, color=M_color, kde_kws={'ls': '--'}, label=f'M1 (prior: {np.nanmedian(M1_prior):.2f})')
    ax_Mdist.axvline(x=np.nanmedian(M1_prior), ls='--', ymax=0.1, color=M_color)
    ax_Mdist.axvline(x=np.nanmedian(M2_prior), ls='-', ymax=0.1, color=M_color)

    sns.distplot(M2_dist, hist=False, color=pr_color, kde_kws={'ls': '-'}, label=f'M2 (posterior: {np.nanmedian(M2_dist):.2f})')
    sns.distplot(M1_dist, hist=False, color=pr_color, kde_kws={'ls': '--'}, label=f'M1 (posterior: {np.nanmedian(M1_dist):.2f})')
    ax_Mdist.axvline(x=np.nanmedian(M1_dist), ls='--', ymax=0.1, color=pr_color)
    ax_Mdist.axvline(x=np.nanmedian(M2_dist), ls='-', ymax=0.1, color=pr_color)

    ax_Mdist.spines['right'].set_visible(False)
    ax_Mdist.spines['top'].set_visible(False)
    ax_Mdist.set_ylabel('KDE')
    ax_Mdist.set_xlabel('soil moisture (v/v)')
    ax_Mdist.legend(frameon=False, loc='upper right', fontsize=11, ncol=2)

    #-----------------------------------------------------------
    ax_spec = plt.subplot(gs[4:, 3:])
    dcon = 0.01
    ntau = 51

    psd_pseudo, freqs_pseudo = p2k.calc_plot_psd(trw_pseudo, year_ann, plot_fig=False, anti_alias=False, dcon=dcon, ntau=ntau)
    psd_proxy, freqs_proxy = p2k.calc_plot_psd(trw_value, trw_time, plot_fig=False, anti_alias=False, dcon=dcon, ntau=ntau)

    lw = 2
    ax_spec.loglog(1/freqs_pseudo, psd_pseudo, lw=lw, color=trw_color, label='pseudoproxy')
    ax_spec.loglog(1/freqs_proxy, psd_proxy, lw=lw, color='gray', label='proxy')
    ax_spec.set_xticks(period_ticks)
    ax_spec.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_spec.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_spec.invert_xaxis()
    ax_spec.set_ylabel('PSD')
    ax_spec.set_xlabel('Period (years)')
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['top'].set_visible(False)
    # ax_spec.set_ylim([1e-1, 1e1])
    ax_spec.set_xlim([200, 2])
    ax_spec.legend(fontsize=15, loc='upper right', ncol=1, frameon=False)

    #===========================================================

    if calc_corr:
        return fig, corr
    else:
        return fig


def plot_vsl_dashboard_p2k(p2k_id, vsl_res, meta_dict, vsl_params, xlim=[850, 2005],
                           psd_dict_path=None,
                           ls_pseudoproxy='-', ls_proxy='-',
                           calc_corr=False, text_x_fix=0, corr_loc=[1.01, 0.1],
                           beta_params=np.array([
                               [9, 5, 0, 9],
                               [3.5, 3.5, 10, 24],
                               [1.5, 2.8, 0, 0.1],
                               [1.5, 2.5, 0.1, 0.5],
                           ]), seed=0):
    ''' Plot the dashboard to check VSL results

    Args:
        p2k_id (str): the proxy ID
        vsl_res (dict): the detailed result from VSL, including
            - vsl_res[p2k_id]['trw_org']: the original TRW output without normalization
            - vsl_res[p2k_id]['gT']: the growth response related to temperature
            - vsl_res[p2k_id]['gM']: the growth response related to moisture
            - vsl_res[p2k_id]['gE']: the growth response related to latitude
            - vsl_res[p2k_id]['M']: the soil moisutre from the leaky bucket model
        vsl_params (dict): the dict for the parameters for VSL, including
        tas_model (3-D array): surface air temperature in (time, lat, lon) [K]
        pr_model (3-D array): precipitation rate in (time, lat, lon) [kg/m2/s]

    Returns:
        fig (figure)
    '''
    #===========================================================
    # preprocessing
    #-----------------------------------------------------------
    import p2k

    T1 = vsl_params[p2k_id]['T1']
    T2 = vsl_params[p2k_id]['T2']
    M1 = vsl_params[p2k_id]['M1']
    M2 = vsl_params[p2k_id]['M2']

    T1_dist = vsl_params[p2k_id]['params_est'][4]
    T2_dist = vsl_params[p2k_id]['params_est'][5]
    M1_dist = vsl_params[p2k_id]['params_est'][6]
    M2_dist = vsl_params[p2k_id]['params_est'][7]

    trw_obs = meta_dict[p2k_id]['trw_obs']
    time_obs = meta_dict[p2k_id]['time_obs']
    lat_obs = meta_dict[p2k_id]['lat_obs']
    lon_obs = meta_dict[p2k_id]['lon_obs']

    tas_model = meta_dict[p2k_id]['tas_model'] - 273.15
    pr_model = meta_dict[p2k_id]['pr_model'] * 3600*24*30
    time_model = meta_dict[p2k_id]['time_model']
    lat_model = meta_dict[p2k_id]['lat_model']
    lon_model = meta_dict[p2k_id]['lon_model']

    gT = vsl_res[p2k_id]['gT']
    gM = vsl_res[p2k_id]['gM']
    gE = vsl_res[p2k_id]['gE']
    M = vsl_res[p2k_id]['M']

    tas_ann, year_ann = utils.annualize_var(tas_model, time_model)
    pr_ann, year_ann = utils.annualize_var(pr_model, time_model)
    M_ann, year_ann = utils.annualize_var(M, time_model)
    gT_ann, year_ann = utils.annualize_var(gT, time_model)
    gM_ann, year_ann = utils.annualize_var(gM, time_model)

    # pseudo value with bias correction and vairance match
    trw_pseudo = vsl_res[p2k_id]['trw_org']
    trw_pseudo = utils.ts_matching(year_ann, trw_pseudo, time_obs, trw_obs)['value_target']

    #===========================================================
    # plot
    #-----------------------------------------------------------
    tas_color = sns.xkcd_rgb['pale red']
    pr_color = sns.xkcd_rgb['denim blue']
    M_color = 'gray'
    gM_color = M_color
    gT_color = sns.xkcd_rgb['pale red']
    gE_color = sns.xkcd_rgb['amber']
    trw_color = sns.xkcd_rgb['medium green']

    sns.set(style="ticks", font_scale=1.5)
    fig = plt.figure(figsize=[20, 12])
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=3, hspace=0.5)

    ax_tas = plt.subplot(gs[0, 0:3])
    ax_tas.plot(time_model, tas_model, '-', color=tas_color, alpha=0.1)
    ax_tas.plot(year_ann, tas_ann, '-', color=tas_color, label=f'mean={np.nanmean(tas_model):.2f}, max={np.nanmax(tas_model):.2f}, min={np.nanmin(tas_model):.2f}, std={np.nanstd(tas_model):.2f}')
    ax_tas.spines['right'].set_visible(False)
    ax_tas.spines['top'].set_visible(False)
    ax_tas.spines['bottom'].set_visible(False)
    ax_tas.set_ylabel(r'tas. ($^\circ$C)', color=tas_color)
    ax_tas.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_tas.tick_params('y', colors=tas_color)
    ax_tas.spines['left'].set_color(tas_color)
    ax_tas.axhline(y=T1, ls='--', color=tas_color)
    ax_tas.axhline(y=T2, ls='--', color=tas_color)

    ax_tas.text(year_ann[-1]+text_x_fix, T1, f'T1={T1:.2f}', color=tas_color, fontsize=15)
    ax_tas.text(year_ann[-1]+text_x_fix, T2, f'T2={T2:.2f}', color=tas_color, fontsize=15)
    ax_tas.set_xlim(xlim)
    ax_tas.legend(loc='lower left', frameon=False, bbox_to_anchor=(0, -0.2))

    #-----------------------------------------------------------
    ax_pr = plt.subplot(gs[1, 0:3], sharex=ax_tas)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['bottom'].set_visible(False)
    ax_pr.plot(time_model, pr_model, '-', color=pr_color, alpha=0.1)
    ax_pr.plot(year_ann, pr_ann, '-', color=pr_color)
    ax_pr.set_ylabel('acc. pr. (mm)', color=pr_color)
    ax_pr.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_pr.tick_params('y', colors=pr_color)
    ax_pr.spines['left'].set_color(pr_color)

    #-----------------------------------------------------------
    ax_soil = plt.subplot(gs[2, 0:3], sharex=ax_tas)
    ax_soil.plot(time_model, M, '-', color=M_color, alpha=0.1)
    ax_soil.plot(year_ann, M_ann, '-', color=M_color, label=f'mean={np.nanmean(M):.2f}, max={np.nanmax(M):.2f}, min={np.nanmin(M):.2f}, std={np.nanstd(M):.2f}')
    ax_soil.tick_params('y', colors=M_color)
    ax_soil.spines['right'].set_color(M_color)
    ax_soil.spines['right'].set_visible(False)
    ax_soil.spines['top'].set_visible(False)
    ax_soil.spines['bottom'].set_visible(False)
    ax_soil.set_ylabel('soil moisture (v/v)', color=M_color)
    ax_soil.axhline(y=M1, ls='--', color=M_color)
    ax_soil.axhline(y=M2, ls='--', color=M_color)
    ax_soil.text(year_ann[-1]+text_x_fix, M1, f'M1={M1:.2f}', color=M_color, fontsize=15)
    ax_soil.text(year_ann[-1]+text_x_fix, M2, f'M2={M2:.2f}', color=M_color, fontsize=15)
    ax_soil.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 1.2))
    ax_soil.spines['left'].set_color(M_color)
    ax_soil.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    #-----------------------------------------------------------
    ax_map = plt.subplot(gs[0:2, 3:], projection=ccrs.Robinson())
    ax_map.set_title(f'{p2k_id}\nTarget: (lat: {lat_obs:.2f}, lon: {lon_obs:.2f})\nFound: (lat: {lat_model:.2f}, lon: {lon_model:.2f})')

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax_map.gridlines(edgecolor='gray', linestyle=':')
    p = PAGES2k()
    ax_map.scatter(
        lon_obs, lat_obs, marker=p.markers_dict['Tree Rings_WidthPages2'],
        c=p.colors_dict['Tree Rings_WidthPages2'], edgecolor='k', s=50, transform=ccrs.Geodetic()
    )

    #-----------------------------------------------------------
    ax_growth = plt.subplot(gs[3, 0:3], sharex=ax_tas)
    ax_growth.plot(time_model, gT, '-', color=gT_color, alpha=0.1)
    ax_growth.plot(year_ann, gT_ann, '-', color=gT_color, label='gT')
    ax_growth.plot(time_model, gM, '-', color=gM_color, alpha=0.1)
    ax_growth.plot(year_ann, gM_ann, '-', color=gM_color, label=f'gM')

    ax_growth.spines['right'].set_visible(False)
    ax_growth.spines['top'].set_visible(False)
    ax_growth.spines['bottom'].set_visible(False)
    ax_growth.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_growth.set_ylabel('growth')
    ax_growth.set_ylim(0, 1)
    ax_growth.legend(fontsize=15, bbox_to_anchor=(1.15, 1), loc='upper right', ncol=1, frameon=False)

    if calc_corr:
        res = utils.compare_ts(year_ann, trw_pseudo, time_obs, trw_obs)
        corr = res['corr']

    #-----------------------------------------------------------
    ax_trw = plt.subplot(gs[4, 0:3], sharex=ax_tas)
    ax_trw.plot(year_ann, trw_pseudo, ls_pseudoproxy, color=trw_color, label='pseudoproxy')
    ax_trw.plot(time_obs, trw_obs, ls_proxy, color='gray', ms=2, label='proxy')
    ax_trw.spines['right'].set_visible(False)
    ax_trw.spines['top'].set_visible(False)
    ax_trw.set_ylabel('TRW')
    ax_trw.set_xlabel('Year (AD)')
    ax_trw.legend(fontsize=15, bbox_to_anchor=(1.25, 1), loc='upper right', ncol=1, frameon=False)
    ax_trw.set_xlim(xlim)

    if calc_corr:
        ax_trw.text(corr_loc[0], corr_loc[-1], f'corr={corr:.2f}', transform=ax_trw.transAxes, fontsize=15)

    period_ticks=[2, 5, 10, 20, 50, 100, 200, 500, 1000]

    #-----------------------------------------------------------
    T1_a, T1_b, T1_lb, T1_ub = beta_params[0]
    T2_a, T2_b, T2_lb, T2_ub = beta_params[1]
    M1_a, M1_b, M1_lb, M1_ub = beta_params[2]
    M2_a, M2_b, M2_lb, M2_ub = beta_params[3]

    T1_prior = stats.beta.rvs(T1_a, T1_b, loc=T1_lb, scale=T1_ub-T1_lb, size=1001, random_state=seed)
    T2_prior = stats.beta.rvs(T2_a, T2_b, loc=T2_lb, scale=T2_ub-T2_lb, size=1001, random_state=seed)
    M1_prior = stats.beta.rvs(M1_a, M1_b, loc=M1_lb, scale=M1_ub-M1_lb, size=1001, random_state=seed)
    M2_prior = stats.beta.rvs(M2_a, M2_b, loc=M2_lb, scale=M2_ub-M2_lb, size=1001, random_state=seed)

    ax_Tdist = plt.subplot(gs[2, 3:])
    sns.distplot(T2_prior, hist=False, color=M_color, kde_kws={'ls': '-'}, label=f'T2 (prior: {np.nanmedian(T2_prior):.2f})')
    sns.distplot(T1_prior, hist=False, color=M_color, kde_kws={'ls': '--'}, label=f'T1 (prior: {np.nanmedian(T1_prior):.2f})')
    ax_Tdist.axvline(x=np.nanmedian(T1_prior), ls='--', ymax=0.1, color=M_color)
    ax_Tdist.axvline(x=np.nanmedian(T2_prior), ls='-', ymax=0.1, color=M_color)

    sns.distplot(T2_dist, hist=False, color=tas_color, kde_kws={'ls': '-'}, label=f'T2 (posterior: {np.nanmedian(T2_dist):.2f})')
    sns.distplot(T1_dist, hist=False, color=tas_color, kde_kws={'ls': '--'}, label=f'T1 (posterior: {np.nanmedian(T1_dist):.2f})')
    ax_Tdist.axvline(x=np.nanmedian(T1_dist), ls='--', ymax=0.1, color=tas_color)
    ax_Tdist.axvline(x=np.nanmedian(T2_dist), ls='-', ymax=0.1, color=tas_color)

    ax_Tdist.spines['right'].set_visible(False)
    ax_Tdist.spines['top'].set_visible(False)
    ax_Tdist.set_title('Prior/Posterior of parameters')
    ax_Tdist.set_ylabel('KDE')
    ax_Tdist.set_xlabel(r'tas. ($^\circ$C)')
    ax_Tdist.legend(frameon=False, loc='upper right', fontsize=11, ncol=2)

    #-----------------------------------------------------------
    ax_Mdist = plt.subplot(gs[3, 3:])
    sns.distplot(M2_prior, hist=False, color=M_color, kde_kws={'ls': '-'}, label=f'M2 (prior: {np.nanmedian(M2_prior):.2f})')
    sns.distplot(M1_prior, hist=False, color=M_color, kde_kws={'ls': '--'}, label=f'M1 (prior: {np.nanmedian(M1_prior):.2f})')
    ax_Mdist.axvline(x=np.nanmedian(M1_prior), ls='--', ymax=0.1, color=M_color)
    ax_Mdist.axvline(x=np.nanmedian(M2_prior), ls='-', ymax=0.1, color=M_color)

    sns.distplot(M2_dist, hist=False, color=pr_color, kde_kws={'ls': '-'}, label=f'M2 (posterior: {np.nanmedian(M2_dist):.2f})')
    sns.distplot(M1_dist, hist=False, color=pr_color, kde_kws={'ls': '--'}, label=f'M1 (posterior: {np.nanmedian(M1_dist):.2f})')
    ax_Mdist.axvline(x=np.nanmedian(M1_dist), ls='--', ymax=0.1, color=pr_color)
    ax_Mdist.axvline(x=np.nanmedian(M2_dist), ls='-', ymax=0.1, color=pr_color)

    ax_Mdist.spines['right'].set_visible(False)
    ax_Mdist.spines['top'].set_visible(False)
    ax_Mdist.set_ylabel('KDE')
    ax_Mdist.set_xlabel('soil moisture (v/v)')
    ax_Mdist.legend(frameon=False, loc='upper right', fontsize=11, ncol=2)

    #-----------------------------------------------------------
    ax_spec = plt.subplot(gs[4:, 3:])

    if psd_dict_path is None:
        dcon = 0.01
        ntau = 51

        psd_pseudo, freqs_pseudo = p2k.calc_plot_psd(trw_pseudo, year_ann, plot_fig=False, anti_alias=False, dcon=dcon, ntau=ntau)
        psd_proxy, freqs_proxy = p2k.calc_plot_psd(trw_obs, time_obs, plot_fig=False, anti_alias=False, dcon=dcon, ntau=ntau)
    else:
        with open(psd_dict_path, 'rb') as f:
            psd_pseudo_dict, freqs_pseudo_dict, psd_proxy_dict, freqs_proxy_dict = pickle.load(f)

        psd_pseudo, freqs_pseudo, psd_proxy, freqs_proxy = psd_pseudo_dict[p2k_id], freqs_pseudo_dict[p2k_id], psd_proxy_dict[p2k_id], freqs_proxy_dict[p2k_id]

    lw = 2
    ax_spec.loglog(1/freqs_pseudo, psd_pseudo, lw=lw, color=trw_color, label='pseudoproxy')
    ax_spec.loglog(1/freqs_proxy, psd_proxy, lw=lw, color='gray', label='proxy')
    ax_spec.set_xticks(period_ticks)
    ax_spec.get_xaxis().set_major_formatter(ScalarFormatter())
    ax_spec.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax_spec.invert_xaxis()
    ax_spec.set_ylabel('PSD')
    ax_spec.set_xlabel('Period (years)')
    ax_spec.spines['right'].set_visible(False)
    ax_spec.spines['top'].set_visible(False)
    # ax_spec.set_ylim([1e-1, 1e1])
    ax_spec.set_xlim([200, 2])
    ax_spec.legend(fontsize=15, loc='upper right', ncol=1, frameon=False)

    #===========================================================

    if calc_corr:
        return fig, corr
    else:
        return fig


def plot_linreg_residual(mdl, figsize=[20, 15], title=None,
                         font_scale=2, wspace=0.3, hspace=0.3):
    ''' Plot residual analysis
    Args:
        mdl (dict): the linreg model dict from statsmodels

    Ref: adapted from GEOL425L course lab 11 by Julien Emile-Geay
    '''
    # fitted values (need a constant term for intercept)
    sns.set(style='ticks', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    model_fitted_y = mdl.fittedvalues
    # model residuals
    model_residuals = mdl.resid
    # normalized residuals
    model_norm_residuals = mdl.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)

    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=wspace, hspace=hspace)

    ax = {}
    ax[0] = plt.subplot(gs[0])
    # a) residuals vs fitted values
    ax[0].scatter(model_fitted_y, model_residuals)
    ax[0].set_title('a) Residuals vs Fitted Values')
    ax[0].set_xlabel('Fitted values')
    ax[0].set_ylabel('Residuals')
    # b) plot QQ plot of residuals
    ax[1] = plt.subplot(gs[1])
    ax[1].set_title('b) QQ plot of residuals')
    QQ = ProbPlot(model_norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=ax[1])

    # c) Autocorrelation of residuals
    ax[2] = plt.subplot(gs[2])
    autocorrelation_plot(pd.DataFrame(model_residuals),ax[2])
    dw = sm.stats.stattools.durbin_watson(model_residuals)
    ax[2].set_title(f'c) persistence of residuals, DW={dw:4.4f}')
    # need to fix grid

    # d) Scale-Location Plot
    ax[3] = plt.subplot(gs[3])
    ax[3].scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(
        model_fitted_y, model_norm_residuals_abs_sqrt,
        scatter=False, ci=False, lowess=True,
        line_kws={'color': 'black', 'lw': 1, 'alpha': 0.8}, ax=ax[3]
    )
    ax[3].set_title('d) Scale-Location')
    ax[3].set_xlabel('Fitted values')
    ax[3].set_ylabel(r'$\sqrt{|Standardized \; Residuals|}$');

    if title is not None:
        fig.suptitle(title)

    return fig, ax


def plot_nn_loss(history_dict, font_scale=1.5, figsize=[10, 4], lw=2,
                 calib_color=sns.xkcd_rgb['denim blue'], valid_color=sns.xkcd_rgb['orange']):

    sns.set(style='ticks', font_scale=font_scale)
    loss_calib = history_dict['loss']
    loss_valid = history_dict['val_loss']

    nepochs = np.size(loss_calib)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.plot(np.arange(nepochs)+1, loss_calib, '-', label='calibration', lw=lw, color=calib_color)
    ax.plot(np.arange(nepochs)+1, loss_valid, '-', label='validation', lw=lw, color=valid_color)

    calib_min_idx = np.argmin(loss_calib)
    valid_min_idx = np.argmin(loss_valid)
    calib_min_loss = loss_calib[calib_min_idx]
    valid_min_loss = loss_valid[valid_min_idx]

    label_best_calib = f'({calib_min_idx+1}, {calib_min_loss:.2f})'
    label_best_valid = f'({valid_min_idx+1}, {valid_min_loss:.2f})'
    ax.scatter(calib_min_idx+1, calib_min_loss, marker='o', color=calib_color, label=label_best_calib)
    ax.scatter(valid_min_idx+1, valid_min_loss, marker='o', color=valid_color, label=label_best_valid)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False, ncol=2)

    return fig, ax


def plot_nn_predicts(mod_eval_res_dict, data_dict, ref_label='proxy', xlim=[1901, 2000]):
    sns.set(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=[10, 6])
    ax.plot(data_dict['time'], data_dict['y'], label=ref_label, color=sns.xkcd_rgb['grey'], lw=5, alpha=0.5)

    clrs = [
        sns.xkcd_rgb['denim blue'],
        sns.xkcd_rgb['medium green'],
        sns.xkcd_rgb['pale red'],
        sns.xkcd_rgb['orange'],
        sns.xkcd_rgb['purple'],
    ]

    i = 0
    for k, v in mod_eval_res_dict.items():
        ax.plot(data_dict['time_calib'], v['predict_calib'], label=f'{k} calib', color=clrs[i], lw=3)
        ax.plot(data_dict['time_valid'], v['predict_valid'], ':', label=f'{k} valid', color=clrs[i], lw=3)
        i += 1

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False, ncol=1, bbox_to_anchor=[1.3, 1], loc='upper right')
    ax.set_xlabel('Time')
    ax.set_xlim(xlim)
    ax.set_ylabel('Value')

    return fig, ax


def plot_autocorrs(autocorrs_dict, plot_types=None, nlag=10,
                  panelsize=[4, 3], font_scale=1.5, ncol=2,
                  wsp=0.5, hsp=0.5, ylabel='Autocorrelation',
                  title=None, ptype_dict={
                    'Bivalve_d18O': 'bivalve_d18O',
                    'Corals and Sclerosponges_SrCa': 'coral_SrCa',
                    'Corals and Sclerosponges_d18O': 'coral_d18O',
                    'Corals and Sclerosponges_Rates': 'coral_rates',
                    'Ice Cores_d18O': 'ice_d18O',
                    'Ice Cores_dD': 'ice_dD',
                    'Ice Cores_MeltFeature': 'ice_melt',
                    'Lake Cores_Misc': 'lake_misc',
                    'Lake Cores_Varve': 'lake_varve',
                    'Tree Rings_WidthPages2': 'tree_TRW',
                    'Tree Rings_WoodDensity': 'tree_MXD',
                  }, colors_dict=None):
    ''' Plot the autocorrelation boxplot, adapted from statsmodels
    '''
    row_types = autocorrs_dict.keys()
    plot_types = []
    for rt in row_types:
        if rt not in ptype_dict:
            ptype_dict[rt] = rt

        plot_types.append(ptype_dict[rt])

    print(plot_types)

    ntypes = len(plot_types)

    sns.set(style='ticks', font_scale=font_scale)
    nrow = ntypes//ncol
    if ntypes%ncol > 0:
        nrow += 1

    figsize = [(panelsize[0]+wsp)*ncol, (panelsize[1]+hsp)*nrow]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=wsp, hspace=hsp)

    ax = {}
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    i = 0
    if colors_dict is None:
        colors_dict = PAGES2k.colors_dict

    for ptype, autocorrs in autocorrs_dict.items():
        if ptype_dict[ptype] in plot_types:
            print(f'Plotting {ptype}...')
            nrec = len(autocorrs)
            autocorr_array = np.ndarray((nrec, nlag))
            lags = np.arange(1, nlag+1)
            for j, autocorr in enumerate(autocorrs):
                autocorr_array[j] = autocorr[:nlag]

            df = pd.DataFrame(data=autocorr_array, columns=lags)

            # plot
            ax[ptype] = plt.subplot(gs[i])
            sns.boxplot(data=df, color=colors_dict[ptype])

            ax[ptype].set_xlim([-1, 10])
            ax[ptype].set_ylim([-0.4, 1])
            ax[ptype].spines['right'].set_visible(False)
            ax[ptype].spines['top'].set_visible(False)
            if title is None:
                ax[ptype].set_title(f'{ptype_dict[ptype]} (n={len(df)})')
            else:
                ax[ptype].set_title(f'{title}')
            ax[ptype].set_ylabel(ylabel)
            ax[ptype].set_xlabel('Lag')
            ax[ptype].axhline(y=z99 / np.sqrt(nrec), linestyle='--', color='grey')
            ax[ptype].axhline(y=z95 / np.sqrt(nrec), color='grey')
            ax[ptype].axhline(y=0.0, color='black')
            ax[ptype].axhline(y=-z95 / np.sqrt(nrec), color='grey')
            ax[ptype].axhline(y=-z99 / np.sqrt(nrec), linestyle='--', color='grey')
            i += 1

    return fig, ax


def plot_calib_dist(calib_filepath, var='SNR', ptypes=None, bins=None, xticks=None,
                    make_subplots=False, panel_size=[5, 4],
                    nrow=1, ncol=None, grid_ws=0.5, grid_hs=0.5,
                    lgd_args={'fontsize': 15, 'frameon': False}, font_scale=1.5,
                    use_PAGES2k_color=False, verbose=False):

    sns.set(style='ticks', font_scale=font_scale)

    print('>>> Loading calibration data')
    with open(calib_filepath, 'rb') as f:
        calib_data = pickle.load(f)

    lat_list = []
    lon_list = []
    type_list = []
    value_list = []

    ptype_list = []
    if ptypes is not None:
        ptype_list = ptypes

    for k, v in tqdm(calib_data.items(), desc='extracting calibration data'):
        ptype, pid = k
        if ptypes is None and ptype not in ptype_list:
            ptype_list.append(ptype)

        lat, lon = v['lat'], v['lon']
        value = v[var]
        lat_list.append(lat)
        lon_list.append(lon)
        type_list.append(ptype)
        value_list.append(value)

    df = pd.DataFrame({
        'lat': lat_list,
        'lon': lon_list,
        'type': type_list,
        var: value_list,
    })

    if verbose:
        print(df)

    print(f'>>> Plotting {var}')
    if make_subplots:
        ntypes = len(ptype_list)
        if ncol is None:
            ncol = ntypes // nrow
            if ntypes % nrow > 0:
                ncol += 1

        fig = plt.figure(figsize=[panel_size[0]*ncol, panel_size[1]*nrow])
        gs = gridspec.GridSpec(nrow, ncol)
        gs.update(wspace=grid_ws, hspace=grid_hs)

        ax = {}
        for i, ptype in enumerate(ptype_list):
            ax[i] = plt.subplot(gs[i])
            df_target = df[df['type'] == ptype]
            if use_PAGES2k_color:
                sns.distplot(df_target[var].values, label=ptype, kde=False, ax=ax[i],
                             bins=bins, color=PAGES2k.colors_dict[ptype])
            else:
                sns.distplot(df_target[var].values, label=ptype, kde=False, ax=ax[i],
                             bins=bins)

            ax[i].set_xlabel(var)
            ax[i].set_ylabel('number of records')
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].set_title(ptype)
            if xticks is not None:
                ax[i].set_xticks(xticks)
    else:
        fig, ax = plt.subplots(figsize=panel_size)
        for i, ptype in enumerate(ptype_list):
            df_target = df[df['type'] == ptype]
            if use_PAGES2k_color:
                median = np.median(df_target[var].values)
                sns.distplot(df_target[var].values, label=f'{ptype} (median={median:.2f})', kde=False, ax=ax,
                             bins=bins, color=PAGES2k.colors_dict[ptype])
            else:
                median = np.median(df_target[var].values)
                sns.distplot(df_target[var].values, label=f'{ptype} (median={median:.2f})', kde=False, ax=ax,
                             bins=bins)
            ax.legend(**lgd_args)
            if xticks is not None:
                ax.set_xticks(xticks)
            ax.set_xlabel(var)
            ax.set_ylabel('number of records')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(f'{var} distribution')

    return fig, ax


def plot_psd_from_proxyDB(psd_dict, freqs_dict, period_ticks=[2, 5, 10, 20, 50, 100, 200, 500],
                          alpha=0.3, title=None, lw=1, ptype=None):
    sns.set(style='ticks', font_scale=1.5)

    fig, ax = plt.subplots(figsize=[5, 5])
    if title:
        ax.set_title(title)

    if ptype:
        clr = PAGES2k.colors_dict[ptype]
    else:
        clr = None

    for pid, psd in psd_dict.items():
        freqs = freqs_dict[pid]
        ax.loglog(1/freqs, psd, lw=lw, color=clr, alpha=alpha)

    ax.set_xticks(period_ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.invert_xaxis()
    ax.set_ylabel('Spectral Density')
    ax.set_xlabel('Period (years)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([1e-2, 1e3])
    ax.set_xlim([500, 2])
    ax.legend(frameon=False)

    return fig, ax
