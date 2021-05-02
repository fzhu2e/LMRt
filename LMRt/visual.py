import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
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

class PAGES2k(object):
    colors_dict = {
        'bivalve.d18O': sns.xkcd_rgb['gold'],
        'coral.calc': sns.xkcd_rgb['yellow'],
        'coral.SrCa': sns.xkcd_rgb['orange'],
        'coral.d18O': sns.xkcd_rgb['amber'],
        'ice.melt': sns.xkcd_rgb['pale blue'],
        'ice.d18O': sns.xkcd_rgb['light blue'],
        'ice.dD': sns.xkcd_rgb['sky blue'],
        'lake.varve_thickness': sns.xkcd_rgb['dark blue'],
        'tree.TRW': sns.xkcd_rgb['green'],
        'tree.MXD': sns.xkcd_rgb['forest green'],
        'tree.ENSO': sns.xkcd_rgb['sea green'],
        'tas': sns.xkcd_rgb['pale red'],
        'pr': sns.xkcd_rgb['aqua'],
    }

    markers_dict = {
        'bivalve.d18O': 'p',
        'coral.calc': 'P',
        'coral.SrCa': 'X',
        'coral.d18O': 'o',
        'ice.metl': '<',
        'ice.d18O': 'd',
        'ice.dD': '>',
        'lake.varve_thickness': 's',
        'tree.TRW': '^',
        'tree.MXD': 'v',
        'tree.ENSO': '^',
        'tas': '^',
        'pr': 'o',
    }

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

def plot_field_map(field_var, lat, lon, levels=50, add_cyclic_point=True,
                   title=None, title_size=20, title_weight='normal', figsize=[10, 8],
                   site_lats=None, site_lons=None, site_marker='o',
                   site_markersize=50, site_color=sns.xkcd_rgb['amber'],
                   projection='Robinson', transform=ccrs.PlateCarree(),
                   proj_args=None, latlon_range=None, central_longitude=180,
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
        fig = plt.figure(figsize=figsize)

        proj_args = {} if proj_args is None else proj_args
        proj_args_default = {'central_longitude': central_longitude}
        proj_args_default.update(proj_args)
        projection = CartopySettings.projection_dict[projection](**proj_args_default)
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

def plot_proxies(df, year=np.arange(2001), lon_col='lon', lat_col='lat', type_col='type', time_col='time',
                 title=None, title_weight='normal', font_scale=1.5, markers_dict=None, colors_dict=None,
                 plot_timespan=None,  plot_xticks=[850, 1000, 1200, 1400, 1600, 1800, 2000],
                 figsize=[8, 10], projection='Robinson', proj_args=None, central_longitude=180, markersize=50,
                 plot_count=True, nrow=2, ncol=1, wspace=0.5, hspace=0.1,
                 lgd_ncol=1, lgd_anchor_upper=(1, -0.1), lgd_anchor_lower=(1, -0.05),lgd_frameon=False,
                 enumerate_ax=False, enumerate_prop={'weight': 'bold', 'size': 30}, p=PAGES2k,
                 enumerate_anchor_map=[0, 1], enumerate_anchor_count=[0, 1], map_grid_idx=0, count_grid_idx=-1):

    plt.ioff()
    fig = plt.figure(figsize=figsize)

    if not plot_count:
        nrow = 1
        ncol = 1

    gs = gridspec.GridSpec(nrow, ncol)
    gs.update(wspace=wspace, hspace=hspace)

    proj_args = {} if proj_args is None else proj_args
    proj_args_default = {'central_longitude': central_longitude}
    proj_args_default.update(proj_args)
    projection = CartopySettings.projection_dict[projection](**proj_args_default)

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
                c=colors_dict[ptype], edgecolor='k', s=markersize, transform=ccrs.PlateCarree()
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
                facecolor=colors_dict[ptype],
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
                    c=ages, edgecolor='k', s=markersize, transform=ccrs.PlateCarree()
                )
            )
        else:
            s_plots.append(
                ax_map.scatter(
                    lons, lats, marker=p.markers_dict[ptype], cmap=cmap, norm=color_norm,
                    c=marker_color, edgecolor='k', s=markersize, transform=ccrs.PlateCarree()
                )
            )

    if plot_cbar:
        cbar_lm = plt.colorbar(s_plots[0], orientation='vertical',
                               pad=0.05, aspect=10, extend='min',
                               ax=ax_map, fraction=0.05, shrink=0.5)

        cbar_lm.ax.set_title(r'age [yrs]', y=1.05)
        cbar_lm.set_ticks([0, 200, 400, 600, 800, 1000])

    return fig

def in_notebook():
    ''' Check if the code is executed in a Jupyter notebook
    
    Returns
    -------
    
    bool
    
    '''
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


def showfig(fig, close=False):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    close : bool
        if True, close the figure automatically

    See Also
    --------
    pyleoclim.utils.plotting.savefig : saves a figure to a specific path
    pyleoclim.utils.plotting.in_notebook: Functions to sense a notebook environment

    '''
    if in_notebook:
        try:
            from IPython.display import display
        except ImportError as error:
            # Output expected ImportErrors.
            print(f'{error.__class__.__name__}: {error.message}')

        display(fig)

    else:
        plt.show()

    if close:
        closefig(fig)

def closefig(fig=None):
    '''Show the figure

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The matplotlib figure object

    See Also
    --------
    pyleoclim.utils.plotting.savefig : saves a figure to a specific path
    pyleoclim.utils.plotting.in_notebook: Functions to sense a notebook environment

    '''
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()

def savefig(fig, path=None, settings={}, verbose=True):
    ''' Save a figure to a path

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        the figure to save
    path : str
        the path to save the figure, can be ignored and specify in "settings" instead
    settings : dict
        the dictionary of arguments for plt.savefig(); some notes below:
        - "path" must be specified in settings if not assigned with the keyword argument;
          it can be any existed or non-existed path, with or without a suffix;
          if the suffix is not given in "path", it will follow "format"
        - "format" can be one of {"pdf", "eps", "png", "ps"}
        
    See Also
    --------
    
    pyleoclim.utils.plotting.showfig : returns a visual of the figure. 
    '''
    if path is None and 'path' not in settings:
        raise ValueError('"path" must be specified, either with the keyword argument or be specified in `settings`!')

    savefig_args = {'bbox_inches': 'tight', 'path': path}
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
    plt.close(fig)

    if verbose:
        print(f'Figure saved at: "{str(path)}"')


def set_style(style='journal', font_scale=1.0):
    ''' Modify the visualization style
    
    This function is inspired by [Seaborn](https://github.com/mwaskom/seaborn).
    See a demo in the example_notebooks folder on GitHub to look at the different styles
    
    Parameters
    ----------
    
    style : {journal,web,matplotlib,_spines, _nospines,_grid,_nogrid}
        set the styles for the figure:
            - journal (default): fonts appropriate for paper
            - web: web-like font (e.g. ggplot)
            - matplotlib: the original matplotlib style
            In addition, the following options are available:
            - _spines/_nospines: allow to show/hide spines
            - _grid/_nogrid: allow to show gridlines (default: _grid)
    
    font_scale : float
        Default is 1. Corresponding to 12 Font Size. 
    
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)

    font_dict = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
    }

    style_dict = {}
    if 'journal' in style:
        style_dict.update({
            'axes.axisbelow': True,
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'grid.color': 'lightgrey',
            'grid.linestyle': '--',
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    elif 'web' in style:
        style_dict.update({
            'figure.facecolor': 'white',

            'axes.axisbelow': True,
            'axes.facecolor': 'whitesmoke',
            'axes.edgecolor': 'lightgrey',
            'axes.grid': True,
            'grid.color': 'white',
            'grid.linestyle': '-',
            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'text.color': 'grey',
            'axes.labelcolor': 'grey',
            'xtick.color': 'grey',
            'ytick.color': 'grey',

            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],

            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,

            'legend.frameon': False,

            'axes.linewidth': 1,
            'grid.linewidth': 1,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'patch.linewidth': 1,

            'xtick.major.width': 1.25,
            'ytick.major.width': 1.25,
            'xtick.minor.width': 0,
            'ytick.minor.width': 0,
        })
    elif 'matplotlib' in style or 'default' in style:
        mpl.rcParams.update(mpl.rcParamsDefault)
    else:
        print(f'Style [{style}] not availabel! Setting to `matplotlib` ...')
        mpl.rcParams.update(mpl.rcParamsDefault)

    if '_spines' in style:
        style_dict.update({
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,
        })
    elif '_nospines' in style:
        style_dict.update({
            'axes.spines.left': False,
            'axes.spines.bottom': False,
            'axes.spines.right': False,
            'axes.spines.top': False,
        })

    if '_grid' in style:
        style_dict.update({
            'axes.grid': True,
        })
    elif '_nogrid' in style:
        style_dict.update({
            'axes.grid': False,
        })

    # modify font size based on font scale
    font_dict.update({k: v * font_scale for k, v in font_dict.items()})

    for d in [style_dict, font_dict]:
        mpl.rcParams.update(d)