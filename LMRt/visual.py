''' Visualization and post processing
'''
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm

import matplotlib as mpl
import numpy as np
import os
from scipy.stats.mstats import mquantiles
from cartopy import util as cutil

from . import utils


class PAGES2k(object):
    colors_dict = {
        'Bivalve_d18O': sns.xkcd_rgb['gold'],
        'Corals and Sclerosponges_Rates': sns.xkcd_rgb['orange'],
        'Corals and Sclerosponges_SrCa': sns.xkcd_rgb['yellow'],
        'Corals and Sclerosponges_d18O': sns.xkcd_rgb['amber'],
        'Ice Cores_MeltFeature': sns.xkcd_rgb['pale blue'],
        'Ice Cores_d18O': sns.xkcd_rgb['light blue'],
        'Ice Cores_dD': sns.xkcd_rgb['sky blue'],
        'Lake Cores_Misc': sns.xkcd_rgb['blue'],
        'Lake Cores_Varve': sns.xkcd_rgb['dark blue'],
        'Tree Rings_WidthPages2': sns.xkcd_rgb['green'],
        'Tree Rings_WoodDensity': sns.xkcd_rgb['forest green'],
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
    }


def plot_proxies(df, year=np.arange(2001), lon_col='lon', lat_col='lat', type_col='type',
                 title=None, title_weight='normal', font_scale=1.5,
                 figsize=[8, 10], projection=ccrs.Robinson(), markersize=50, plot_count=True,
                 lgd_ncol=1, lgd_anchor_upper=(1, -0.1), lgd_anchor_lower=(1, -0.05),lgd_frameon=False):

    p = PAGES2k()

    sns.set(style='darkgrid', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    if plot_count:
        nrow = 2
    else:
        nrow = 1

    gs = gridspec.GridSpec(nrow, 1)
    gs.update(wspace=0, hspace=0.1)

    ax_map = plt.subplot(gs[0], projection=projection)

    if title:
        ax_map.set_title(title, fontweight=title_weight)

    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)

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
        s_plots.append(
            ax_map.scatter(
                lons, lats, marker=p.markers_dict[ptype],
                c=p.colors_dict[ptype], edgecolor='k', s=markersize, transform=ccrs.Geodetic()
            )
        )

    ax_map.legend(
        s_plots, type_names,
        scatterpoints=1,
        bbox_to_anchor=lgd_anchor_upper,
        loc='lower left',
        ncol=lgd_ncol,
        frameon=lgd_frameon,
    )

    if plot_count:
        ax_count = plt.subplot(gs[1])
        proxy_count = {}
        for index, row in df.iterrows():
            ptype = row['type']
            time = row['time']
            if ptype not in proxy_count.keys():
                proxy_count[ptype] = np.zeros(np.size(year))

            for k in time:
                if k < np.max(year):
                    proxy_count[ptype][int(k)] += 1

        cumu_count = np.zeros(np.size(year))
        cumu_last = np.copy(cumu_count)
        idx = np.argsort(max_count)
        for ptype in type_set[idx]:
            cumu_count += proxy_count[ptype]
            ax_count.fill_between(
                year, cumu_last, cumu_count,
                color=p.colors_dict[ptype],
                label=f'{ptype}',
                alpha=0.8,
            )
            cumu_last = np.copy(cumu_count)

        ax_count.set_xlabel('Year (AD)')
        ax_count.set_ylabel('number of proxies')
        handles, labels = ax_count.get_legend_handles_labels()
        ax_count.legend(handles[::-1], labels[::-1], frameon=lgd_frameon, bbox_to_anchor=lgd_anchor_lower, loc='lower left')

    return fig


def plot_proxy_age_map(df, lon_col='lon', lat_col='lat', type_col='type', time_col='time',
                       title=None, title_weight='normal', font_scale=1.5,
                       figsize=[12, 10], projection=ccrs.Robinson, central_longitude=0, markersize=150,
                       lgd_ncol=1, lgd_anchor=(1, -0.1), lgd_frameon=False,
                       add_nino34_box=False, add_nino12_box=False):

    p = PAGES2k()

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
        ax_map.plot(x, y, '--', transform=ccrs.Geodetic(), color='gray')

    if add_nino34_box:
        x, y = [-170, -170, -120, -120, -170], [5, -5, -5, 5, 5]
        ax_map.plot(x, y, '--', transform=ccrs.Geodetic(), color='gray')

    color_norm = Normalize(vmin=0, vmax=1000)

    cmap = cm.get_cmap('viridis_r', 10)
    cmap.set_under(sns.xkcd_rgb['cream'])
    cmap.set_over('black')

    ages = []
    for idx, row in df.iterrows():
        ages.append(1950-np.min(row['time']))

    df[time_col].values

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
        s_plots.append(
            ax_map.scatter(
                lons, lats, marker=p.markers_dict[ptype], cmap=cmap, norm=color_norm,
                c=ages, edgecolor='k', s=markersize, transform=ccrs.Geodetic()
            )
        )

    cbar_lm = plt.colorbar(s_plots[0], orientation='vertical',
                           pad=0.05, aspect=10, extend='min',
                           ax=ax_map, fraction=0.05, shrink=0.5)

    cbar_lm.ax.set_title(r'age [yrs]', y=1.05)
    cbar_lm.set_ticks([0, 200, 400, 600, 800, 1000])

    return fig


def plot_field_map(field_var, lat, lon, levels=50, add_cyclic_point=True,
                   title=None, title_size=20, title_weight='normal', figsize=[10, 8],
                   projection=ccrs.Robinson, transform=ccrs.PlateCarree(),
                   central_longitude=0, latlon_range=None,
                   land_alpha=1, ocean_alpha=1,
                   land_color=sns.xkcd_rgb['silver'], ocean_color=sns.xkcd_rgb['silver'],
                   land_zorder=None, ocean_zorder=None,
                   clim=None, cmap='RdBu_r', extend='both', mode='mesh', add_gridlines=False,
                   cbar_labels=None, cbar_pad=0.05, cbar_orientation='vertical', cbar_aspect=10,
                   cbar_fraction=0.15, cbar_shrink=0.5, cbar_title=None, font_scale=1.5):

    if add_cyclic_point:
        if mode == 'latlon':
            field_var_c, lon_c = cutil.add_cyclic_point(field_var, lon)
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

            lon_c[:, :-1] = lon
            lon_c[:, -1] = lon[:, 0]

            lat_c[:, :-1] = lat
            lat_c[:, -1] = lat[:, 0]

            field_var_c[:, :-1] = field_var
            field_var_c[:, -1] = field_var[:, 0]
    else:
        field_var_c, lat_c, lon_c = field_var, lat, lon

    sns.set(style='ticks', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)

    projection = projection(central_longitude=central_longitude)
    ax = plt.subplot(projection=projection)

    if title:
        plt.title(title, fontsize=title_size, fontweight=title_weight)

    if latlon_range:
        ax.set_extent(latlon_range, crs=transform)
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor='k', alpha=land_alpha, zorder=land_zorder)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, edgecolor='k', alpha=ocean_alpha, zorder=ocean_zorder)
    ax.coastlines()

    if add_gridlines:
        ax.gridlines(edgecolor='gray', linestyle=':', crs=transform)

    cmap = plt.get_cmap(cmap)

    if mode == 'latlon':
        im = ax.contourf(lon_c, lat_c, field_var_c, levels, transform=transform, cmap=cmap, extend=extend)

    elif mode == 'mesh':
        if type(levels) is int:
            levels = MaxNLocator(nbins=levels).tick_values(np.nanmax(field_var_c), np.nanmin(field_var_c))
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        im = ax.pcolormesh(lon_c, lat_c, field_var_c, transform=transform, cmap=cmap, norm=norm)

    if clim:
        im.set_clim(clim)

    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
                        fraction=cbar_fraction, shrink=cbar_shrink)

    if cbar_labels is not None:
        cbar.set_ticks(cbar_labels)

    if cbar_title:
        cbar.ax.set_title(cbar_title)

    return fig


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

    inst_gmt, inst_time = utils.load_inst_analyses(
        ana_pathdict, var=var, verif_yrs=verif_yrs, ref_period=ref_period)

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


def plot_ts_from_jobs(
    exp_dir, time_span=(0, 2000), savefig_path=None,
    plot_vars=['tas_sfc_Amon_gm_ens', 'tas_sfc_Amon_nhm_ens', 'tas_sfc_Amon_shm_ens'],
    qs=[0.025, 0.25, 0.5, 0.75, 0.975], pannel_size=[10, 4], ylabel='T anom. (K)',
    font_scale=1.5, hspace=0.5, ylim=[-1, 1], color=sns.xkcd_rgb['pale red'],
    title=None, plot_title=True, title_y=1,
    lgd_ncol=3, lgd_bbox_to_anchor=None,
    lgd_order=[0, 2, 3, 1], style='ticks',
    bias_correction=False,
    ref_value=None, ref_time=None, ref_color='k', ref_ls='-', ref_label='reference', ref_alpha=1,
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

    print(plot_vars)
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
        'nino1+2': 'Annual Niño 1+2 index',
        'nino3': 'Annual Niño 3 index',
        'nino3.4': 'Annual Niño 3.4 index',
        'nino4': 'Annual Niño 4 index',
    }

    for plot_i, var in enumerate(plot_vars):

        ts_qs, year = utils.load_ts_from_jobs(exp_dir, qs, var=var)

        mask = (year >= time_span[plot_i][0]) & (year <= time_span[plot_i][-1])
        ts_qs = ts_qs[mask, :]
        year = year[mask]

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        ax = plt.subplot(gs[plot_i, 0])
        if qs[2] == 0.5:
            label = 'median'
        else:
            label = f'{qs[2]*100}%'

        if title is None and var in ax_title.keys():
            title_str = ax_title[var]
        else:
            title_str = title

        ax.plot(year, ts_qs[:, 2], '-', color=color, alpha=1, label=f'{label}')
        ax.fill_between(year, ts_qs[:, -2], ts_qs[:, 1], color=color, alpha=0.5,
                        label=f'{qs[1]*100}% to {qs[-2]*100}%')
        ax.fill_between(year, ts_qs[:, -1], ts_qs[:, 0], color=color, alpha=0.1,
                        label=f'{qs[0]*100}% to {qs[-1]*100}%')
        ax.set_ylabel(ylabel[plot_i])
        ax.set_xlabel('Year (AD)')
        ax.set_ylim(ylim)

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

            ax.plot(ref_t, ref_v, ls=ref_ls, color=ref_color, alpha=ref_alpha, label=f'{ref_l}')
            if plot_title[plot_i]:
                ax.set_title(f'{title_str} (corr={corr:.2f}; CE={ce:.2f})', y=title_y)

            if plot_i == 0:
                if lgd_order:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(
                        [handles[idx] for idx in lgd_order], [labels[idx] for idx in lgd_order],
                        loc='upper center', ncol=lgd_ncol, frameon=False, bbox_to_anchor=lgd_bbox_to_anchor,
                    )
                else:
                    ax.legend(loc='upper center', ncol=lgd_ncol, frameon=False, bbox_to_anchor=lgd_bbox_to_anchor)
        else:
            ax.set_title(title_str, y=title_y)
            ax.legend(loc='upper center', ncol=lgd_ncol, frameon=False, bbox_to_anchor=lgd_bbox_to_anchor)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close(fig)

    return fig


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


def plot_sea_res(res, style='ticks', font_scale=2, figsize=[10, 6], signif_fontsize=15, ls='-o', color='k',
                 shade_alpha=0.3, signif_alpha=0.3, signif_color='k', signif_text_loc_fix=(0.1, -0.01),
                 xticks=None, title=None):
    ''' Plot SEA results
    '''
    sns.set(style=style, font_scale=font_scale)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(res['composite_yr'], res['composite_qs'][1], ls, color=color)
    ax.fill_between(res['composite_yr'], res['composite_qs'][0], res['composite_qs'][-1], color=color, alpha=shade_alpha)

    for i, qs_v in enumerate(res['qs_signif']):
        ax.plot(res['composite_yr'], res['composite_qs_signif'][i], '--', color=signif_color, alpha=signif_alpha)
        ax.text(res['composite_yr'][-1]+signif_text_loc_fix[0], res['composite_qs_signif'][i][-1]+signif_text_loc_fix[-1],
                f'{qs_v*100:g}%', color=signif_color, alpha=signif_alpha, fontsize=signif_fontsize)

    ax.set_ylabel('T anom. (K)')
    ax.set_xlabel('Years relative to event year')
    ax.axvline(x=0, ls=':', color='grey')
    ax.axhline(y=0, ls=':', color='grey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xticks:
        ax.set_xticks(xticks)
    if title:
        ax.set_title(title)

    return fig, ax
