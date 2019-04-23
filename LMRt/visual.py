''' Visualization and post processing
'''
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import numpy as np
import os
import glob
from scipy.stats.mstats import mquantiles
from cartopy import util as cutil

from . import utils


def plot_field_map(field_var, lat, lon, levels=50, add_cyclic_point=False,
                   title=None, title_size=20, title_weight='normal', figsize=[10, 8],
                   projection=ccrs.Robinson(), transform=ccrs.PlateCarree(),
                   clim=None, cmap='RdBu_r', extend='both',
                   cbar_labels=None, cbar_pad=0.05, cbar_orientation='vertical', cbar_aspect=10,
                   cbar_fraction=0.15, cbar_shrink=0.5, cbar_title='[K]', font_scale=1.5):

    if add_cyclic_point:
        field_var_c, lon_c = cutil.add_cyclic_point(field_var, lon)
    else:
        field_var_c, lon_c = field_var, lon

    sns.set(style='white', font_scale=font_scale)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=projection)

    if title:
        plt.title(title, fontsize=title_size, fontweight=title_weight)

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='gray', alpha=0.3)
    ax.coastlines()

    im = ax.contourf(lon_c, lat, field_var_c, levels, extend=extend,
                     transform=transform, cmap=cmap)

    if clim:
        im.set_clim(clim)

    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect,
                        fraction=cbar_fraction, shrink=cbar_shrink)

    if cbar_labels is not None:
        cbar.set_ticks(cbar_labels)

    if cbar_title:
        cbar.ax.set_title(cbar_title)

    return fig


def plot_gmt_vs_inst(gmt_qs, year, ana_pathdict,
                     verif_yrs=np.arange(1880, 2001), ref_period=[1951, 1980],
                     var='gmt', lmr_label='LMR'):
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

    sns.set(style="darkgrid", font_scale=2)
    fig, ax = plt.subplots(figsize=[16, 10])

    ax.plot(verif_yrs, lmr_gmt[:, 1], '-', lw=3, color=sns.xkcd_rgb['black'], alpha=1, label=lmr_label)
    ax.fill_between(verif_yrs, lmr_gmt[:, 0], lmr_gmt[:, -1], color=sns.xkcd_rgb['black'], alpha=0.1)
    for name in inst_gmt.keys():
        ax.plot(inst_time[name], inst_gmt[name], '-', alpha=1,
                label=f'{name} (corr={corr_vs_lmr[name]:.2f}; CE={ce_vs_lmr[name]:.2f})')

    ax.set_xlim([syear, eyear])
    ax.set_ylim([-0.6, 0.8])
    ax.set_ylabel('Temperature anomaly (K)')
    ax.set_xlabel('Year (AD)')
    ax.set_title('Global mean temperature')
    ax.legend(frameon=False)

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


def plot_gmt_ts(exp_dir, savefig_path=None, plot_vars=['gmt_ensemble', 'nhmt_ensemble', 'shmt_ensemble'],
                qs=[0.025, 0.25, 0.5, 0.75, 0.975], pannel_size=[10, 4], font_scale=1.5, hspace=0.5, ylim=[-1, 1],
                plot_prior=False, prior_var_name='tas_sfc_Amon'):
    ''' Plot timeseries

    Args:
        exp_dir (str): the path of the results directory that contains subdirs r0, r1, ...

    Returns:
        fig (figure): the output figure
    '''
    # load data
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    paths = sorted(glob.glob(os.path.join(exp_dir, 'r*')))
    filename = 'gmt_ensemble.npz'
    data = np.load(os.path.join(paths[0], filename))
    gmt_tmp = data['gmt_ensemble']
    nt = np.shape(gmt_tmp)[0]
    nEN = np.shape(gmt_tmp)[-1]
    nMC = len(paths)

    nvar = len(plot_vars)
    sns.set(style="darkgrid", font_scale=font_scale)
    fig = plt.figure(figsize=[pannel_size[0], pannel_size[1]*nvar])

    ax_title = {
            'gmt_ensemble': 'Global mean temperature',
            'shmt_ensemble': 'SH mean temperature',
            'nhmt_ensemble': 'NH mean temperature',
            }

    for plot_i, var in enumerate(plot_vars):
        gmt = np.ndarray((nt, nEN*nMC))
        for i, path in enumerate(paths):
            data = np.load(os.path.join(path, filename))
            gmt[:, nEN*i:nEN+nEN*i] = data[var]

        gmt_qs = mquantiles(gmt, qs, axis=-1)

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        to = np.arange(nt)
        ax = plt.subplot(gs[plot_i, 0])
        if qs[2] == 0.5:
            label='median'
        else:
            label='{}%'.format(qs[2]*100)

        ax.plot(to, gmt_qs[:,2], '-', color=sns.xkcd_rgb['pale red'], alpha=1, label='{}'.format(label))
        ax.fill_between(to, gmt_qs[:,-2], gmt_qs[:,1], color=sns.xkcd_rgb['pale red'], alpha=0.5,
                label='{}% to {}%'.format(qs[1]*100, qs[-2]*100))
        ax.fill_between(to, gmt_qs[:,-1], gmt_qs[:,0], color=sns.xkcd_rgb['pale red'], alpha=0.1,
                label='{}% to {}%'.format(qs[0]*100, qs[-1]*100))
        ax.set_title(ax_title[var])
        ax.set_ylabel('T anom. (K)')
        ax.set_xlabel('Year (AD)')
        ax.legend(loc='upper center', ncol=3, frameon=False)
        ax.set_ylim(ylim)

        if plot_prior:
            prior_gmt = np.zeros([nMC,nEN,nt])
            prior_nhmt = np.zeros([nMC,nEN,nt])
            prior_shmt = np.zeros([nMC,nEN,nt])
            for citer, path in enumerate(paths):
                data = np.load(os.path.join(path, 'Xb_one.npz'))
                Xb_one = data['Xb_one']
                Xb_one_coords = data['Xb_one_coords']
                state_info = data['state_info'].item()
                posbeg = state_info[prior_var_name]['pos'][0]
                posend = state_info[prior_var_name]['pos'][1]
                tas_prior = Xb_one[posbeg:posend+1, :]
                tas_coords = Xb_one_coords[posbeg:posend+1, :]
                nlat, nlon = state_info[prior_var_name]['spacedims']
                lat_lalo = tas_coords[:, 0].reshape(nlat, nlon)
                nstate, nens = tas_prior.shape
                tas_lalo = tas_prior.transpose().reshape(nens, nlat, nlon)
                [gmt,nhmt,shmt] = utils.global_hemispheric_means(tas_lalo, lat_lalo[:, 0])

                prior_gmt[citer,:,:]  = np.repeat(gmt[:,np.newaxis],nt,1)
                prior_nhmt[citer,:,:] = np.repeat(nhmt[:,np.newaxis],nt,1)
                prior_shmt[citer,:,:] = np.repeat(shmt[:,np.newaxis],nt,1)

            if var == 'gmt_ensemble':
                gmtp = prior_gmt.transpose(2,0,1).reshape(nt,-1)
            elif var == 'nhmt_ensemble':
                gmtp = prior_nhmt.transpose(2,0,1).reshape(nt,-1)
            elif var == 'shmt_ensemble':
                gmtp = prior_shmt.transpose(2,0,1).reshape(nt,-1)

            gmtp_qs = mquantiles(gmtp, qs, axis=-1)

            ax.plot(to, gmtp_qs[:,2], '-', color=sns.xkcd_rgb['grey'], alpha=1)
            ax.fill_between(to, gmtp_qs[:,3], gmtp_qs[:,1], color=sns.xkcd_rgb['grey'], alpha=0.5)
            ax.fill_between(to, gmtp_qs[:,-1], gmtp_qs[:,0], color=sns.xkcd_rgb['grey'], alpha=0.1)

    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_gmt_ts_from_jobs(exp_dir, savefig_path=None, plot_vars=['gmt_ens', 'nhmt_ens', 'shmt_ens'],
                          qs=[0.025, 0.25, 0.5, 0.75, 0.975], pannel_size=[10, 4],
                          font_scale=1.5, hspace=0.5, ylim=[-1, 1]):
    ''' Plot timeseries

    Args:
        exp_dir (str): the path of the results directory that contains subdirs r0, r1, ...

    Returns:
        fig (figure): the output figure
    '''
    # load data
    if not os.path.exists(exp_dir):
        raise ValueError('ERROR: Specified path of the results directory does not exist!!!')

    nvar = len(plot_vars)

    sns.set(style="darkgrid", font_scale=font_scale)
    fig = plt.figure(figsize=[pannel_size[0], pannel_size[1]*nvar])

    ax_title = {
        'gmt_ens': 'Global mean temperature',
        'shmt_ens': 'SH mean temperature',
        'nhmt_ens': 'NH mean temperature',
    }

    for plot_i, var in enumerate(plot_vars):

        gmt_qs, year = utils.load_gmt_from_jobs(exp_dir, qs, var=var)

        # plot
        gs = gridspec.GridSpec(nvar, 1)
        gs.update(wspace=0, hspace=hspace)

        ax = plt.subplot(gs[plot_i, 0])
        if qs[2] == 0.5:
            label='median'
        else:
            label='{}%'.format(qs[2]*100)

        ax.plot(year, gmt_qs[:,2], '-', color=sns.xkcd_rgb['pale red'], alpha=1, label='{}'.format(label))
        ax.fill_between(year, gmt_qs[:,-2], gmt_qs[:,1], color=sns.xkcd_rgb['pale red'], alpha=0.5,
                label='{}% to {}%'.format(qs[1]*100, qs[-2]*100))
        ax.fill_between(year, gmt_qs[:,-1], gmt_qs[:,0], color=sns.xkcd_rgb['pale red'], alpha=0.1,
                label='{}% to {}%'.format(qs[0]*100, qs[-1]*100))
        ax.set_title(ax_title[var])
        ax.set_ylabel('T anom. (K)')
        ax.set_xlabel('Year (AD)')
        ax.legend(loc='upper center', ncol=3, frameon=False)
        ax.set_ylim(ylim)

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
    cbar1 = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=pad, fraction=fraction)
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
    cbar2 = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=pad, fraction=fraction)
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
    cbar3 = fig.colorbar(im, ax=ax3, orientation='horizontal', pad=pad, fraction=fraction)
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
    cbar4 = fig.colorbar(im, ax=ax4, orientation='horizontal', pad=pad, fraction=fraction)
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
    Xevents_qs = mquantiles(Xcomp, qs, axis=-1)

    sns.set(style="darkgrid", font_scale=2)
    fig, ax = plt.subplots(figsize=figsize)

    if title_str:
        ax.set_title(title_str)
    ax.plot(tcomp, Xevents_qs[:, 1], '-', lw=3, color=clr)
    ax.fill_between(tcomp, Xevents_qs[:, 0], Xevents_qs[:, -1], alpha=0.3, color=clr)
    ax.axvline(x=0, ls=':', color='grey')
    ax.axhline(y=0, ls=':', color='grey')
    ax.set_xlabel('Year')
    ax.set_ylim(ylim)
    ax.set_yticks(np.arange(-0.6, 0.4, 0.2))
    ax.set_ylabel(ylabel)

    return fig
