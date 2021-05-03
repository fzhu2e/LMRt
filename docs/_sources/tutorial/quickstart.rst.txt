LMRt Quickstart
===============

In this tutorial, we will demonstrate a basic workflow of LMR. We will
assimilate coral records from the PAGES2k v2 database and use CCSM4 as
the prior. GISTEMP will be used as the instrumental temperature
observation for the calibration of the linear regression based PSM.

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    
    import LMRt
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr

.. code:: ipython3

    import wget
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    url = 'https://earth.usc.edu/~fengzhu/lmrt_tests/PAGES2k_CCSM4_GISTEMP.tar.xz'
    wget.download(url)


::


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-12-9605b1838e56> in <module>
          4 
          5 url = 'https://earth.usc.edu/~fengzhu/lmrt_tests/PAGES2k_CCSM4_GISTEMP.tar.xz'
    ----> 6 wget.download(url)
    

    ~/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/wget.py in download(url, out, bar)
        524     else:
        525         binurl = url
    --> 526     (tmpfile, headers) = ulib.urlretrieve(binurl, tmpfile, callback)
        527     filename = detect_filename(url, out, headers)
        528     if outdir:


    ~/Apps/miniconda3/envs/presto/lib/python3.8/urllib/request.py in urlretrieve(url, filename, reporthook, data)
        274 
        275             while True:
    --> 276                 block = fp.read(bs)
        277                 if not block:
        278                     break


    ~/Apps/miniconda3/envs/presto/lib/python3.8/http/client.py in read(self, amt)
        456             # Amount is given, implement using readinto
        457             b = bytearray(amt)
    --> 458             n = self.readinto(b)
        459             return memoryview(b)[:n].tobytes()
        460         else:


    ~/Apps/miniconda3/envs/presto/lib/python3.8/http/client.py in readinto(self, b)
        500         # connection, and the user is reading more bytes than will be provided
        501         # (for example, reading in 1k chunks)
    --> 502         n = self.fp.readinto(b)
        503         if not n and b:
        504             # Ideally, we would raise IncompleteRead if the content-length


    ~/Apps/miniconda3/envs/presto/lib/python3.8/socket.py in readinto(self, b)
        667         while True:
        668             try:
    --> 669                 return self._sock.recv_into(b)
        670             except timeout:
        671                 self._timeout_occurred = True


    ~/Apps/miniconda3/envs/presto/lib/python3.8/ssl.py in recv_into(self, buffer, nbytes, flags)
       1239                   "non-zero flags not allowed in calls to recv_into() on %s" %
       1240                   self.__class__)
    -> 1241             return self.read(nbytes, buffer)
       1242         else:
       1243             return super().recv_into(buffer, nbytes, flags)


    ~/Apps/miniconda3/envs/presto/lib/python3.8/ssl.py in read(self, len, buffer)
       1097         try:
       1098             if buffer is not None:
    -> 1099                 return self._sslobj.read(len, buffer)
       1100             else:
       1101                 return self._sslobj.read(len)


    KeyboardInterrupt: 


Preprocessing
-------------

We will first create the ``job`` object and then perform the
preprocessing steps, after which the ``job`` will be ready to run.

.. code:: ipython3

    job = LMRt.ReconJob()
    job.load_configs(cfg_path='../examples/pages2k_CCSM4/configs.yml', verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_configs() >>> loading reconstruction configurations from: ../examples/pages2k_CCSM4/configs.yml[0m
    [1m[32mLMRt: job.load_configs() >>> job.configs created[0m
    [1m[36mLMRt: job.load_configs() >>> job.configs["job_dirpath"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon[0m
    [1m[32mLMRt: job.load_configs() >>> /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon created[0m
    {'anom_period': [1951, 1980],
     'job_dirpath': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon',
     'job_id': 'LMRt_quickstart',
     'obs_path': {'tas': './data/obs/gistemp1200_ERSSTv4.nc'},
     'obs_varname': {'tas': 'tempanomaly'},
     'prior_path': {'tas': './data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc'},
     'prior_regrid_ntrunc': 42,
     'prior_season': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
     'prior_varname': {'tas': 'tas'},
     'proxy_frac': 0.75,
     'proxydb_path': './data/proxy/pages2k_dataset.pkl',
     'psm_calib_period': [1850, 2015],
     'ptype_psm': {'coral.SrCa': 'linear',
                   'coral.calc': 'linear',
                   'coral.d18O': 'linear'},
     'ptype_season': {'coral.SrCa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      'coral.calc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      'coral.d18O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
     'recon_loc_rad': 25000,
     'recon_nens': 100,
     'recon_period': [0, 2000],
     'recon_seeds': [0,
                     1,
                     2,
                     3,
                     4,
                     5,
                     6,
                     7,
                     8,
                     9,
                     10,
                     11,
                     12,
                     13,
                     14,
                     15,
                     16,
                     17,
                     18,
                     19],
     'recon_timescale': 1,
     'recon_vars': 'tas'}


The 1st preprocessing step is to calibrate and forward the PSM, which
consists of: + loading the proxy database, and filtering and
seasonalizing the records + loading the instrumental observations (for
calibration) + loading the prior simulations (for forwarding) + perform
the calibration + perform the forward operator

.. code:: ipython3

    job.load_proxydb(verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_proxydb() >>> job.configs["proxydb_path"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/proxy/pages2k_dataset.pkl[0m
    [1m[32mLMRt: job.load_proxydb() >>> 692 records loaded[0m
    [1m[32mLMRt: job.load_proxydb() >>> job.proxydb created[0m


.. code:: ipython3

    job.filter_proxydb(verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.filter_proxydb() >>> filtering proxy records according to: ['coral.d18O', 'coral.SrCa', 'coral.calc'][0m
    [1m[32mLMRt: job.filter_proxydb() >>> 95 records remaining[0m


.. code:: ipython3

    job.seasonalize_proxydb(verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.seasonalize_proxydb() >>> seasonalizing proxy records according to: {'coral.d18O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.SrCa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.calc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}[0m
    [1m[32mLMRt: job.seasonalize_proxydb() >>> 95 records remaining[0m
    [1m[32mLMRt: job.seasonalize_proxydb() >>> job.proxydb updated[0m


.. code:: ipython3

    job.load_prior(verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_prior() >>> loading model prior fields from: {'tas': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc'}[0m
    Time axis not overlap with the reference period [1951, 1980]; use its own time period as reference [850.04, 1850.96].
    [1m[30mLMRt: job.load_prior() >>> raw prior[0m
    Dataset Overview
    -----------------------
    
         Name:  tas
       Source:  /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
        Shape:  time:12012, lat:192, lon:288
    [1m[32mLMRt: job.load_prior() >>> job.prior created[0m


.. code:: ipython3

    job.load_obs(verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_obs() >>> loading instrumental observation fields from: {'tas': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/obs/gistemp1200_ERSSTv4.nc'}[0m
    [1m[32mLMRt: job.load_obs() >>> job.obs created[0m


.. code:: ipython3

    %%time
    job_dirpath = job.configs['job_dirpath']
    seasonalized_prior_path = os.path.join(job_dirpath, 'seasonalized_prior.pkl')
    seasonalized_obs_path = os.path.join(job_dirpath, 'seasonalized_obs.pkl')
    prior_loc_path = os.path.join(job_dirpath, 'prior_loc.pkl')
    obs_loc_path = os.path.join(job_dirpath, 'obs_loc.pkl')
    calibed_psm_path = os.path.join(job_dirpath, 'calibed_psm.pkl')
    
    job.calibrate_psm(
        seasonalized_prior_path=seasonalized_prior_path,
        seasonalized_obs_path=seasonalized_obs_path,
        prior_loc_path=prior_loc_path,
        obs_loc_path=obs_loc_path,
        calibed_psm_path=calibed_psm_path,
        verbose=True,
    )


.. parsed-literal::

    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["seasonalized_prior_path"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/seasonalized_prior.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["seasonalized_obs_path"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/seasonalized_obs.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["prior_loc_path"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/prior_loc.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["obs_loc_path"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/obs_loc.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["calibed_psm_path"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/calibed_psm.pkl[0m
    [1m[32mLMRt: job.seasonalize_ds_for_psm() >>> job.seasonalized_prior created[0m
    [1m[32mLMRt: job.proxydb.find_nearest_loc() >>> job.proxydb.prior_lat_idx & job.proxydb.prior_lon_idx created[0m
    [1m[32mLMRt: job.proxydb.get_var_from_ds() >>> job.proxydb.records[pid].prior_time & job.proxydb.records[pid].prior_value created[0m
    [1m[36mLMRt: job.seasonalize_ds_for_psm() >>> job.configs["ptype_season"] = {'coral.d18O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.SrCa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.calc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}[0m
    [1m[36mLMRt: job.seasonalize_ds_for_psm() >>> Seasonalizing variables from obs with season: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][0m


.. parsed-literal::

    /Users/fzhu/Github/LMRt_rework/LMRt/utils.py:243: RuntimeWarning: Mean of empty slice
      tmp = np.nanmean(var[inds, ...], axis=0)
    Searching nearest location:   6%|▋         | 6/95 [00:00<00:01, 57.58it/s]

.. parsed-literal::

    [1m[32mLMRt: job.seasonalize_ds_for_psm() >>> job.seasonalized_obs created[0m


.. parsed-literal::

    Searching nearest location: 100%|██████████| 95/95 [00:01<00:00, 63.04it/s]
    Calibrating PSM:   8%|▊         | 8/95 [00:00<00:01, 74.76it/s]

.. parsed-literal::

    [1m[32mLMRt: job.proxydb.find_nearest_loc() >>> job.proxydb.obs_lat_idx & job.proxydb.obs_lon_idx created[0m
    [1m[32mLMRt: job.proxydb.get_var_from_ds() >>> job.proxydb.records[pid].obs_time & job.proxydb.records[pid].obs_value created[0m
    [1m[32mLMRt: job.proxydb.init_psm() >>> job.proxydb.records[pid].psm initialized[0m
    [1m[36mLMRt: job.calibrate_psm() >>> PSM calibration period: [1850, 2015][0m


.. parsed-literal::

    Calibrating PSM:  67%|██████▋   | 64/95 [00:00<00:00, 87.57it/s]

.. parsed-literal::

    The number of overlapped data points is 0 < 25. Skipping ...


.. parsed-literal::

    Calibrating PSM: 100%|██████████| 95/95 [00:01<00:00, 87.70it/s]

.. parsed-literal::

    [1m[32mLMRt: job.proxydb.calib_psm() >>> job.proxydb.records[pid].psm calibrated[0m
    [1m[32mLMRt: job.proxydb.calib_psm() >>> job.proxydb.calibed created[0m
    CPU times: user 10.3 s, sys: 381 ms, total: 10.7 s
    Wall time: 3.13 s


.. parsed-literal::

    


.. code:: ipython3

    job.forward_psm(verbose=True)


.. parsed-literal::

    Forwarding PSM: 100%|██████████| 95/95 [00:00<00:00, 1226.43it/s]

.. parsed-literal::

    [1m[32mLMRt: job.proxydb.forward_psm() >>> job.proxydb.records[pid].psm forwarded[0m


.. parsed-literal::

    


The 2nd preprocessing step is to seasonalize and regrid the prior
fields.

.. code:: ipython3

    job.seasonalize_prior(verbose=True)
    job.regrid_prior(verbose=True)


.. parsed-literal::

    [1m[30mLMRt: job.seasonalize_prior() >>> seasonalized prior w/ season [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][0m
    Dataset Overview
    -----------------------
    
         Name:  tas
       Source:  /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
        Shape:  time:1001, lat:192, lon:288
    [1m[32mLMRt: job.seasonalize_ds_for_psm() >>> job.prior updated[0m
    [1m[30mLMRt: job.regrid_prior() >>> regridded prior[0m
    Dataset Overview
    -----------------------
    
         Name:  tas
       Source:  /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
        Shape:  time:1001, lat:42, lon:63
    [1m[32mLMRt: job.regrid_prior() >>> job.prior updated[0m


Now we are ready to dump the ``job`` object, so that we may quickly load
and continue the job any time we want without repeating the
preprocessing steps above. Note that the ``job.seasonalized_prior`` and
``job.seasonalized_obs`` below are precalculated for PSM calibration,
which may consist fields apply to multiple seasonalities. We don’t need
them any more once the PSMs are calibrated, so we delete them before
dumping the ``job`` object to make the size minimal.

.. code:: ipython3

    job.save()
    
    # The above equals to below:
    # del(job.seasonalized_prior)
    # del(job.seasonalized_obs)
    # pd.to_pickle(job, os.path.join(job_dirpath, 'job.pkl'))

Data assimilation
-----------------

Now we are ready to perform the assimilation steps. As an example, we
only run one Monte-Carlo itermation by setting
``recon_seeds=np.arange(1)``. To perform say, 10, iterations, one may
set ``recon_seeds=np.arange(10)``.

.. code:: ipython3

    %%time
    # job_dirpath = '...'  # set a correct directory path
    # job = pd.read_pickle(os.path.join(job_dirpath, 'job.pkl'))
    job.run(recon_seeds=np.arange(1), verbose=True)


.. parsed-literal::

    KF updating:   3%|▎         | 58/2001 [00:00<00:03, 570.79it/s]

.. parsed-literal::

    [1m[36mLMRt: job.run() >>> job.configs["recon_seeds"] = [0][0m
    [1m[36mLMRt: job.run() >>> job.configs["save_settings"] = {'compress_dict': {'zlib': True, 'least_significant_digit': 1}, 'output_geo_mean': False, 'target_lats': [], 'target_lons': [], 'output_full_ens': False, 'dtype': 32}[0m
    [1m[36mLMRt: job.run() >>> job.configs saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job_configs.yml[0m
    [1m[36mLMRt: job.run() >>> seed: 0 | max: 0[0m
    [1m[36mLMRt: job.run() >>> randomized indices for prior and proxies saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job_r00_idx.pkl[0m
    Proxy Database Overview
    -----------------------
         Source:        /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/proxy/pages2k_dataset.pkl
           Size:        70
    Proxy types:        {'coral.calc': 6, 'coral.SrCa': 19, 'coral.d18O': 45}


.. parsed-literal::

    KF updating: 100%|██████████| 2001/2001 [00:42<00:00, 47.25it/s] 


.. parsed-literal::

    [1m[36mLMRt: job.save_recon() >>> Reconstructed fields saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job_r00_recon.nc[0m
    [1m[36mLMRt: job.run() >>> DONE![0m
    CPU times: user 6min 1s, sys: 13.9 s, total: 6min 15s
    Wall time: 1min 32s


Once done, we will get the struture below in the “recon” directory:

::

   .
   ├── calibed_psm.pkl
   ├── job.pkl
   ├── job_configs.yml
   ├── job_r00_idx.pkl
   ├── job_r00_recon.nc
   ├── obs_loc.pkl
   ├── prior_loc.pkl
   ├── seasonalized_obs.pkl
   └── seasonalized_prior.pkl

Reproducing the reconstruction based on a given well-defined configuration YAML file
------------------------------------------------------------------------------------

We will get a configurations YAML file under the ``job_dirpath`` once
the assimilation step is done, with which we may reproduce the
reconstruction. Note that to avoid overwriting old reconstruction, we’ve
modified the last part of the ``job_dirpath`` in the YAML file to be
``recon_test``.

.. code:: ipython3

    job = LMRt.ReconJob()
    job.run_cfg('../examples/pages2k_CCSM4/recon/job_configs.yml', verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_configs() >>> loading reconstruction configurations from: ../examples/pages2k_CCSM4/recon/job_configs.yml[0m
    [1m[32mLMRt: job.load_configs() >>> job.configs created[0m
    [1m[36mLMRt: job.load_configs() >>> job.configs["job_dirpath"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test[0m
    [1m[32mLMRt: job.load_configs() >>> /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test created[0m
    {'anom_period': [1951, 1980],
     'job_dirpath': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test',
     'job_id': 'LMRt_quickstart',
     'obs_path': {'tas': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/obs/gistemp1200_ERSSTv4.nc'},
     'obs_varname': {'lat': 'lat',
                     'lon': 'lon',
                     'tas': 'tempanomaly',
                     'time': 'time'},
     'precalc': {'calibed_psm_path': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/calibed_psm.pkl',
                 'obs_loc_path': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/obs_loc.pkl',
                 'prep_savepath': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job.pkl',
                 'prior_loc_path': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/prior_loc.pkl',
                 'seasonalized_obs_path': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/seasonalized_obs.pkl',
                 'seasonalized_prior_path': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/seasonalized_prior.pkl'},
     'prior_path': {'tas': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc'},
     'prior_regrid_ntrunc': 42,
     'prior_season': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
     'prior_varname': {'lat': 'lat', 'lon': 'lon', 'tas': 'tas', 'time': 'time'},
     'proxy_frac': 0.75,
     'proxydb_path': '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/proxy/pages2k_dataset.pkl',
     'psm_calib_period': [1850, 2015],
     'ptype_psm': {'coral.SrCa': 'linear',
                   'coral.calc': 'linear',
                   'coral.d18O': 'linear'},
     'ptype_season': {'coral.SrCa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      'coral.calc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                      'coral.d18O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
     'recon_loc_rad': 25000,
     'recon_nens': 100,
     'recon_period': [0, 2000],
     'recon_seeds': [0],
     'recon_timescale': 1,
     'recon_vars': 'tas',
     'save_settings': {'compress_dict': {'least_significant_digit': 1,
                                         'zlib': True},
                       'dtype': 32,
                       'output_full_ens': False,
                       'output_geo_mean': False,
                       'target_lats': [],
                       'target_lons': []}}
    [1m[36mLMRt: job.prepare() >>> job.configs["job_dirpath"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test[0m
    [1m[36mLMRt: job.prepare() >>> job.configs["precalc"]["prep_savepath"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job.pkl[0m


.. parsed-literal::

    KF updating:   0%|          | 0/2001 [00:00<?, ?it/s]

.. parsed-literal::

    [1m[36mLMRt: job.prepare() >>> Prepration data loaded from: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job.pkl[0m
    [1m[36mLMRt: job.save_job() >>> Prepration data saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test/job.pkl[0m
    [1m[36mLMRt: job.save_job() >>> job.configs["precalc"]["prep_savepath"] = /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test/job.pkl[0m
    [1m[36mLMRt: job.run() >>> job.configs["recon_seeds"] = [0][0m
    [1m[36mLMRt: job.run() >>> job.configs["recon_vars"] = tas[0m
    [1m[36mLMRt: job.run() >>> job.configs["recon_nens"] = 100[0m
    [1m[36mLMRt: job.run() >>> job.configs["proxy_frac"] = 0.75[0m
    [1m[36mLMRt: job.run() >>> job.configs["recon_period"] = [0, 2000][0m
    [1m[36mLMRt: job.run() >>> job.configs["recon_timescale"] = 1[0m
    [1m[36mLMRt: job.run() >>> job.configs["recon_loc_rad"] = 25000[0m
    [1m[36mLMRt: job.run() >>> job.configs["save_settings"] = {'compress_dict': {'least_significant_digit': 1, 'zlib': True}, 'output_geo_mean': False, 'target_lats': [], 'target_lons': [], 'output_full_ens': False, 'dtype': 32}[0m
    [1m[36mLMRt: job.run() >>> job.configs saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test/job_configs.yml[0m
    [1m[36mLMRt: job.run() >>> seed: 0 | max: 0[0m
    [1m[36mLMRt: job.run() >>> randomized indices for prior and proxies saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test/job_r00_idx.pkl[0m
    Proxy Database Overview
    -----------------------
         Source:        /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/data/proxy/pages2k_dataset.pkl
           Size:        70
    Proxy types:        {'coral.calc': 6, 'coral.SrCa': 19, 'coral.d18O': 45}


.. parsed-literal::

    KF updating: 100%|██████████| 2001/2001 [00:45<00:00, 43.96it/s] 


.. parsed-literal::

    [1m[36mLMRt: job.save_recon() >>> Reconstructed fields saved to: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon_test/job_r00_recon.nc[0m
    [1m[36mLMRt: job.run() >>> DONE![0m


Once done, we will get the struture below in the “recon” directory:

::

   .
   ├── job.pkl
   ├── job_configs.yml
   ├── job_r00_idx.pkl
   └── job_r00_recon.nc

Visualization functionalities
-----------------------------

Plot the whole loaded proxy database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # job_dirpath = '...'  # set a correct directory path
    # job_dirpath = '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon'
    job = pd.read_pickle(os.path.join(job_dirpath, 'job.pkl'))
    fig, ax = job.proxydb.plot()


.. parsed-literal::

    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_29_1.png


Plot a specific proxy record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    fig, ax = job.proxydb.records['Ocn_103'].plot()



.. image:: quickstart_files/quickstart_31_0.png


Plot a prior/obs field
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    fig, ax = job.prior.fields['tas'].plot(idx_t=-1)


.. parsed-literal::

    /Users/fzhu/Github/LMRt_rework/LMRt/visual.py:218: MatplotlibDeprecationWarning: The 'extend' parameter to Colorbar has no effect because it is overridden by the mappable; it is deprecated since 3.3 and will be removed two minor releases later.
      cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,



.. image:: quickstart_files/quickstart_33_1.png


.. code:: ipython3

    fig, ax = job.obs.fields['tas'].plot(idx_t=-1)



.. image:: quickstart_files/quickstart_34_0.png


Plot the reconstructed series and fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    job_dirpath = '/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon'
    res = LMRt.ReconRes(job_dirpath, verbose=True)


.. parsed-literal::

    recon_paths: ['/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job_r00_recon.nc']
    idx_paths: ['/Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job_r00_idx.pkl']
    job_path: /Users/fzhu/Github/LMRt_rework/examples/pages2k_CCSM4/recon/job.pkl


.. code:: ipython3

    res.get_vars(['tas', 'nino3.4'], verbose=True)


.. parsed-literal::

    [1m[30mLMRt: res.get_var() >>> loading variable: tas[0m
    [1m[30mLMRt: res.get_var() >>> loading variable: nino3.4[0m
    [1m[32mLMRt: res.get_var() >>> res.vars filled w/ varnames: ['tas', 'nino3.4'] and "year | lat | lon"[0m


.. code:: ipython3

    fig, ax = res.vars['nino3.4'].plot_envelope(xlim=[850, 2000])



.. image:: quickstart_files/quickstart_38_0.png


.. code:: ipython3

    fig, ax = res.vars['tas'].field_list[0].plot()


.. parsed-literal::

    /Users/fzhu/Github/LMRt_rework/LMRt/visual.py:218: MatplotlibDeprecationWarning: The 'extend' parameter to Colorbar has no effect because it is overridden by the mappable; it is deprecated since 3.3 and will be removed two minor releases later.
      cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_39_1.png


Plot validation of the reconstructed field against a target field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ds = LMRt.Dataset().load_nc(
        {'tas':'../examples/pages2k_CCSM4/data/obs/tas_sfc_Amon_20CR_185101-201112.nc'},
        anom_period=[1951, 1980],
        varname_dict={'tas': 'tas'}
    )
    target_fd = ds.fields['tas']
    target_fd = target_fd.seasonalize(list(range(1, 13)))

.. code:: ipython3

    corr_fd = res.vars['tas'].validate(target_fd, stat='corr')
    fig, ax = corr_fd.plot()


.. parsed-literal::

    Calculating metric: corr: 100%|██████████| 42/42 [00:15<00:00,  2.65it/s]
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Users/fzhu/Github/LMRt_rework/LMRt/visual.py:218: MatplotlibDeprecationWarning: The 'extend' parameter to Colorbar has no effect because it is overridden by the mappable; it is deprecated since 3.3 and will be removed two minor releases later.
      cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_42_1.png


.. code:: ipython3

    R2_fd = res.vars['tas'].validate(target_fd, stat='R2')
    fig, ax = R2_fd.plot()


.. parsed-literal::

    Calculating metric: R2: 100%|██████████| 42/42 [00:15<00:00,  2.66it/s]
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Users/fzhu/Github/LMRt_rework/LMRt/visual.py:218: MatplotlibDeprecationWarning: The 'extend' parameter to Colorbar has no effect because it is overridden by the mappable; it is deprecated since 3.3 and will be removed two minor releases later.
      cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_43_1.png


.. code:: ipython3

    ce_fd = res.vars['tas'].validate(target_fd, stat='CE')
    fig, ax = ce_fd.plot()


.. parsed-literal::

    Calculating metric: CE: 100%|██████████| 42/42 [00:01<00:00, 23.60it/s]



.. image:: quickstart_files/quickstart_44_1.png


Plot validation of the reconstructed series against a target field/series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    corr_nino34_fd = res.vars['nino3.4'].validate(target_fd, stat='corr')
    fig, ax = corr_nino34_fd.plot()


.. parsed-literal::

    Calculating metric: corr: 100%|██████████| 91/91 [01:41<00:00,  1.11s/it]
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)
    /Users/fzhu/Github/LMRt_rework/LMRt/visual.py:218: MatplotlibDeprecationWarning: The 'extend' parameter to Colorbar has no effect because it is overridden by the mappable; it is deprecated since 3.3 and will be removed two minor releases later.
      cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, pad=cbar_pad, aspect=cbar_aspect, extend=extend,
    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_46_1.png


.. code:: ipython3

    from scipy.io import loadmat
    
    data = loadmat('../examples/pages2k_CCSM4/data/obs/NINO34_BC09.mat')
    syr, eyr = 1873, 2000
    nyr = eyr-syr+1
    nino34 = np.zeros(nyr)
    for i in range(nyr):
        nino34[i] = np.mean(data['nino34'][i*12:12+i*12])
        
    target_series = LMRt.Series(time=np.arange(syr, eyr+1), value=nino34, label='BC09')
    fig, ax = target_series.plot()



.. image:: quickstart_files/quickstart_47_0.png


.. code:: ipython3

    fig, ax = res.vars['nino3.4'].validate(target_series, verbose=True).plot(xlim=[1880, 2000])


.. parsed-literal::

    [1m[36mLMRt: res.ReconSeries.validate() >>> valid_period = [1880, 2000][0m



.. image:: quickstart_files/quickstart_48_1.png


Plot validation of the reconstructed field against the whole proxy database or a single proxy record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    fig, ax = res.vars['tas'].validate(job.proxydb, stat='R2').plot()


.. parsed-literal::

    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_50_1.png


.. code:: ipython3

    fig, ax = res.vars['tas'].validate(job.proxydb.records['Ocn_103'], stat='corr').plot()


.. parsed-literal::

    /Users/fzhu/Apps/miniconda3/envs/presto/lib/python3.8/site-packages/cartopy/mpl/geoaxes.py:387: MatplotlibDeprecationWarning: 
    The 'inframe' parameter of draw() was deprecated in Matplotlib 3.3 and will be removed two minor releases later. Use Axes.redraw_in_frame() instead. If any parameter follows 'inframe', they should be passed as keyword, not positionally.
      return matplotlib.axes.Axes.draw(self, renderer=renderer,



.. image:: quickstart_files/quickstart_51_1.png


