Your 1st LMR reconstruction
===========================

This tutorial demonstrates the basic low-level workflow in LMRt. We will
assimilate coral records from the PAGES2k v2 database and use CCSM4 as
the prior. GISTEMP will be used as the instrumental temperature
observation for the calibration of the linear regression based PSM.
After this tutorial, you are expected to get your 1st LMR reconstruction
that is ready for analysis!

Test data preparation
---------------------

To go through this tutorial, please prepare test data following the
steps: 1. Download the test case named “PAGES2k_CCSM4_GISTEMP” with this
`link <https://drive.google.com/drive/folders/1UGn-LNd_tGSjPUKa52E6ffEM-ms2VD-N?usp=sharing>`__.
2. Create a directory named “testcases” in the same directory where this
notebook sits. 3. Put the unzipped direcotry “PAGES2k_CCSM4_GISTEMP”
into “testcases”.

Below, we first load some useful packages, including our ``LMRt``.

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    
    import LMRt
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr

Preprocessing
-------------

We will first create the ``job`` object and then perform the
preprocessing steps, after which the ``job`` will be ready to run.

.. code:: ipython3

    job = LMRt.ReconJob()
    job.load_configs(cfg_path='./testcases/PAGES2k_CCSM4_GISTEMP/configs.yml', verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_configs() >>> loading reconstruction configurations from: ./testcases/PAGES2k_CCSM4_GISTEMP/configs.yml[0m
    [1m[32mLMRt: job.load_configs() >>> job.configs created[0m
    [1m[36mLMRt: job.load_configs() >>> job.configs["job_dirpath"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon[0m
    [1m[32mLMRt: job.load_configs() >>> /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon created[0m
    {'anom_period': [1951, 1980],
     'job_dirpath': '/Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon',
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

    [1m[36mLMRt: job.load_proxydb() >>> job.configs["proxydb_path"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/proxy/pages2k_dataset.pkl[0m
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

    [1m[36mLMRt: job.load_prior() >>> loading model prior fields from: {'tas': '/Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc'}[0m
    Time axis not overlap with the reference period [1951, 1980]; use its own time period as reference [850.04, 1850.96].
    [1m[30mLMRt: job.load_prior() >>> raw prior[0m
    Dataset Overview
    -----------------------
    
         Name:  tas
       Source:  /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
        Shape:  time:12012, lat:192, lon:288
    [1m[32mLMRt: job.load_prior() >>> job.prior created[0m


.. code:: ipython3

    job.load_obs(verbose=True)


.. parsed-literal::

    [1m[36mLMRt: job.load_obs() >>> loading instrumental observation fields from: {'tas': '/Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/obs/gistemp1200_ERSSTv4.nc'}[0m
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

    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["seasonalized_prior_path"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/seasonalized_prior.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["seasonalized_obs_path"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/seasonalized_obs.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["prior_loc_path"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/prior_loc.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["obs_loc_path"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/obs_loc.pkl[0m
    [1m[36mLMRt: job.calibrate_psm() >>> job.configs["precalc"]["calibed_psm_path"] = /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/calibed_psm.pkl[0m
    [1m[36mLMRt: job.seasonalize_ds_for_psm() >>> job.configs["ptype_season"] = {'coral.d18O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.SrCa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.calc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}[0m
    [1m[36mLMRt: job.seasonalize_ds_for_psm() >>> Seasonalizing variables from prior with season: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][0m


.. parsed-literal::

    Searching nearest location:   0%|          | 0/95 [00:00<?, ?it/s]

.. parsed-literal::

    [1m[32mLMRt: job.seasonalize_ds_for_psm() >>> job.seasonalized_prior created[0m


.. parsed-literal::

    Searching nearest location: 100%|██████████| 95/95 [00:05<00:00, 17.58it/s]
    /Users/fzhu/Github/LMRt/LMRt/utils.py:243: RuntimeWarning: Mean of empty slice
      tmp = np.nanmean(var[inds, ...], axis=0)


.. parsed-literal::

    [1m[32mLMRt: job.proxydb.find_nearest_loc() >>> job.proxydb.prior_lat_idx & job.proxydb.prior_lon_idx created[0m
    [1m[32mLMRt: job.proxydb.get_var_from_ds() >>> job.proxydb.records[pid].prior_time & job.proxydb.records[pid].prior_value created[0m
    [1m[36mLMRt: job.seasonalize_ds_for_psm() >>> job.configs["ptype_season"] = {'coral.d18O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.SrCa': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'coral.calc': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}[0m
    [1m[36mLMRt: job.seasonalize_ds_for_psm() >>> Seasonalizing variables from obs with season: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][0m


.. parsed-literal::

    Searching nearest location:   7%|▋         | 7/95 [00:00<00:01, 61.08it/s]

.. parsed-literal::

    [1m[32mLMRt: job.seasonalize_ds_for_psm() >>> job.seasonalized_obs created[0m


.. parsed-literal::

    Searching nearest location: 100%|██████████| 95/95 [00:01<00:00, 64.02it/s]
    Calibrating PSM:   6%|▋         | 6/95 [00:00<00:01, 59.89it/s]

.. parsed-literal::

    [1m[32mLMRt: job.proxydb.find_nearest_loc() >>> job.proxydb.obs_lat_idx & job.proxydb.obs_lon_idx created[0m
    [1m[32mLMRt: job.proxydb.get_var_from_ds() >>> job.proxydb.records[pid].obs_time & job.proxydb.records[pid].obs_value created[0m
    [1m[32mLMRt: job.proxydb.init_psm() >>> job.proxydb.records[pid].psm initialized[0m
    [1m[36mLMRt: job.calibrate_psm() >>> PSM calibration period: [1850, 2015][0m


.. parsed-literal::

    Calibrating PSM:  67%|██████▋   | 64/95 [00:00<00:00, 86.15it/s]

.. parsed-literal::

    The number of overlapped data points is 0 < 25. Skipping ...


.. parsed-literal::

    Calibrating PSM: 100%|██████████| 95/95 [00:01<00:00, 88.34it/s]

.. parsed-literal::

    [1m[32mLMRt: job.proxydb.calib_psm() >>> job.proxydb.records[pid].psm calibrated[0m
    [1m[32mLMRt: job.proxydb.calib_psm() >>> job.proxydb.calibed created[0m
    CPU times: user 24.3 s, sys: 3.36 s, total: 27.7 s
    Wall time: 20.3 s


.. parsed-literal::

    


.. code:: ipython3

    job.forward_psm(verbose=True)


.. parsed-literal::

    Forwarding PSM: 100%|██████████| 95/95 [00:00<00:00, 1455.54it/s]

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
       Source:  /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
        Shape:  time:1001, lat:192, lon:288
    [1m[32mLMRt: job.seasonalize_ds_for_psm() >>> job.prior updated[0m
    [1m[30mLMRt: job.regrid_prior() >>> regridded prior[0m
    Dataset Overview
    -----------------------
    
         Name:  tas
       Source:  /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/prior/tas_sfc_Amon_CCSM4_past1000_085001-185012.nc
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

    KF updating:   3%|▎         | 59/2001 [00:00<00:03, 588.29it/s]

.. parsed-literal::

    [1m[36mLMRt: job.run() >>> job.configs["recon_seeds"] = [0][0m
    [1m[36mLMRt: job.run() >>> job.configs["save_settings"] = {'compress_dict': {'zlib': True, 'least_significant_digit': 1}, 'output_geo_mean': False, 'target_lats': [], 'target_lons': [], 'output_full_ens': False, 'dtype': 32}[0m
    [1m[36mLMRt: job.run() >>> job.configs saved to: /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/job_configs.yml[0m
    [1m[36mLMRt: job.run() >>> seed: 0 | max: 0[0m
    [1m[36mLMRt: job.run() >>> randomized indices for prior and proxies saved to: /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/job_r00_idx.pkl[0m
    Proxy Database Overview
    -----------------------
         Source:        /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/data/proxy/pages2k_dataset.pkl
           Size:        70
    Proxy types:        {'coral.calc': 6, 'coral.SrCa': 19, 'coral.d18O': 45}


.. parsed-literal::

    KF updating: 100%|██████████| 2001/2001 [00:44<00:00, 44.65it/s] 


.. parsed-literal::

    [1m[36mLMRt: job.save_recon() >>> Reconstructed fields saved to: /Users/fzhu/Github/LMRt/docs/tutorial/testcases/PAGES2k_CCSM4_GISTEMP/recon/job_r00_recon.nc[0m
    [1m[36mLMRt: job.run() >>> DONE![0m
    CPU times: user 6min 16s, sys: 16.1 s, total: 6min 33s
    Wall time: 1min 32s


Once done, we will get the struture below in the “recon” directory:

::

   .
   ├── calibed_psm.pkl
   ├── job_configs.yml
   ├── job_r00_idx.pkl
   ├── job_r00_recon.nc
   ├── job.pkl
   ├── obs_loc.pkl
   ├── prior_loc.pkl
   ├── seasonalized_obs.pkl
   └── seasonalized_prior.pkl

