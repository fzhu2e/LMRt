[![PyPI](https://img.shields.io/pypi/v/LMRt.svg)]()
[![](https://img.shields.io/badge/platform-Mac_Linux-green.svg)]()
[![](https://img.shields.io/badge/language-Python3-red.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2655097.svg)](https://doi.org/10.5281/zenodo.2655097)

# LMR Turbo (LMRt)

A lightweight, packaged version of the [Last Millennium Reanalysia (LMR)](https://github.com/modons/LMR) framework,
inspired by LMR_lite.py originated by Professor Hakim (Univ. of Washington).
Ultimately, it aims to provide following features:

+ Greater flexibility
    + Easy installation
    + Easy importing and usage in Jupyter notebooks (or scripts)
    + No assumption of a fixed folder structure; just feed the correct files to functions
    + Easy setup for different priors, proxies, and Proxy System Models (PSMs) included in [PRYSM API](https://github.com/fzhu2e/prysm-api)
+ Faster speed
    + Easy parallel computing with multiprocessing and other techniques

## Results

### Mean temperature
![Mean temperature](notebooks/figs/gmt.png)

### Niño 3.4 index
![Niño 3.4](notebooks/figs/nino34.png)


## Package dependencies
+ [tqdm](https://github.com/tqdm/tqdm): A fast, extensible progress bar for Python and CLI (`pip install tqdm`)
+ [prysm-api](https://github.com/fzhu2e/prysm-api): The API for PRoxY System Modeling (PRYSM) (`pip install prysm-api`)
+ [dotmap](https://github.com/drgrib/dotmap): Dot access dictionary with dynamic hierarchy creation and ordered iteration (`pip install dotmap`)
+ [xarray](https://github.com/pydata/xarray): N-D labeled arrays and datasets in Python (`pip install xarray`)
+ [netCDF4](https://github.com/Unidata/netcdf4-python): the python interface for netCDF4 format (`conda install netCDF4`)
+ [pyspharm](https://code.google.com/archive/p/pyspharm/): an  object-oriented python interface to the NCAR SPHEREPACK library (`conda install -c conda-forge pyspharm`)

## How to install
Once the above dependencies are installed, simply
```bash
pip install LMRt
```
and you are ready to
```python
import LMRt
```
in python.

## Notebook tutorials
+ [a quickstart](https://nbviewer.jupyter.org/github/fzhu2e/LMRt/blob/master/notebooks/01.lmrt_quickstart.ipynb)
+ [building Ye files](https://nbviewer.jupyter.org/github/fzhu2e/LMRt/blob/master/notebooks/02.build_Ye.ipynb)

## References
+ Hakim, G. J., J. Emile‐Geay, E. J. Steig, D. Noone, D. M. Anderson, R. Tardif, N. Steiger, and W. A. Perkins, 2016: The last millennium climate reanalysis project: Framework and first results. Journal of Geophysical Research: Atmospheres, 121, 6745–6764, https://doi.org/10.1002/2016JD024751.
+ Tardif, R., Hakim, G. J., Perkins, W. A., Horlick, K. A., Erb, M. P., Emile-Geay, J., Anderson, D. M., Steig, E. J., and Noone, D.: Last Millennium Reanalysis with an expanded proxy database and seasonal proxy modeling, Clim. Past Discuss., https://doi.org/10.5194/cp-2018-120, in review, 2018.

## License
BSD License (see the details [here](LICENSE))

## How to cite
If you find this package useful, please cite it with DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2655097.svg)](https://doi.org/10.5281/zenodo.2655097)

... and welcome to Star and Fork!

