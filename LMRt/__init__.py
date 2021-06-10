#!/usr/bin/env python3
import wget
import os

from .reconjob import ReconJob
from .reconres import (
    ReconRes,
    ReconSeries,
    ReconField,
)

from .visual import (
    set_style,
    showfig,
    savefig,
    closefig,
)

from pyleoclim.core.ui import (
    Series,
    EnsembleSeries,
)

from .gridded import (
    Field,
    Dataset,
)

from .utils import (
    pp,
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)

from .proxy import (
    ProxyDatabase,
    ProxyRecord,
)

# Need a file server for the below functionality...
#
# def load_testcase(casename):
#     testcases = {}
#     testcases['PAGES2k_CCSM4_GISTEMP'] = {
#         'configs': '<URL-to-the-file>',
#     }
#     case_dirpath = f'./testcases/{casename}'
#     os.makedirs(case_dirpath, exist_ok=True)
    
#     # download configs.yml
#     cfg_path = os.path.join(case_dirpath, 'configs.yml')
#     if os.path.exists(cfg_path):
#         os.remove(cfg_path)
#     p_header(f'LMRt: >>> Downloading configuration YAML file to: {cfg_path}')
#     wget.download(testcases[casename]['configs'], cfg_path)