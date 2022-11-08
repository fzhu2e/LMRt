.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2655097.svg
   :target: https://doi.org/10.5281/zenodo.2655097

.. image:: https://img.shields.io/github/last-commit/fzhu2e/LMRt/master
    :target: https://github.com/fzhu2e/LMRt

.. image:: https://img.shields.io/github/license/fzhu2e/LMRt
    :target: https://github.com/fzhu2e/LMRt/blob/master/LICENSE

.. image:: https://img.shields.io/pypi/pyversions/LMRt
    :target: https://pypi.org/project/LMRt

.. image:: https://img.shields.io/pypi/v/LMRt.svg
    :target: https://pypi.org/project/LMRt

****************
LMR Turbo (LMRt)
****************


LMR Turbo (LMRt) is a lightweight, packaged version of the `Last Millennium Reanalysia (LMR) <https://github.com/modons/LMR>`_ framework,
inspired by LMR_lite.py originated by `Professor Hakim <https://atmos.washington.edu/~hakim/>`_.
LMRt aims to provide following extra features:

+ a package that is easy to install and import in scripts or Jupyter notebooks
+ modularized workflows at different levels:

  + the low-level workflow focuses on the flexibility and customizability
  + the high-level workflow focuses on the convenience of repeating Monte-Carlo iterations
  + the top-level workflow focuses on the convenience of reproducing an experiment purely based on a given configuration YAML file

+ convenient visualization functionalities for diagnosis and validations (leveraging the :code:`Series` and :code:`EnsembleSeries` of the `Pyleoclim <https://github.com/LinkedEarth/Pyleoclim_util>`_ UI)

A preview of the results
========================

Mean temperature
----------------
.. figure:: https://github.com/fzhu2e/LMRt/raw/master/imgs/gmt.png
    :alt: Mean temperature

Niño 3.4 index
--------------
.. figure:: https://github.com/fzhu2e/LMRt/raw/master/imgs/nino34.png
    :alt: Niño 3.4


Documentation
=============

+ Homepage: https://fzhu2e.github.io/LMRt
+ Installation: https://fzhu2e.github.io/LMRt/installation.html
+ Tutorial (html): https://fzhu2e.github.io/LMRt/tutorial.html
+ Tutorial (Jupyter notebooks): https://github.com/fzhu2e/LMRt/tree/master/docsrc/tutorial

References of the LMR framework
===============================

+ Hakim, G. J., J. Emile-Geay, E. J. Steig, D. Noone, D. M. Anderson, R. Tardif, N. Steiger, and W. A. Perkins, 2016: The last millennium climate reanalysis project: Framework and first results. Journal of Geophysical Research: Atmospheres, 121, 6745–6764, https://doi.org/10.1002/2016JD024751.
+ Tardif, R., Hakim, G. J., Perkins, W. A., Horlick, K. A., Erb, M. P., Emile-Geay, J., et al. (2019). Last Millennium Reanalysis with an expanded proxy database and seasonal proxy modeling. Climate of the Past, 15(4), 1251–1273. https://doi.org/10.5194/cp-15-1251-2019


Published studies using LMRt
============================
+ Zhu, F., Emile-Geay, J., Hakim, G. J., King, J., & Anchukaitis, K. J. (2020). Resolving the Differences in the Simulated and Reconstructed Temperature Response to Volcanism. Geophysical Research Letters, 47(8), e2019GL086908. https://doi.org/10.1029/2019GL086908
+ Zhu, F., Emile-Geay, J., Anchukaitis, K. J., Hakim, G. J., Wittenberg, A. T., Morales, M. S., Toohey, M., & King, J. (2022). A re-appraisal of the ENSO response to volcanism with paleoclimate data assimilation. Nature Communications, 13(1), 747. https://doi.org/10.1038/s41467-022-28210-1


How to cite
===========
If you find this package useful, please cite it with `DOI: 10.5281/zenodo.2655097 <https://doi.org/10.5281/zenodo.2655097>`_ along with the below studies:

.. code-block::

    @article{zhu_re-appraisal_2022,
    	title = {A re-appraisal of the {ENSO} response to volcanism with paleoclimate data assimilation},
    	volume = {13},
    	issn = {2041-1723},
    	url = {https://www.nature.com/articles/s41467-022-28210-1},
    	doi = {10.1038/s41467-022-28210-1},
    	language = {en},
    	number = {1},
    	journal = {Nature Communications},
    	author = {Zhu, Feng and Emile-Geay, Julien and Anchukaitis, Kevin J. and Hakim, Gregory J. and Wittenberg, Andrew T. and Morales, Mariano S. and Toohey, Matthew and King, Jonathan},
    	month = feb,
    	year = {2022},
    	pages = {747},
    }

    @article{zhu_resolving_2020,
    	title = {Resolving the {Differences} in the {Simulated} and {Reconstructed} {Temperature} {Response} to {Volcanism}},
    	volume = {47},
    	issn = {1944-8007},
    	url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019GL086908},
    	doi = {10.1029/2019GL086908},
    	language = {en},
    	number = {8},
    	journal = {Geophysical Research Letters},
    	author = {Zhu, Feng and Emile-Geay, Julien and Hakim, Gregory J. and King, Jonathan and Anchukaitis, Kevin J.},
    	year = {2020},
    	pages = {e2019GL086908},
    }
