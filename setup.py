from setuptools import setup, find_packages

__author__ = 'Feng Zhu'
__email__ = 'fengzhu@usc.edu'
__version__ = '0.6.8'

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='LMRt',
    version=__version__,
    description='A lightweight, packaged version of the Last Millennium Reanalysis (LMR) framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Feng Zhu',
    author_email='fengzhu@usc.edu',
    url='https://github.com/fzhu2e/LMRt',
    packages=find_packages(),
    license="MIT license",
    zip_safe=False,
    package_data={
        'LMRt': ['cfg/*.yml', 'data/*'],
    },
    keywords='Paleoclimate Data Assimilation',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'tqdm',
        'dotmap',
        'pyyaml',
        'prysm-api',
        'xarray',
        'statsmodels',
        'nitime',
        'netCDF4',
        'seaborn',
        'scikit-learn',
        'keras',
        'tensorflow',
    ],
)
