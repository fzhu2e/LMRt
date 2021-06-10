from setuptools import setup, find_packages
__version__ = '0.7.7'

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(
    name='LMRt',  # required
    version=__version__,
    description='LMR turbo',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu',
    author_email='fengzhu@usc.edu',
    url='https://github.com/fzhu2e/LMRt',
    packages=find_packages(),
    include_package_data=True,
    license='GPL-3.0 license',
    zip_safe=False,
    scripts=['bin/LMRt'],
    keywords='LMRt',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'termcolor',
        'pyyaml',
        'pandas',
        'cftime',
        'tqdm',
        'xarray',
        'netCDF4',
        'statsmodels',
        'seaborn',
        'pyleoclim',
    ],
)
