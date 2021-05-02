from setuptools import setup, find_packages
import LMRt

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='LMRt',  # required
    version=LMRt.__version__,
    description="LMR turbo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Feng Zhu",
    author_email='fengzhu@usc.edu',
    url='https://github.com/fzhu2e/LMRt',
    packages=find_packages(),
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    scripts=['bin/LMRt'],
    keywords='LMRt',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'termcolors',
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
