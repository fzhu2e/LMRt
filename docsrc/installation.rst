Installallation
===============


Install the Conda environment
-----------------------------

You may skip this step if your Conda environment has been setup already.

Step 1: Download the installation script for miniconda3
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

MacOS
'''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

Linux
'''''
.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Step 2: Install Miniconda3
"""""""""""""""""""""""""""

.. code-block:: bash

  chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-Linux-x86_64.sh

During the installation, a path :code:`<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., :code:`~/.bashrc` or :code:`~/.zshrc`) as below to enable the :code:`conda` package manager (remember to change :code:`<base-path>` with your real location):

.. code-block:: bash

  export PATH="<base-path>/bin:$PATH"
  . <base-path>/etc/profile.d/conda.sh

Step 3: Test your Installation
"""""""""""""""""""""""""""""""

.. code-block:: bash

  source ~/.bashrc  # assume you are using Bash shell
  which python  # should return a path under <base-path>
  which conda  # should return a path under <base-path>


Install LMRt
------------


Taking a clean install as example, first let's create a new environment named :code:`LMRt` via :code:`conda`

.. code-block:: bash

    conda create -n LMRt python=3.8
    conda activate LMRt

Then install two dependencies that is not able to be installed via :code:`pip`:

.. code-block:: bash

    conda install -c conda-forge cartopy pyspharm jupyterlab

Once the above dependencies have been installed, simply

.. code-block:: bash

    pip install LMRt

and you are ready to

.. code-block:: python

    import LMRt

in python.
