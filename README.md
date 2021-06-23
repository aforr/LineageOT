# LineageOT


This code accompanies the paper https://biorxiv.org/cgi/content/short/2020.07.31.231621v2. Documentation is available at https://lineageot.readthedocs.io/.


### Requirements and installation guide
##### Installing dependencies
First, clone this repository with
```
git clone https://github.com/aforr/LineageOT
```
Dependencies are listed in `conda_requirements.yml` as well as in `pip_requirements.txt`; earlier versions of the packages may work but have not been tested. If you are using Anaconda, they can be installed with 
```
conda env create -f conda_requirements.yml
```
Installation may take a few minutes. Activate the environment with
```
conda activate lineageOT
```
If you are using pip and not Anaconda, install dependencies with
```
pip install numpy cython
pip install -r pip_requirements.txt
```

##### Installing LineageOT
Once you have set up and activated the environment, install the most recent LineageOT by running
```
pip install .
```
from the repository's base directory. Alternatively, install from PyPI with
```
pip install lineageot
```



No specific operating system is required, though there may be a bug in one of the dependencies in certain versions of MacOS (https://github.com/PythonOT/POT/issues/93). The code has been tested on OS X 10.14.6 and Ubuntu 16.04.




### Examples

An example of LineageOT applied to simulated data is in `examples/simulation_demo.ipynb`. Running the notebook for one simulation type took around 10 minutes on a "normal" desktop.

To fit LineageOT couplings in your own system, follow the steps in `examples/pipeline_demo.ipynb` replacing the synthetic `AnnData` object with your data.
We recommend then using the downstream analysis tools available in the Waddington-OT package: https://broadinstitute.github.io/wot/.
