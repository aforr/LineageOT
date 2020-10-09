# LineageOT-demo


Demonstration code for https://biorxiv.org/cgi/content/short/2020.07.31.231621v1


### Requirements and installation guide
Dependencies are listed in lineageOT.yml; earlier versions of the packages may work but have not been tested. If you are using Anaconda, they can be installed with 
'''
conda env create -f lineageOT.yml
'''
Installation may take a few minutes. Activate the environment with
'''
conda activate lineageOT
'''
You should then be able to run 'main.ipynb' to replicate the simulations presented in the paper.

No specific operating system is required, though there is a bug in one of the dependencies in certain versions of MacOS (https://github.com/PythonOT/POT/issues/93). The code has been tested on OS X 10.14.6 and Ubuntu 16.4.




### Demo

An example run is in main.ipynb. Running the notebook for one simulation type took around 10 minutes on a "normal" desktop.