.. LineageOT documentation master file, created by
   sphinx-quickstart on Mon Apr 12 18:32:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LineageOT's documentation!
=====================================

LineageOT is a package for analyzing lineage-traced single-cell sequencing time series. It extends `Waddington-OT <https://broadinstitute.github.io/wot/>`_ to compute temporal couplings using measurements of both gene expression and lineage trees. The LineageOT couplings can be used directly by the downstream analysis tools of the Waddington-OT package, which we do not duplicate here. For full details, see our `paper <https://www.nature.com/articles/s41467-021-25133-1>`_.

All of the functionality required for running LineageOT is in the ``core`` module. The remaining modules have implementation functions and code for reproducing analyses in the paper. 

The source code, with installation instructions and examples, is available at https://github.com/aforr/LineageOT.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   auto_examples/index


Core pipeline
-------------
.. automodule:: lineageot.core
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
