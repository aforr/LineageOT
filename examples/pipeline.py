"""
========================
Minimal pipeline example
========================
"""

import anndata
import lineageot
import numpy as np

rng = np.random.default_rng()

###############################################################################
# Creating data
# -------------
#
# First we make a minimal fake AnnData object to run LineageOT on.


t1 = 5;
t2 = 10;

n_cells_1 = 5;
n_cells_2 = 10;
n_cells = n_cells_1 + n_cells_2;

n_genes = 5;

barcode_length = 10;

adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                        obs = {"time" : np.concatenate([t1*np.ones(n_cells_1), t2*np.ones(n_cells_2)])},
                        obsm = {"barcodes" : rng.integers(low = -1, high = 10, size = (n_cells, barcode_length))}
                       )

###############################################################################
# Fitting a lineage tree
# ----------------------
#
# Before running LineageOT, we need to build a lineage tree from the observed barcodes.
# This step is not optimized. We provide an implementation of a heuristic algorithm called neighbor joining.
# Feel free to use your own preferred tree construction algorithm.
#
# The tree should be formatted as a NetworkX ``DiGraph`` in the same way as the output of ``lineageot.fit_tree()``
# Each node is annotated with ``'time'`` (which indicates either the time of sampling (for observed cells) or the time of division (for unobserved ancestors).
# Edges are directed from parent to child and are annotated with ``'time'`` equal to the child node's ``'time_to_parent'``.
# Observed node indices correspond to their row in ``adata[adata.obs['time'] == t2]``. 

lineage_tree_t2 = lineageot.fit_tree(adata[adata.obs['time'] == t2], t2)


###############################################################################
# Running LineageOT
# -----------------
#
# Once we have a lineage tree annotated with time, we can compute a LineageOT coupling.
coupling = lineageot.fit_lineage_coupling(adata, t1, t2, lineage_tree_t2)

###############################################################################
# Saving 
# ------
# The LineageOT package does not include functionality for downstream analysis and plotting.
# We recommend transitioning to other packages, like `Waddington-OT <https://broadinstitute.github.io/wot/>`_, after computing a coupling.
# This saves the fitted coupling in a format Waddington-OT can import.

lineageot.save_coupling_as_tmap(coupling, t1, t2, './tmaps/example')
