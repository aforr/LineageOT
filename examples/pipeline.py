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
# Here we make a minimal fake AnnData object to run LineageOT on.


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
# Running LineageOT
# -----------------

lineage_tree_t2 = lineageot.fit_tree(adata[adata.obs['time'] == t2], t2)
coupling = lineageot.fit_lineage_coupling(adata, t1, t2, lineage_tree_t2)

###############################################################################
# Saving 
# ------
# This saves the fitted coupling in a format Waddington-OT can import
lineageot.save_coupling_as_tmap(coupling, t1, t2, './tmaps/example')
