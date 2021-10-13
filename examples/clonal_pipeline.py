"""
=====================================
LineageOT with static lineage tracing
=====================================
"""

# While designed for dynamic lineage tracing with continuously edited barcodes,
# LineageOT can be applied to any time course where a lineage tree can be created,
# including static barcoding data. 
#
# With some forms of static barcoding, more information is available than LineageOT uses.
# LineageOT does not account for the possibility that the same barcode could be observed
# at multiple time points. If that happens in your data, you can still use LineageOT,
# but should also consider other methods.

import anndata
import lineageot
import numpy as np

rng = np.random.default_rng()

###############################################################################
# Creating data
# -------------
#
# First we make a minimal fake AnnData object to run LineageOT on. Here, the lineage
# information is encoded in a Boolean matrix with cells as rows and clones as column,
# where entry ``[i, j]`` is 1 if and only if cell ``i`` belongs to clone ``j``.
# This example has two disjoint clones
#
# In addition to the clone identities, LineageOT also needs a time for each clone. This is encoded in the vector ``clone_times``, whose entries give the time of labeling of the clones.

t1 = 5;
t2 = 10;

n_cells_1 = 4;
n_cells_2 = 10;
n_cells = n_cells_1 + n_cells_2;

n_genes = 5;


clones = np.concatenate([np.kron(np.identity(2),np.ones((2,1))), np.kron(np.identity(2), np.ones((5,1)))])
print(clones)
clone_times = np.array([0, 0]) # both clones labeled at time 0
adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                        obs = {"time" : np.concatenate([t1*np.ones(n_cells_1), t2*np.ones(n_cells_2)])},
                        obsm = {"X_clone" : clones}
                       )

###############################################################################
# Fitting a lineage tree
# ----------------------
#
# Before running LineageOT, we need to build a lineage tree from the observed barcodes.
# For clonal data where the clones are not nested, we provide an algorithm to construct a set of stars.
# This step is not optimized. 
# Feel free to use your own preferred tree construction algorithm.
#
# The tree should be formatted as a NetworkX ``DiGraph`` in the same way as the output of ``lineageot.fit_tree()``
# Each node is annotated with ``'time'`` (which indicates either the time of sampling (for observed cells) or the time of division (for unobserved ancestors).
# Edges are directed from parent to child and are annotated with ``'time'`` equal to the child node's ``'time_to_parent'``.
# Observed node indices correspond to their row in ``adata[adata.obs['time'] == t2]``. 

lineage_tree_t2 = lineageot.fit_tree(adata[adata.obs['time'] == t2], t2, clone_times = clone_times, method = 'clones')


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
