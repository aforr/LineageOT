
import anndata
import numpy as np
import os
import ot

import lineageot.inference as inf


def fit_tree(adata, time, barcodes_key = 'barcodes', method = 'neighbor join'):
    """
    Fits a lineage tree to lineage barcodes of all cells in adata. To compute the lineage tree for a specific time point,
    filter adata before calling fit_tree. The fitted tree is annotated with node times but not states.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with lineage-traced cells
    time : Number
        Time of sampling of the cells of adata
    barcodes_key : str, default 'barcodes'
        Key in adata.obsm containing cell barcodes.
        Each row of adata.obsm[barcodes_key] should be a barcode where each entry corresponds to a possibly-mutated site.
        A positive number indicates an observed mutation, zero indicates no mutation, and -1 indicates the site was not observed.
    method : str
        Inference method used to fit tree to barcodes. Currently 'neighbor join' is the only option.

    Returns
    -------
    tree : Networkx DiGraph
        A fitted lineage tree
    """

    # compute distances
    barcode_length = adata.obsm[barcodes_key].shape[1]
    # last row is (unobserved) root of the tree
    lineage_distances = inf.barcode_distances(np.concatenate([adata.obsm[barcodes_key], np.zeros([1,barcode_length])]))

    # compute tree
    fitted_tree = inf.neighbor_join(lineage_distances)

    # annotate tree with node times
    inf.add_leaf_barcodes(fitted_tree, adata.obsm[barcodes_key])
    inf.add_leaf_times(fitted_tree, time)

    # Estimating a uniform mutation rate for all target sites
    rate_estimate = inf.rate_estimator(adata.obsm[barcodes_key], time)
    inf.annotate_tree(fitted_tree, 
                      rate_estimate*np.ones(barcode_length),
                      time_inference_method = 'least_squares');


    return fitted_tree



def fit_lineage_coupling(adata, time_1, time_2, lineage_tree_t2, time_key = 'time', state_key = None, epsilon = 0.05, normalize_cost = True):
    """
    Fits a LineageOT coupling between the cells in adata at time_1 and time_2. 
    In the process, annotates the lineage tree with observed and estimated cell states.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    time_1 : Number
        The earlier time point in adata. All times are relative to the root of the tree.
    time_2 : Number
        The later time point in adata. All times are relative to the root of the tree.
    lineage_tree_t2 : Networkx DiGraph
        The lineage tree fitted to cells at time_2. Nodes should already be annotated with times. Annotations related to cell state will be added. 
    time_key : str (default 'time')
        Key in adata.obs and lineage_tree_t2 containing cells' time labels
    state_key : str (default None)
        Key in adata.obsm containing cell states. If None, uses adata.X.
    epsilon : float (default 0.05)
        Entropic regularization parameter for optimal transport
    normalize_cost : bool (default True)
        Whether to rescale the cost matrix by its median before fitting a coupling. 
        Normalizing this way allows us to choose a reasonable default epsilon for data of any scale

    Returns
    -------
    coupling : AnnData
        AnnData containing the lineage coupling. 
        Cells from time_1 are in coupling.obs, cells from time_2 are in coupling.var, and the coupling matrix is coupling.X
    """

    state_arrays = {}
    if state_key == None:
        state_arrays['early'] = adata[adata.obs[time_key] == time_1].X
        state_arrays['late'] = adata[adata.obs[time_key] == time_2].X
    else:
        state_arrays['early'] = adata[adata.obs[time_key] == time_1].obsm[state_key]
        state_arrays['late'] = adata[adata.obs[time_key] == time_2].obsm[state_key]

    # annotate tree
    inf.add_leaf_x(lineage_tree_t2, state_arrays['late'])


    # Add inferred ancestor nodes and states
    inf.add_node_times_from_division_times(lineage_tree_t2)

    inf.add_nodes_at_time(lineage_tree_t2, time_1)

    observed_nodes = [n for n in inf.get_leaves(lineage_tree_t2, include_root = False)]
    inf.add_conditional_means_and_variances(lineage_tree_t2, observed_nodes)

    ancestor_info = inf.get_ancestor_data(lineage_tree_t2, time_1)


    # compute cost matrix
    lineageOT_cost = ot.utils.dist(state_arrays['early'], ancestor_info[0])@np.diag(ancestor_info[1]**(-1))

    if normalize_cost:
        lineageOT_cost = lineageOT_cost/np.median(lineageOT_cost)

    # fit coupling
    coupling_matrix = ot.sinkhorn([], [], lineageOT_cost, epsilon)


    # reformat coupling as anndata
    coupling = anndata.AnnData(X = coupling_matrix,
                               obs = adata[adata.obs[time_key] == time_1].obs,
                               var = adata[adata.obs[time_key] == time_2].obs
                               )

    return coupling





def save_coupling_as_tmap(coupling, time_1, time_2, tmap_out):
    """
    Saves a LineageOT coupling for downstream analysis with Waddington-OT. 
    A sequence of saved couplings can be loaded in ``wot`` with 
    ``wot.tmap.TransportMapModel.from_directory(tmap_out)``

    Parameters
    ----------
    coupling : AnnData
        The coupling to save.
    time_1 : Number
        The earlier time point in adata. All times are relative to the root of the tree.
    time_2 : Number
        The later time point in adata. All times are relative to the root of the tree.
    tmap_out : str
        The path and prefix to the save file name.
    """
    # Normalize columns to sum to 1
    col_sums = np.sum(coupling.X, axis = 1)
    coupling.X = coupling.X/col_sums[:, np.newaxis]

    # Add constant relative growth rates for initial cells
    coupling.obs['g0'] = 1
    coupling.obs['g1'] = 1

    # Save
    file_name = tmap_out + '_' + str(time_1) + '_' + str(time_2) + '.h5ad'
    file_dir = os.path.dirname(file_name)
    
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    coupling.write(file_name)

    # Change normalization back to what it was.
    coupling.X = coupling.X*col_sums[:, np.newaxis]
    return
