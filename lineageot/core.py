
import anndata
import newick
import numpy as np
import os
import ot

import lineageot.inference as inf


def fit_tree(adata, time, barcodes_key = 'barcodes', clones_key = "X_clone", clone_times = None, method = 'neighbor join'):
    """
    Fits a lineage tree to lineage barcodes of all cells in adata. To compute the lineage tree for a specific time point,
    filter adata before calling fit_tree. The fitted tree is annotated with node times but not states.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with lineage-traced cells
    time : Number
        Time of sampling of the cells of adata relative to most recent common ancestor (for dynamic lineage tracing) or labeling time (for static lineage tracing).
    barcodes_key : str, default 'barcodes'
        Key in adata.obsm containing cell barcodes. Ignored if using clonal data.
        If using barcode data, each row of adata.obsm[barcodes_key] should be a barcode where each entry corresponds to a possibly-mutated site.
        A positive number indicates an observed mutation, zero indicates no mutation, and -1 indicates the site was not observed.
    clones_key: str, default 'X_clone'
        Key in adata.obsm containing clonal data. Ignored if using barcodes directly.
        If using clonal data, adata.obsm[clones_key] should be a num_cells x num_clones boolean matrix.
        Each entry is 1 if the corresponding cell belongs to the corresponding clone and zero otherwise.
    clone_times: Vector of length num_clones, default None
        Ignored unless method is 'clones'.
        Each entry contains the time of labeling of the corresponding column of adata.obsm[clones_key].
    method : str
        Inference method used to fit tree.
        Current options are 'neighbor join' (for barcodes from dynamic lineage tracing), 'non-nested clones' (for non-nested clones from static lineage tracing), or 'clones' (for possibly-nested clones from static lineage tracing).

    Returns
    -------
    tree : Networkx DiGraph
        A fitted lineage tree. 
        Each node is annotated with 'time_to_parent' and 'time' (which indicates either the time of sampling (for observed cells) or the time of division (for unobserved ancestors)). 
        Edges are directed from parent to child and are annotated with 'time' equal to the child node's 'time_to_parent'.
        Observed node indices correspond to their row in adata.
    """

    if method == "neighbor join":
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
    elif method == "non-nested clones":
        # check to confirm clones are not nested
        if not np.all(np.sum(adata.obsm[clones_key], 1) == 1):
            raise ValueError("The tree fitting method 'non-nested clones' assumes each cell is a member of exactly one clone. This is not the case for your data.")

        fitted_tree = inf.make_tree_from_nonnested_clones(adata.obsm[clones_key], time)

    elif method == "clones":
        if clone_times is None:
            raise ValueError("clone_times must be specified in order to fit a tree to nested clones.")
        clone_times = np.array(clone_times) # allowing clone_times to be passed as a raw list without causing errors later
        fitted_tree = inf.make_tree_from_clones(adata.obsm[clones_key], time, clone_times)
    else:
        raise ValueError("'" + method + "' is not an available method for fitting trees.")

    return fitted_tree


def read_newick(filename, leaf_labels, leaf_time = None):
    """
    Loads a tree saved in Newick format and adds annotations required for LineageOT.
    
    Parameters
    ----------
    filename : str
        The name of the file to load from.
    leaf_labels : list
        The label of each leaf in the Newick tree, sorted to align with the gene expression AnnData object filtered to cells at the corresponding time.
    leaf_time : float (default None)
        The time of sampling of the leaves. If unspecified, the root of the tree is assigned time 0.
    Returns
    -------
    tree : Networkx DiGraph
        The saved tree, in LineageOT's format.
        Each node is annotated with 'time_to_parent' and 'time' (which indicates either the time of sampling (for observed cells) or the time of division (for unobserved ancestors)). 
        Edges are directed from parent to child and are annotated with 'time' equal to the child node's 'time_to_parent'.
        Observed node indices correspond to their index in leaf_labels, which should match their row in the gene expression AnnData object filtered to cells at the corresponding time.
    """
    newick_tree = newick.read(filename)[0]
    tree = inf.convert_newick_to_networkx(newick_tree, leaf_labels, leaf_time)
    return tree



def fit_lineage_coupling(adata, time_1, time_2, lineage_tree_t2, time_key = 'time', state_key = None, epsilon = 0.05, normalize_cost = True, ot_method = 'sinkhorn', marginal_1 = [], marginal_2 = [], balance_reg = np.inf):
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
    ot_method : str (default 'sinkhorn')
        Method used for the optimal transport solver. 
        Choose from 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' and 'sinkhorn_epsilon_scaling' for balanced transport
        and 'sinkhorn', 'sinkhorn_stabilized', and 'sinkhorn_reg_scaling' for unbalanced transport.
        'sinkhorn' is recommended unless you encounter numerical problems.
        See PythonOT docs for more details.
    marginal_1 : Vector (default [])
        Marginal distribution (relative growth rates) for cells at time 1. If empty, assumed uniform.
    marginal_2 : Vector (default [])
        Marginal distribution (relative growth rates) for cells at time 2. If empty, assumed uniform.
    balance_reg : Number
        Regularization parameter for unbalanced transport. Smaller values allow more flexibility in growth rates. If infinite, marginals are treated as hard constraints.
    
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
    inf.add_nodes_at_time(lineage_tree_t2, time_1)

    observed_nodes = [n for n in inf.get_leaves(lineage_tree_t2, include_root = False)]

    # Split tree into components that share information
    components = inf.get_components(lineage_tree_t2)
    # Add annotations for each component separately
    for comp in components:
        inf.add_conditional_means_and_variances(comp, observed_nodes)

    # collect predicted ancestral states
    ancestor_info = inf.get_ancestor_data(lineage_tree_t2, time_1)

    # change backend of ancestor_info to AnnData ArrayView
    # to match state_arrays['early'], because POT v0.8 requires
    # matching backends
    ancestor_states = anndata.AnnData(X = ancestor_info[0])[:,:].X

    # compute cost matrix
    # (converted to numpy array to match default marginal backend)
    lineageOT_cost = np.array(ot.utils.dist(state_arrays['early'], ancestor_states)@np.diag(ancestor_info[1]**(-1)))

    if normalize_cost:
        lineageOT_cost = lineageOT_cost/np.median(lineageOT_cost)

    # fit coupling
    if balance_reg == np.inf:
        coupling_matrix = ot.sinkhorn(marginal_1, marginal_2, lineageOT_cost, epsilon, method = ot_method)
    else:
        coupling_matrix = ot.unbalanced.sinkhorn_unbalanced(marginal_1, marginal_2, lineageOT_cost, epsilon, balance_reg, method = ot_method)


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
