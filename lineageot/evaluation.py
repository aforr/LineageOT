# Evaluating fitted couplings
# Also includes some utility functinos for manipulating and plotting couplings


import numpy as np
import ot
import warnings
import matplotlib.pyplot as plt

def l2_difference(coupling_1, coupling_2):
    return np.linalg.norm(coupling_1 - coupling_2, 'fro')

def scaled_l2_difference(coupling_1, coupling_2):
    return l2_difference(coupling_1, coupling_2)/np.linalg.norm(coupling_2, 'fro')


def pairwise_squared_distances(data):
    """
    Returns the pairwise squared distances between rows of the data matrix
    """
    n = data.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d = data[i,:]-data[j,:]
            distances[i,j] = d@d
            distances[j,i] = distances[i,j]

    return distances
            


def coupling_to_coupling_cost_matrix(source, target):
    """
    Returns the (n_source*n_target)*(n_source*n_target) cost matrix
    for a W2 distance between two couplings of source and target

    Source and target here are just expression samples, without barcodes
    """
    n_source = source.shape[0]
    n_target = target.shape[0]

    cost_matrix = np.zeros((n_source*n_target, n_source*n_target))

    source_distances = pairwise_squared_distances(source)
    target_distances = pairwise_squared_distances(target)

    #UPDATE: vectorize
    for source_1 in range(n_source):
        for target_1 in range(n_target):
            index_1 = source_1*n_target + target_1
            for source_2 in range(source_1, n_source):
                for target_2 in range(n_target):
                    index_2 = source_2*n_target + target_2
                    
                    cost_matrix[index_1, index_2] = (source_distances[source_1, source_2]
                                                   + target_distances[target_1, target_2])
                    cost_matrix[index_2, index_1] = cost_matrix[index_1, index_2]
            
    return cost_matrix


def coupling_W2(coupling_1, coupling_2, source, target, epsilon):
    """
    Returns the entropically-regularized W2 distance between two couplings
    """
    cost_matrix = coupling_to_coupling_cost_matrix(source, target)
    return ot.sinkhorn2(coupling_1.flatten(), coupling_2.flatten(), cost_matrix, epsilon)








def squeeze_coupling_by_late_cluster(c, index):
    c1 = np.sum(c[:,  index], 1)
    c2 = np.sum(c[:, ~index], 1)
    return np.array([c1, c2]).T

def squeeze_coupling(c, row_cluster_labels = None, column_cluster_labels = None):
    if row_cluster_labels is None:
        row_cluster_labels = range(c.shape[0])
    if column_cluster_labels is None:
        column_cluster_labels = range(c.shape[1])
    row_clusters = np.unique(row_cluster_labels)
    num_row_clusters = len(row_clusters)
    column_clusters = np.unique(column_cluster_labels)
    num_column_clusters = len(column_clusters)
    
    squeezed_c = np.zeros([num_row_clusters, num_column_clusters])
    # if you have lots of clusters find a way to do this loop in numpy
    for i in range(num_row_clusters):
        for j in range(num_column_clusters):
            squeezed_c[i,j] = np.sum(c[np.ix_(row_cluster_labels == row_clusters[i],
                                              column_cluster_labels == column_clusters[j])])
    return squeezed_c

def tv(coupling1, coupling2):
    return np.linalg.norm((coupling1 - coupling2).flatten(), 1)/2

def normalize_columns(coupling):
    s = np.sum(coupling, 0)
    n = coupling.shape[1]
    return coupling*(s**(-1))/n



def expand_coupling_independent(c, true_coupling):
    s = np.sum(true_coupling, 1)
    conditional_coupling = true_coupling.T*(s**(-1))
    return conditional_coupling@c

def expand_coupling(c, true_coupling, distances, matched_dim = 0, max_dims_used = np.inf, xs_used = None):
    """
    Parameters
    ----------
    c : ndarray, shape (nx, ny) if matched_dim == 0, (ny, nx) if matched_dim == 1
        Coupling between source x and target y
    true_coupling : ndarray, shape (nx, nz) if matched_dim == 0, (nz, nx) if matched_dim == 1
        Reference coupling between x and z
    distances : ndarray, shape (nz, ny)
        Pairwise distances between z and y
    matched_dim : int
        Dimension in which c and true coupling
    max_dims_used: int or np.inf
        Set a finite value here to do an approximate calculation based on min(nx, max_dims_used) elements of x
    xs_used : list or None
        Indices of matched_dim to use in approximate calculation. If None and max_dims_used<nx, indices are randomly selected.
    
    Returns
    -------
    expanded_coupling : ndarray, shape same as true_couplings
        Optimal coupling between z and y consistent with the coupling c
    """
    if matched_dim == 1:
        c = c.T
        true_coupling = true_coupling.T
        
    expanded_coupling = np.zeros([true_coupling.shape[1], c.shape[1]])
    
    n_x = c.shape[0]
    x_marginal = np.sum(c, 1)
    if max_dims_used < x_marginal.size and xs_used is None:
        xs_used = np.random.choice(range(x_marginal.size), size = max_dims_used, p = x_marginal, replace = False)

    if not xs_used is None:
        xs_not_used = [i for i in range(x_marginal.size) if not i in xs_used]
        x_marginal[xs_not_used] = 0
        x_marginal = x_marginal/np.sum(x_marginal)
    for i in range(n_x):
        p_true = true_coupling[i, :]/np.sum(true_coupling[i, :])
        p_c = c[i, :]/np.sum(c[i, :])
        if x_marginal[i] > 0:
            expanded_coupling = expanded_coupling + ot.emd(p_true, p_c, distances)*x_marginal[i]
    
    if abs(np.sum(expanded_coupling) - 1) > 10**(-8):
        warnings.warn(("Expanded coupling not computed correctly. " +
                       "Check for infeasibility in row and column OT calculations. Updating the python optimal transport package may help (https://github.com/PythonOT/POT/issues/93).\n" +
                       "If total mass - 1 is small, this may not significantly affect downstream results.\n" +
                       "Total mass - 1: " +
                       str(np.sum(expanded_coupling) -1)))

    if matched_dim == 1:
        return expanded_coupling.T
    else:
        return expanded_coupling



def sample_indices_from_coupling(c, num_samples=None, return_all = False, thr = 10**(-6)):
    """
    Generates [row, column] samples from the coupling c

    If return_all is True, then returns all indices with coupling values above the threshold
    """

    if return_all:
        return [[i, j] for i in range(c.shape[0]) for j in range(c.shape[1]) if c[i,j] > thr]
    else:
        linear_samples = np.random.choice(range(c.size), size = num_samples, p = c.flat)
        
        if num_samples in [1, None]:
            linear_samples = [linear_samples]
            
        n_cols = c.shape[1]
        return np.array([[s//n_cols  , s%n_cols ] for s in linear_samples])



def sample_coordinates_from_coupling(c, row_points, column_points, num_samples=None, return_all = False, thr = 10**(-6)):
    """
    Generates [x, y] samples from the coupling c.

    If return_all is True, returns [x,y] coordinates of every pair with coupling value >thr
    """

    index_samples = sample_indices_from_coupling(c, num_samples = num_samples, return_all = return_all, thr = thr)

    return np.array([ [row_points[s[0], :], column_points[s[1],:]] for s in index_samples])


def sample_interpolant(coupling, row_points, column_points, t=0.5, num_samples=None, return_all=False, thr=10**(-6)):
    """
    Samples from the interpolated distribution implied by the coupling

    If return_all is True, returns the interpolants between every pair with coupling value >thr. This is the exact interpolant
    distribution if and only if all nonzero values of the coupling are identical and >thr.
    """

    # Sample start and end points
    coordinates = sample_coordinates_from_coupling(coupling,
                                                   row_points,
                                                   column_points,
                                                   num_samples = num_samples,
                                                   return_all = return_all,
                                                   thr = thr)

    # Average start and end points, weighted by t
    return np.array([(1-t)*pair[0] + t*pair[1] for pair in coordinates])



def plot2D_samples_mat(xs, xt, G, thr=1e-8, alpha_scale = 1, **kwargs):
    """ Plot matrix M  in 2D with  lines using alpha values

    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.

    Copied function from PythonOT and added alpha_scale parameter


    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    b : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        parameters given to the plot functions (default color is black if
        nothing given)
    """
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                plt.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        alpha=alpha_scale*G[i, j] / mx, **kwargs)


def print_metrics(couplings, cost_func, cost_func_name, log = False):
    """
    Prints cost_func evaluated for each coupling in the dictionary couplings
    """
    
    l = max([len(c) for c in couplings.keys()])
    print(cost_func_name)
    for c in couplings.keys():
        loss = cost_func(couplings[c])
        if log:
            loss = np.log(loss)
        print(c.ljust(l), ": ", "{:.3f}".format(loss))
    print("\n")
    return


def plot_metrics(couplings, cost_func, cost_func_name, epsilons, log = False, points=False, scale=1.0, label_font_size=18, tick_font_size=12):
    """
    Plots cost_func evaluated as a function of epsilon
    """
    zero_offset = epsilons[0]/2
    all_ys = []
    if "lineageOT" in couplings.keys():
        ys = np.array([cost_func(c) for c in [couplings['lineage entropic rna ' + str(e)] for e in epsilons]])
        plt.plot(epsilons, ys/scale, label = "LineageOT, true tree")
        if points:
            plt.scatter([zero_offset], [cost_func(couplings["lineageOT"])/scale])
        all_ys.append(ys)
    if "OT" in couplings.keys():
        ys = np.array([cost_func(c) for c in [couplings['entropic rna ' + str(e)] for e in epsilons]])
        plt.plot(epsilons, ys/scale, label = "Entropic OT")
        if points:
            plt.scatter([zero_offset], [cost_func(couplings["OT"])/scale])
        all_ys.append(ys)
    if "lineageOT, fitted" in couplings.keys():
        ys = np.array([cost_func(c) for c in [couplings['fitted lineage rna ' + str(e)] for e in epsilons]])
        plt.plot(epsilons, ys/scale, label = "LineageOT, fitted tree")
        if points:
            plt.scatter([zero_offset], [cost_func(couplings["lineageOT, fitted"])/scale])
        all_ys.append(ys)

    plt.ylabel(cost_func_name, fontsize=label_font_size)
    plt.xlabel("Entropy parameter", fontsize=label_font_size)
    plt.xscale("log")

    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    if points:
        plt.xlim([0.9*zero_offset, epsilons[-1]])
    else:
        plt.xlim([epsilons[0], epsilons[-1]])

    ylims = plt.ylim([0, None])
    # upper limit should be at least 1
    plt.ylim([0, max(ylims[1], 1)])

    plt.legend(fontsize=tick_font_size)
    return all_ys
