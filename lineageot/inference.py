# Functions for handling trees and coupling inference

import copy
import ete3
import networkx as nx
import newick
import numpy as np
import warnings
from cvxopt.solvers import qp as cvxopt_qp
from cvxopt import matrix as cvxopt_matrix
from numbers import Number

import lineageot.simulation as sim





###############################
# General inference functions #
###############################

def rate_estimator(barcode_array, time):
    """
    Estimates the mutation rate based on the number of unmutated
    barcodes remaining.
    """
    num_nonzeros = np.count_nonzero(barcode_array)
    num_zeros = barcode_array.size - num_nonzeros
    return -np.log(num_zeros/barcode_array.size)/time


def scaled_Hamming_distance(barcode1, barcode2):
    """
    Computes the distance between two barcodes, adjusted for

    (1) the number of sites where both cells were measured and

    (2) distance between two scars is twice the distance from
                            
        scarred to unscarred
    """
    shared_indices = (barcode1 >= 0) & (barcode2 >= 0)
    b1 = barcode1[shared_indices]

    # There may not be any sites where both were measured
    if len(b1) == 0:
        return np.nan
    b2 = barcode2[shared_indices]

    differences = b1 != b2
    double_scars = differences & (b1 != 0) & (b2 != 0)


    return (np.sum(differences) + np.sum(double_scars))/len(b1)

def barcode_distances(barcode_array):
    """
    Computes all pairwise lineage distances between barcodes
    """
    num_cells = barcode_array.shape[0]
    lineage_distances = np.zeros([num_cells, num_cells])

    for i in range(num_cells):
        barcode1 = barcode_array[i,:]
        for j in range(i):
            barcode2 = barcode_array[j]

            lineage_distances[i,j] = scaled_Hamming_distance(barcode1, barcode2)
            lineage_distances[j,i] = lineage_distances[i,j]
    return lineage_distances


def OT_cost(coupling, cost):
    return np.tensordot(coupling, cost, 2)


##########################
# Tree-related functions #
##########################

def list_tree_to_digraph(list_tree):
    """
    Converts a tree stored as nested lists to a networkx DiGraph

    Internal nodes are indexed by negative integers, leaves by nonnegative integers
    """
    next_internal_node = -1
    next_leaf_node = 0
    T, next_internal_node, next_leaf_node, subtree_root = recursive_list_tree_to_digraph(list_tree, 
                                                                                         next_internal_node, 
                                                                                         next_leaf_node)

    barcode_length = len(T.nodes[0]['cell'].barcode)
    T.add_node('root', cell=sim.Cell(np.nan, np.zeros(barcode_length)), time_to_parent = 0)
    T.add_edge('root', subtree_root, time=T.nodes[subtree_root]['time_to_parent'])

    return T

    



def recursive_list_tree_to_digraph(list_tree, next_internal_node, next_leaf_node):
    """
    Recursive helper function for list_tree_to_digraph

    Returns (current_tree, next_internal_node_label, root_of_current_tree)
    """


    if len(list_tree) == 1:
        assert(not isinstance(list_tree[0], list))
        T = nx.DiGraph()
        T.add_node(next_leaf_node, cell=list_tree[0][0], time_to_parent=list_tree[0][1])
        return T, next_internal_node, next_leaf_node + 1, next_leaf_node
    else:
        assert(len(list_tree) == 3) # assuming binary tree: list has two leaves and the cell at the current node
        left_subtree, next_internal_node, next_leaf_node, left_root = recursive_list_tree_to_digraph(list_tree[0],
                                                                                                     next_internal_node, 
                                                                                                     next_leaf_node)
        right_subtree, next_internal_node, next_leaf_node, right_root = recursive_list_tree_to_digraph(list_tree[1], 
                                                                                                       next_internal_node, 
                                                                                                       next_leaf_node)

        T = nx.compose(left_subtree, right_subtree)
        T.add_node(next_internal_node, cell=list_tree[2][0], time_to_parent=list_tree[2][1])
        T.add_edge(next_internal_node, left_root, time=T.nodes[left_root]['time_to_parent'])
        T.add_edge(next_internal_node, right_root, time=T.nodes[right_root]['time_to_parent'])

        return T, next_internal_node - 1, next_leaf_node, next_internal_node
        

















def annotate_tree(tree, mutation_rates, time_inference_method = 'independent', overwrite_times = False):
    """
    Adds barcodes and times to internal (ancestor) nodes so likelihoods can be computed

    Barcodes are inferred by putting minimizing the number of mutation events required,
    assuming a model with no back mutations and a known initial barcode
    """

    recursive_add_barcodes(tree, 'root')
    add_times(tree, mutation_rates, time_inference_method, overwrite = overwrite_times)
    add_times_to_edges(tree)
    return tree

def add_leaf_barcodes(tree, barcode_array):
    """
    Adds barcodes from barcode_array to the corresponding leaves of the tree
    """
    
    num_cells = barcode_array.shape[0]
    for cell in range(num_cells):
        if 'cell' in tree.nodes[cell]:
            tree.nodes[cell]['cell'].barcode = barcode_array[cell, :]
        else:
            tree.nodes[cell]['cell'] = sim.Cell(np.nan, barcode_array[cell, :])
    tree.nodes['root']['cell'] = sim.Cell(np.nan, np.zeros(barcode_array.shape[1]))
    return tree

def add_leaf_x(tree, x_array):
    """
    Adds expression vectors from x_array to the corresponding leaves of the tree
    """
    num_cells = x_array.shape[0]
    for cell in range(num_cells):
        if 'cell' in tree.nodes[cell]:
            tree.nodes[cell]['cell'].x = x_array[cell, :]
        else:
            tree.nodes[cell]['cell'] = sim.Cell(x_array[cell, :], np.nan)
    return tree

    


def recursive_add_barcodes(tree, current_node):
    """
    Fills in the barcodes for internal nodes for a tree whose leaves have barcodes

    Minimizes the number of mutation events that occur, assuming no backmutations
    and a known initial barcode
    """


    children = tree.successors(current_node)
    child_barcodes = []
    for child in children:
        recursive_add_barcodes(tree, child)
        child_barcodes.append(tree.nodes[child]['cell'].barcode)
    if 'cell' in tree.nodes[current_node]:
        return
    else:
        # if we've somehow ended up at a leaf with no cell, this will fail
        # likely this means that the tree was improperly annotated
        # (e.g., too few barcodes were added)
        current_barcode = child_barcodes[0].copy()
        if len(child_barcodes) == 1:
            warnings.warn("Some cell has only one descendant. Assuming no mutations occurred between them.")
        else:
            # assume that if a site is missing in one child it equals the
            # value in the other child
            current_barcode[child_barcodes[0] == -1] = child_barcodes[1][child_barcodes[0] == -1]
            # if the two children differ and neither is missing, set parent to unmutated
            current_barcode[(child_barcodes[0] != child_barcodes[1]) &
                            (child_barcodes[1] != -1)] = 0
        
        tree.nodes[current_node]['cell'] = sim.Cell(np.nan, current_barcode)
        return



def add_times(tree, mutation_rates, time_inference_method, overwrite = False):
    """
    Adds estimated division times/edge lengths to a tree

    The tree should already have all node barcodes estimated
    """
    for node in tree.nodes:
        if node == 'root':
            tree.nodes[node]['ml_time_to_parent'] = 0
        else:
            tree.nodes[node]['ml_time_to_parent'] = estimate_division_time(tree.nodes[node],
                                                                        tree.nodes[next(tree.predecessors(node))],
                                                                        mutation_rates
                                                                           )
    if time_inference_method == 'independent':
        # Infer times for edges entirely independently of each other,
        # ignoring constraints (like that leaves were sampled at the
        # same time)
        for node in tree.nodes:
            if (not ('time_to_parent' in tree.nodes[node])) or overwrite:
                tree.nodes[node]['time_to_parent'] = tree.nodes[node]['ml_time_to_parent']

    elif time_inference_method == 'least_squares':
        # Finds estimate of internal node times that minimizes the
        # squared error of the implied division times (relative to
        # the ml_time_to_parent estimate
        internal_nodes = get_internal_nodes(tree)
        leaves = get_leaves(tree)

        final_time = tree.nodes[leaves[0]]['time']
        # root should have time zero
        assert(tree.nodes[leaves[-1]]['time'] == 0)

        laplacian = nx.linalg.laplacian_matrix(nx.Graph(tree),
                                               nodelist = internal_nodes + leaves)
        incidence_matrix = nx.linalg.incidence_matrix(tree,
                                                      nodelist = internal_nodes + leaves,
                                                      oriented = True)
        
        edge_target_times = np.array([tree.nodes[e[1]]['ml_time_to_parent'] for e in tree.edges()])
        leaf_and_root_times = np.ones(len(leaves))*final_time
        leaf_and_root_times[-1] = 0

        b = (incidence_matrix[range(len(internal_nodes)), :] @ edge_target_times
             - laplacian[np.ix_(range(len(internal_nodes)),
                                range(len(internal_nodes), 
                                      len(internal_nodes)+len(leaves))
                                )] @ leaf_and_root_times)




        laplacian = np.array(laplacian.todense().astype(np.double))
        incidence_matrix = np.array(incidence_matrix.todense().astype(np.double))

        P = laplacian[np.ix_(range(len(internal_nodes)), range(len(internal_nodes)))]
        G = -incidence_matrix[range(len(internal_nodes)),:].T
        h = incidence_matrix[range(len(internal_nodes), len(internal_nodes) + len(leaves))].T @ leaf_and_root_times

        # In the least-squares internal_node_times may not be consistent with the ordering given by the tree
        # The quadratic program enforces the constraint that all edges have positive time
        internal_node_times = np.reshape(cvxopt_qp_from_numpy(P, -b, G, h), (len(internal_nodes)))
        add_node_times_from_dict(tree,
                                 'root',
                                 {node:time for node, time in zip(internal_nodes, internal_node_times)})
            
        add_division_times_from_vertex_times(tree)
                                               
    else:
        raise NotImplementedError(time_inference_method + ' is not an accepted time inference method')
    return


def estimate_division_time(child, parent, mutation_rates):
    """
    Estimates the lifetime of child, i.e. the time between when parent
    divided to make child and when child divided

    Input arguments are nodes in a lineage tree, i.e. dicts
    """

    #TODO: is there a better estimator?
    c_barcode = child['cell'].barcode
    p_barcode = parent['cell'].barcode

    available_entries = (c_barcode != -1) & (p_barcode == 0)
    differing_entries = ((c_barcode != p_barcode) &
                         available_entries
                         )
    
    #TODO: handle possible division by zero
    return np.sum(differing_entries)/np.sum(mutation_rates[available_entries])
        


def add_division_times_from_vertex_times(tree, current_node = 'root'):
    """
    Adds 'time_to_parent' variables to nodes, based on 'time' annotations
    """
    
    for child in tree.successors(current_node):
        tree.nodes[child]['time_to_parent'] = (tree.nodes[child]['time']
                                               - tree.nodes[current_node]['time']
                                               )
        add_division_times_from_vertex_times(tree, current_node = child)

    return
    

def add_times_to_edges(tree):
    """
    Labels each edge of tree with 'time' taken from
    'time_to_parent' of its endpoint
    """
    for (u, v) in tree.edges:
        t = tree.nodes[v]['time_to_parent']
        if t > 0:
            tree.edges[u,v]['time'] = t
        else:
            warnings.warn('Nonpositive time to parent encountered: t = '
                          + str(t)
                          + ' for edge (' + str(u) + ', ' + str(v) + ').')
            tree.edges[u,v]['time'] = np.finfo(float).eps
    return tree

def add_inverse_times_to_edges(tree):
    """
    Labels each edge of the tree with 'inverse time' equal
    to 1/edge['time']
    """
    for (u, v) in tree.edges:
        tree.edges[u,v]['inverse time'] = 1/tree.edges[u,v]['time']
    return tree


def add_leaf_times(tree, final_time):
    """
    Adds the known final time to all leaves
    and 0 as the root time
    """
    leaves = get_leaves(tree)
    for leaf in leaves:
        if leaf == 'root':
            tree.nodes[leaf]['time'] = 0
            tree.nodes[leaf]['time_to_parent'] = 0
        else:
            tree.nodes[leaf]['time'] = final_time
    return

def add_node_times_from_dict(tree, current_node, time_dict):
    """
    Adds times from time_dict to current_node and its descendants
    """
    if current_node in time_dict:
        # enforce children not having earlier times than parents
        tree.nodes()[current_node]['time'] = np.maximum(time_dict[current_node],
                                                        tree.nodes()[next(tree.predecessors(current_node))]['time'])
    for child in tree.successors(current_node):
        add_node_times_from_dict(tree, child, time_dict)

    return

def add_node_times_from_division_times(tree, current_node = 'root', overwrite = False):
    """
    Adds 'time' variable to all descendants of current_node based on the 'time_to_parent' variable
    """
    if current_node == 'root':
        if overwrite | (not 'time' in tree.nodes[current_node]):
            tree.nodes[current_node]['time'] = 0
    else:
        parent = next(tree.predecessors(current_node))
        if overwrite | (not 'time' in tree.nodes[current_node]):
            tree.nodes[current_node]['time'] = tree.nodes[parent]['time'] + tree.nodes[current_node]['time_to_parent']

    children = tree.successors(current_node)
    for child in children:
        add_node_times_from_division_times(tree, child)

    return


def remove_times(tree):
    """
    Removes time annotations from nodes and edges of a tree
    """
    for node in tree.nodes:
        tree.nodes[node].pop('time_to_parent')
        if 'time' in tree.nodes[node]:
            tree.nodes[node].pop('time')
    for edge in tree.edges:
        tree.edges[edge].pop('time')
    return


def get_leaves(tree, include_root = True):
    """
    Returns a list of the leaf nodes of a tree
    including the root
    """
    leaves = [node for node in tree if tree.degree(node) <= 1]
    string_leaves = [l for l in leaves if (type(l) == str) and (l != 'root')]
    number_leaves = [l for l in leaves if isinstance(l, Number)]

    number_leaves.sort()
    string_leaves.sort()
    if include_root & ('root' in tree):
        string_leaves = string_leaves + ['root']

    return number_leaves + string_leaves


def get_internal_nodes(tree):
    """
    Returns a list of the non-leaf nodes of a tree
    """
    nodes = [node for node in tree if tree.degree(node) >= 2]
    nodes.sort()
    return nodes

def get_leaf_descendants(tree, node):
    """
    Returns a list of the leaf nodes of the tree that are
    descendants of node
    """
    if tree.out_degree(node) == 0:
        return [node]
    else:
        children = tree.successors(node)
        leaf_descendants = []
        for child in children:
            leaf_descendants = leaf_descendants + get_leaf_descendants(tree, child)
        return leaf_descendants
    return

def compute_leaf_times(tree, num_leaves):
    """
    Computes the list of times of the leaves by adding 'time_to_parent' along the path to 'root'
    """
    rna,  barcodes = extract_data_arrays(tree)
    times = np.zeros(barcodes.shape[0])
    for leaf in range(barcodes.shape[0]):
        current_node = leaf
        while not current_node == 'root':
            times[leaf] = times[leaf] + tree.nodes[current_node]['time_to_parent']
            current_node = next(tree.predecessors(current_node))

    return times

def extract_data_arrays(tree):
    """
    Returns arrays of the RNA expression and barcodes from leaves of the tree

    Each row of each array is a cell
    """

    leaves = get_leaves(tree, include_root = False)
    expressions = np.array([tree.nodes[leaf]['cell'].x for leaf in leaves])
    barcodes = np.array([tree.nodes[leaf]['cell'].barcode for leaf in leaves])
    return expressions, barcodes


def extract_ancestor_data_arrays(late_tree, time, params):
    """
    Returns arrays of the RNA expression and barcodes for ancestors of leaves of the tree
    
    Each row of each array is a leaf node
    """
    
    leaves = get_leaves(late_tree, include_root = False)


    early_tree = truncate_tree(late_tree, time, params)
    cells_early = get_leaves(early_tree, include_root = False)
    
    
    expressions = np.nan*np.ones([len(leaves), params.num_genes])
    barcodes = np.nan*np.ones([len(leaves), params.barcode_length])

    for cell in cells_early:
        parent = next(early_tree.predecessors(cell))
        late_tree_cell = None
        for child in late_tree.successors(parent):
            if late_tree.nodes[child]['cell'].seed == early_tree.nodes[cell]['cell'].seed:
                late_tree_cell = child
                break
        if late_tree_cell == None:
            raise ValueError("A leaf in early_tree does not appear in late_tree. Cannot find coupling." +
                             "\nCheck whether either tree has been modified since truncating.")

        descendants = get_leaf_descendants(late_tree, late_tree_cell)

        expressions[descendants, :] = early_tree.nodes[cell]['cell'].x
        barcodes[descendants, :] = early_tree.nodes[cell]['cell'].barcode

    return expressions, barcodes
    













# Truncating a tree at an earlier endpoint


def truncate_tree(tree, new_end_time, params, inplace = False, current_node = 'root', next_leaf_to_add = 0):
    """
    Removes all nodes at times greater than new_end_time
    and adds new leaves at exactly new_end_time

    params: simulation parameters used to create tree
    """
    if not inplace:
        tree = copy.deepcopy(tree)

    if not ('time' in tree.nodes['root']):
        add_node_times_from_division_times(tree)

    if tree.nodes[current_node]['time'] >= new_end_time:
        parent = next(tree.predecessors(current_node))

        initial_cell = tree.nodes[parent]['cell'].deepcopy()
        initial_cell.seed = tree.nodes[current_node]['cell'].seed

        new_cell = sim.evolve_cell(initial_cell, 
                               new_end_time - tree.nodes[parent]['time'],
                               params)

        remove_node_and_descendants(tree, current_node)
        
        tree.add_node(next_leaf_to_add)
        tree.nodes[next_leaf_to_add]['time'] = new_end_time
        tree.nodes[next_leaf_to_add]['time_to_parent'] = new_end_time - tree.nodes[parent]['time']
        tree.nodes[next_leaf_to_add]['cell'] = new_cell

        tree.add_edge(parent, next_leaf_to_add, time = new_end_time - tree.nodes[parent]['time'])
        next_leaf_to_add = next_leaf_to_add + 1
    else:
        # not just tree.successors(node) because that changes as nodes are removed
        children = [child for child in tree.successors(current_node)]
        for child in children:
            tree, next_leaf_to_add = truncate_tree(tree,
                                                   new_end_time,
                                                   params,
                                                   inplace = True,
                                                   current_node = child,
                                                   next_leaf_to_add = next_leaf_to_add)
    
    if inplace:
        # proxy for "if in recursive case"
        return tree, next_leaf_to_add
    else:
        return tree


def remove_node_and_descendants(tree, node):
    """
    Removes a node and all its descendants from the tree
    """

    # not just tree.successors(node) because that changes as
    # nodes are removed
    children = [child for child in tree.successors(node)]
    for child in children:
        remove_node_and_descendants(tree, child)

    parent = next(tree.predecessors(node))

    tree.remove_edge(parent, node)
    tree.remove_node(node)

    return tree




def resample_cells(tree, params, current_node = 'root', inplace = False):
    """
    Runs a new simulation of the cell evolution on a fixed tree
    """
    if not inplace:
        tree = copy.deepcopy(tree)

    for child in tree.successors(current_node):
        initial_cell = tree.nodes[current_node]['cell'].deepcopy()
        initial_cell.reset_seed()
        
        tree.nodes[child]['cell'] = sim.evolve_cell(initial_cell,
                                               tree.nodes[child]['time_to_parent'],
                                               params)
        resample_cells(tree, params, current_node = child, inplace = True)
        
    return tree


def get_true_coupling(early_tree, late_tree):
    """
    Returns the coupling between leaves of early_tree and their descendants in
    late_tree. Assumes that early_tree is a truncated version of late_tree

    The marginal over the early cells is uniform; if cells have different
    numbers of descendants, the marginal over late cells will not be uniform.
    """
    num_cells_early = len(get_leaves(early_tree)) - 1
    num_cells_late = len(get_leaves(late_tree)) - 1
    
    coupling = np.zeros([num_cells_early, num_cells_late])
    
    cells_early = get_leaves(early_tree, include_root = False)
    
    
    for cell in cells_early:
        parent = next(early_tree.predecessors(cell))
        late_tree_cell = None
        for child in late_tree.successors(parent):
            if late_tree.nodes[child]['cell'].seed == early_tree.nodes[cell]['cell'].seed:
                late_tree_cell = child
                break
        if late_tree_cell == None:
            raise ValueError("A leaf in early_tree does not appear in late_tree. Cannot find coupling." +
                             "\nCheck whether either tree has been modified since truncating.")
        descendants = get_leaf_descendants(late_tree, late_tree_cell)
        coupling[cell, descendants] = 1/(num_cells_early*len(descendants))
    
    return coupling
    
    
    
def get_lineage_distances_across_time(early_tree, late_tree):
    """                                                                                                                                                                                                     
    Returns the matrix of lineage distances between leaves of early_tree and leaves in                                                                                                                              
    late_tree. Assumes that early_tree is a truncated version of late_tree                                                                                                                                  
    """
    num_cells_early = len(get_leaves(early_tree)) - 1
    num_cells_late = len(get_leaves(late_tree)) - 1

    d = np.zeros([num_cells_early, num_cells_late])

    cells_early = get_leaves(early_tree, include_root = False)
    cells_late = get_leaves(late_tree, include_root = False)


    # get length of path up to parent of early_cell and back to early_cell
    for late_cell in cells_late:
        distance_dictionary, tmp = nx.single_source_dijkstra(nx.Graph(late_tree),
                                                            late_cell,
                                                            weight = 'time')
        for early_cell in cells_early:
            d[early_cell, late_cell] = (distance_dictionary[next(early_tree.predecessors(early_cell))]
                                        + early_tree.nodes[early_cell]['time_to_parent'])

    # correct distances to descendants
    for early_cell in cells_early:
        parent = next(early_tree.predecessors(early_cell))
        late_cell = None
        for child in late_tree.successors(parent):
            if late_tree.nodes[child]['cell'].seed == early_tree.nodes[early_cell]['cell'].seed:
                late_cell = child
                break
        if late_cell is not None:
            descendants = get_leaf_descendants(late_tree, late_cell)
            d[early_cell, descendants] = (d[early_cell, descendants]
                                          - 2*early_tree.nodes[early_cell]['time_to_parent'])

    return d

























# For OT using the lineage information, we want to have nodes for ancestors at the early sampling time

def add_nodes_at_time(tree, time_to_add, current_node = 'root', num_nodes_added = 0):
    """
    Splits every edge (u,v) where u['time'] < time_to_add < v['time']

    into (u, w) and (w, v) with w['time'] = time_to_add

    Newly added nodes {w} are labeled as tuples (time_to_add, i)

    The input tree should be annotated with node times already
    """

    if tree.nodes[current_node]['time'] == time_to_add:
        # Do not add a new node if we already have one at the correct time
        return num_nodes_added
    else:
        current_children = [child for child in tree.successors(current_node)]
        # Not iterating over edges directly because we're adding edges along the way
        for child in current_children:
            if tree.nodes[child]['time'] > time_to_add:
                split_edge(tree, (current_node, child), (time_to_add, num_nodes_added))
                
                # Add annotations and correct old ones
                tree.nodes[(time_to_add, num_nodes_added)]['time'] = time_to_add
                tree.nodes[(time_to_add, num_nodes_added)]['time_to_parent'] = time_to_add - tree.nodes[current_node]['time']
                tree.nodes[child]['time_to_parent'] = tree.nodes[child]['time'] - time_to_add
                tree.edges[current_node, (time_to_add, num_nodes_added)]['time'] = time_to_add - tree.nodes[current_node]['time']
                tree.edges[(time_to_add, num_nodes_added), child]['time'] = tree.nodes[child]['time'] - time_to_add
                
                num_nodes_added = num_nodes_added + 1
            elif tree.nodes[child]['time'] <= time_to_add:
                num_nodes_added = add_nodes_at_time(tree,
                                                    time_to_add,
                                                    current_node = child,
                                                    num_nodes_added = num_nodes_added)

        return num_nodes_added



def split_edge(tree, edge, new_node):
    tree.remove_edge(edge[0], edge[1])
    tree.add_node(new_node)
    tree.add_edge(edge[0], new_node)
    tree.add_edge(new_node, edge[1])
    return




def get_components(graph, edge_length_key = 'time'):
    """
    Returns subgraph views corresponding to connected components of the graph
    if edges of infinite length are removed
    
    Parameters
    ----------
    graph: NetworkX graph
    edge_length_key: default 'time'

    Returns
    -------
    subgraphs: List of NetworkX subgraph views
    """
    # copying graph as undirected
    g = nx.Graph(graph)
    
    edges_to_remove = [e for e in g.edges if g.edges[e][edge_length_key] == np.inf]
    g.remove_edges_from(edges_to_remove)
    
    component_node_sets = nx.connected_components(g)

    return [graph.subgraph(component_nodes) for component_nodes in component_node_sets] 












def add_conditional_means_and_variances(tree, observed_nodes):
    """
    Adds the mean and variance of the posterior on 'x' for each of the unobserved
    nodes, conditional on the observed values of 'x' in observed_nodes,
    assuming that differences along edges are Gaussian with variance equal to 
    the length of the edge.

    In doing so, also adds inverse time annotations to edges.

    If no nodes in tree are observed, inverse time annotations are added but
    conditional means and variances are not (as there is nothing to condition on).
    """
    node_list = [n for n in tree.nodes]
    
    add_inverse_times_to_edges(tree)
    l = nx.laplacian_matrix(nx.Graph(tree), nodelist = node_list, weight = 'inverse time')
    
    unobserved_nodes = [n for n in node_list if not n in observed_nodes]
    # Resorting so the order of indices in all matrices match
    # and removing observed_nodes not in tree
    observed_nodes = [n for n in node_list if n in observed_nodes]
    if len(observed_nodes) == 0:
        # no observed nodes in this component
        return
    unobserved_node_indices = [i for i in range(len(node_list)) if not node_list[i] in observed_nodes]
    observed_node_indices = [i for i in range(len(node_list)) if node_list[i] in observed_nodes]
    
    conditioned_precision = np.array(l[np.ix_(unobserved_node_indices, unobserved_node_indices)].todense())
    conditioned_covariance = np.linalg.inv(conditioned_precision)
    
    x_differences = np.array([tree.nodes[n]['cell'].x - tree.nodes[observed_nodes[0]]['cell'].x
                              for n in observed_nodes])
    
    conditioned_means = -1*(conditioned_covariance@l[np.ix_(unobserved_node_indices, observed_node_indices)]@x_differences)
    
    for n in observed_nodes:
        tree.nodes[n]['x mean'] = tree.nodes[n]['cell'].x
        tree.nodes[n]['x variance'] = 0
        
    for i, n in zip(range(len(unobserved_nodes)), unobserved_nodes):
        tree.nodes[n]['x mean'] = conditioned_means[i, :] + tree.nodes[observed_nodes[0]]['cell'].x
        tree.nodes[n]['x variance'] = conditioned_covariance[i, i] # assumed isotropic
    
    return
        


def get_ancestor_data(tree, time, leaf = None):
    if leaf == None:
        # get all leaf ancestors
        leaves = [l for l in get_leaves(tree) if not l == 'root']
        data = [get_ancestor_data(tree, time, l) for l in leaves]
        return np.array([d[0] for d in data]), np.array([d[1] for d in data])
    else:
        current_node = leaf
        while tree.nodes[current_node]['time'] > time:
            current_node = next(tree.predecessors(current_node))
        if tree.nodes[current_node]['time'] < time:
            error = "Tree has no ancestor of cell " + str(leaf) + " at time " + str(time)
            raise ValueError(error)
        return (tree.nodes[current_node]['x mean'], tree.nodes[current_node]['x variance'])
    return









##########################
# Tree fitting functions #
##########################

def make_tree_from_nonnested_clones(clone_matrix, time, root_time_factor = 1000):
    """
    Creates a forest of stars from clonally-labeled data. The centers of the stars are connected to a root far in the past.

    Parameters
    ----------
    clone_matrix: Boolean array with shape [num_cells, num_clones]
        Each entry is 1 if the corresponding cell belongs to the corresponding clone and zero otherwise.
        Each cell should belong to exactly one clone.
    time: Number
        The time of sampling of cells relative to initial clonal labelling.
    root_time_factor: Number, default 1000
        Relative time to root of tree (i.e., most recent common ancestor of all cells).
        The time of the root is set to -root_time_factor*time.
        The default is large so minimal information is shared across clones.
    Returns
    -------
    fitted_tree: NetworkX DiGraph
        A tree annotated with edge and node times
    """
    fitted_tree = nx.DiGraph()

    n_cells, n_clones = clone_matrix.shape
    clone_labels = [-i-1 for i in range(n_clones) if np.sum(clone_matrix[:, i]) > 0] # negative so distinct from cells, skipping empty clones
    
    # add nodes for cells
    fitted_tree.add_nodes_from(range(n_cells), time = time, time_to_parent = time)

    # add nodes for clone progenitors
    fitted_tree.add_nodes_from(clone_labels, time = 0, time_to_parent = root_time_factor*time)

    # add edges from clone progenitors to observed cells
    for clone in clone_labels:
        cells_in_clone = np.nonzero(clone_matrix[:, -clone - 1])[0]
        fitted_tree.add_edges_from([(clone, cell, {'time': time}) for cell in cells_in_clone])

    # add root
    fitted_tree.add_node('root', time = -root_time_factor*time)

    # add edges to root
    fitted_tree.add_edges_from([('root', i, {'time': root_time_factor*time}) for i in clone_labels])

    return fitted_tree


def make_tree_from_clones(clone_matrix, time, clone_times, root_time = -np.inf):
    """
    Adds a leaf for each row in clone_matrix to clone_reference_tree. The parent is set as the
    clone that the cell is a member of with the latest labeling time.

    clone_reference_tree is edited in place rather than returned.

    Parameters
    ----------
    clone_matrix: Boolean array with shape [num_cells, num_clones]
        Each entry is 1 if the corresponding cell belongs to the corresponding clone and zero otherwise.
    clone_times: Vector of length num_clones
        Each entry has the time of labeling of the corresponding clone.
    time: Number
        The time of sampling of cells.
    root_time: Number, default -np.inf
        The time of the most recent common ancestor of all clones. 
        If -np.inf, clones are effectively treated as unrelated

    Returns
    -------
    fitted_tree: NetworkX DiGraph
        A tree annotated with edge and node times
    """
    fitted_tree = make_clone_reference_tree(clone_matrix, clone_times, root_time)
    add_samples_to_clone_tree(clone_matrix, clone_times, fitted_tree, time)
    return fitted_tree


def add_samples_to_clone_tree(clone_matrix, clone_times, clone_reference_tree, sampling_time):
    """
    Adds a leaf for each row in clone_matrix to clone_reference_tree. The parent is set as the
    clone that the cell is a member of with the latest labeling time.

    clone_reference_tree is edited in place rather than returned.

    Parameters
    ----------
    clone_matrix: Boolean array with shape [num_cells, num_clones]
        Each entry is 1 if the corresponding cell belongs to the corresponding clone and zero otherwise.
    clone_times: Vector of length num_clones
        Each entry has the time of labeling of the corresponding clone.
    clone_reference_tree: 
        The tree of lineage relationships among clones.
    sampling_time: Number
        The time of sampling of the cells. Should be greater than all clone labeling times.

    Returns
    -------
    """
    n_cells, n_clones = clone_matrix.shape
    for cell in range(n_cells):
        parent_index = get_parent_clone_of_leaf(cell, clone_matrix, clone_times)
        parent_label = 'clone_' + str(parent_index)
        edge_time = sampling_time - clone_reference_tree.nodes[parent_label]['time']
        clone_reference_tree.add_node(cell, time = sampling_time, time_to_parent = edge_time)
        clone_reference_tree.add_edge(parent_label, cell, time = edge_time)
    return


def get_parent_clone_of_leaf(leaf, clone_matrix, clone_times):
    """
    Returns the index of the clone that the leaf is a member of with the latest labeling time.
    """
    candidate_clones = np.nonzero(clone_matrix[leaf, :])[0]
    candidate_times = clone_times[candidate_clones]
    maximal_time_indices = np.nonzero(candidate_times == np.max(candidate_times))[0]

    if len(maximal_time_indices) != 1:
        raise ValueError("Could not unambiguously assign a sampled cell to a parent clone.")
    else:
        return candidate_clones[maximal_time_indices[0]]


def make_clone_reference_tree(clone_matrix, clone_times, root_time = -np.inf):
    """
    Makes a tree with nodes for each clone.
    
    Parameters
    ----------
    clone_matrix: Boolean array with shape [num_cells, num_clones]
        Each entry is 1 if the corresponding cell belongs to the corresponding clone and zero otherwise.
    clone_times: Vector of length num_clones
        Each entry has the time of labeling of the corresponding clone.
    root_time: Number, default -np.inf
        The time of the most recent common ancestor of all clones. 
        If -np.inf, clone subtrees are effectively treated independently.

    Returns
    -------
    clone_reference_tree: NetworkX DiGraph
        A tree of clones (not sampled cells), annotated with edge and node times
    """

    n_cells, n_clones = clone_matrix.shape
    unique_times = np.unique(clone_times) # output is sorted
    clone_labels = ['clone_' + str(i) for i in range(n_clones)]

    # make base level of tree
    clone_reference_tree = nx.DiGraph()
    clone_reference_tree.add_node('root', time = root_time)

    current_clone_indices = np.nonzero(clone_times == unique_times[0])[0]
    current_clone_labels = [clone_labels[i] for i in current_clone_indices]
    clone_reference_tree.add_nodes_from(current_clone_labels, time = unique_times[0], time_to_parent = unique_times[0] - root_time)
    clone_reference_tree.add_edges_from([('root', label, {'time': unique_times[0] - root_time}) for label in current_clone_labels])

    # make subsequent levels
    for level, labeling_time in zip(range(1, len(unique_times)), unique_times[1:]):
        current_clone_indices = np.nonzero(clone_times == labeling_time)[0]
        current_clone_labels = [clone_labels[i] for i in current_clone_indices]

        clone_reference_tree.add_nodes_from(current_clone_labels, time = labeling_time)

        for clone_index in current_clone_indices:
            parent_index = find_parent_clone(clone_index, clone_matrix, clone_times)
            if parent_index == 'root':
                parent_label = 'root'
            else:
                parent_label = clone_labels[parent_index]

            time_to_parent = labeling_time - clone_reference_tree.nodes[parent_label]['time']
            
            clone_reference_tree.nodes[clone_labels[clone_index]]['time_to_parent'] = time_to_parent
            clone_reference_tree.add_edge(parent_label, clone_labels[clone_index], time = time_to_parent)

    return clone_reference_tree


def find_parent_clone(clone, clone_matrix, clone_times):
    """
    Returns the parent of a subclone, assuming this is uniquely defined as the clone from
    an earlier time point whose barcode was observed in a cell from the subclone.

    Parameters
    ----------
    clone: int
        Index of clone whose parent will be returned
    clone_matrix: Boolean array with shape [num_cells, num_clones]
        Each entry is 1 if the corresponding cell belongs to the corresponding clone and zero otherwise.
    clone_times: Vector of length num_clones
        Each entry has the time of labeling of the corresponding clone.

    Returns
    -------
    parent: int
       Index of parent clone.
    """
    clone_is_earlier = clone_times < clone_times[clone]
    clone_is_shared = np.array(np.sum(clone_matrix[np.nonzero(clone_matrix[:, clone] == 1)[0], :], 0) > 0)
    clone_is_shared = np.reshape(clone_is_shared, clone_is_shared.size) # does nothing if everyhing was ndarray, but corrects an issue with sparse matrices

    candidates = np.nonzero(clone_is_earlier & clone_is_shared)[0]

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        # no parent clone observed
        return 'root'
    else:
        print(clone_is_earlier)
        print(clone_is_shared)
        print(clone, candidates)
        raise ValueError("The input clone matrix does not unambiguously define a clonal lineage tree. Please use another method to create the tree.")


###################################
# Neighbor joining implementation #
###################################

class NeighborJoinNode:
    def __init__(self, subtree, subtree_root, has_global_root):
        self.subtree = subtree
        self.subtree_root = subtree_root
        self.has_global_root = has_global_root
        return




def neighbor_join(distance_matrix):
    """
    Creates a tree by neighbor joining with the input distance matrix

    Final row/column of distance_matrix assumed to correspond to the root
    (unmutated) barcode
    """
    n = distance_matrix.shape[0]

    assert(n >= 3) # no need to bother with neighbor joining if one or two nodes

    unjoined_nodes = []
    for i in range(n-1):
        subtree = nx.DiGraph()
        subtree.add_node(i)
        node = NeighborJoinNode(subtree, i, False)
        unjoined_nodes.append(node)

    subtree = nx.DiGraph()
    subtree.add_node('root')
    node = NeighborJoinNode(subtree, 'root', True)
    unjoined_nodes.append(node)

    next_node_to_add = -1 # integer label of next node to add, will count down
    # (leaf nodes have positive integer labels, root is labeled 'root')

    while len(unjoined_nodes) > 3:
        # recursive part of neighbor joining algorithm

        # Compute Q matrix
        Q = compute_q_matrix(distance_matrix)

        # Pick nodes to join
        nodes_to_join = pick_joined_nodes(Q)
        
        distances = distances_to_joined_node(distance_matrix, nodes_to_join)

        # Remove those nodes and add merged node
        new_node = join_nodes(unjoined_nodes[nodes_to_join[0]], unjoined_nodes[nodes_to_join[1]], next_node_to_add, distances)
        next_node_to_add = next_node_to_add - 1
        
        unjoined_nodes = [unjoined_nodes[i] for i in range(len(unjoined_nodes)) if i not in nodes_to_join]
        unjoined_nodes.append(new_node)

        # Compute new distances
        distance_matrix = compute_new_distances(distance_matrix, nodes_to_join)



    # only three remaining nodes now; join an arbitrary two of them directly
    distances = distances_to_joined_node(distance_matrix, [0,1])
    last_join = join_nodes(unjoined_nodes[0], unjoined_nodes[1], next_node_to_add, distances)
    next_node_to_add = next_node_to_add - 1
    
    last_edge_distance = (distance_matrix[2, 0] 
                          + distance_matrix[2, 1]
                          - distance_matrix[1, 0])/2
    fitted_tree = nx.compose(unjoined_nodes[2].subtree, last_join.subtree)
    
    # add 'time_to_parent' to the node that is not the root
    if last_join.has_global_root:
        assert(not unjoined_nodes[2].has_global_root)
        fitted_tree.add_edge(last_join.subtree_root, unjoined_nodes[2].subtree_root)
        fitted_tree.nodes[unjoined_nodes[2].subtree_root]['time_to_parent'] = last_edge_distance
    else:
        assert(unjoined_nodes[2].has_global_root)
        fitted_tree.add_edge(unjoined_nodes[2].subtree_root, last_join.subtree_root)
        fitted_tree.nodes[last_join.subtree_root]['time_to_parent'] = last_edge_distance

    
    return fitted_tree


        

def distances_to_joined_node(distance_matrix, nodes_to_join):
    pair_distance = distance_matrix[nodes_to_join[0], nodes_to_join[1]]
    n = distance_matrix.shape[0]

    d1 = pair_distance/2 + (np.sum(distance_matrix[:, nodes_to_join[0]])
                            - np.sum(distance_matrix[:, nodes_to_join[1]])
                            )/ (2*(n-2))

    d2 = pair_distance - d1

    return d1, d2



def join_nodes(node1, node2, new_root, distances):
    T = nx.compose(node1.subtree, node2.subtree)
    T.add_node(new_root)

    if node1.has_global_root:
        parent = node1.subtree_root
        child = new_root
    else:
        parent = new_root
        child = node1.subtree_root
    T.add_edge(parent, child)
    T.nodes[child]['time_to_parent'] = distances[0]

    if node2.has_global_root:
        parent = node2.subtree_root
        child = new_root
    else:
        parent = new_root
        child = node2.subtree_root

    T.add_edge(parent, child)
    T.nodes[child]['time_to_parent'] = distances[1]

    return NeighborJoinNode( T, new_root,  node1.has_global_root | node2.has_global_root)



def compute_q_matrix(distance_matrix):
    """
    Computes the Q-matrix for neighbor joining
    """

    n = distance_matrix.shape[0]

    Q = (n-2)*distance_matrix
    Q = Q - np.sum(distance_matrix, axis = 0)
    Q = (Q.transpose() - np.sum(distance_matrix, 1)).transpose()
    # never pick to merge a node with itself
    Q[range(n), range(n)] = np.inf
    return Q


def pick_joined_nodes(Q):
    """
    In default neighbor joining, returns the indices of the pair
    of nodes with the lowest Q value

    TODO: extend to allow stochastic neighbor joining
    """
    return np.unravel_index(np.argmin(Q), Q.shape)


def compute_new_distances(distance_matrix, nodes_to_join):
    
    n = distance_matrix.shape[0]
    kept_nodes = [i for i in range(n) if i not in nodes_to_join]

    # Leave distances between unmerged nodes unchanged
    new_distances = np.zeros([n-1, n-1])
    new_distances[:(n-2), :(n-2)] = distance_matrix[np.ix_(kept_nodes, kept_nodes)]

    distances_to_new_node = (distance_matrix[nodes_to_join[0], :]
                             + distance_matrix[nodes_to_join[1], :]
                             - distance_matrix[nodes_to_join[0],
                                               nodes_to_join[1]]
                             )/2
    distances_to_new_node = distances_to_new_node[kept_nodes]

    new_distances[n-2, 0:(n-2)] = distances_to_new_node
    new_distances[0:(n-2), n-2] = distances_to_new_node

    return new_distances
    





def tree_accuracy(tree1, tree2):
    """
    Returns the fraction of nontrivial splits appearing
    in both trees
    """
    return 1 - tree_discrepancy(tree1, tree2)

def tree_discrepancy(tree1, tree2):
    """
    Computes a version of the Robinson-Foulds distance
    between two trees rescaled to be between 0 and 1
    """
    n_leaves = len(get_leaves(tree1))
    d = robinson_foulds(tree1, tree2)
    return d/(2*(n_leaves - 3))


def robinson_foulds(tree1, tree2):
    """
    Computes the Robinson-Foulds distance between two trees
    """
    leaves1 = get_leaves(tree1)
    leaves2 = get_leaves(tree2)

    # Robinson-Foulds is only defined if the trees have the same leaf set
    assert(leaves1 == leaves2)

    ete3_t1 = tree_to_ete3(tree1)
    ete3_t2 = tree_to_ete3(tree2)
    return ete3_t1.robinson_foulds(ete3_t2)[0]








####################################
# Converting trees between formats #
####################################

def convert_newick_to_networkx(newick_tree, leaf_labels, leaf_time = None, root_label = 'root', unlabeled_nodes_added = 0, at_global_root = True):
    """
    Converts a tree from the Newick package's format to LineageOT's annotated NetworkX DiGraph. Ignores existing annotations, except edge lengths.

    Parameters
    ----------
    newick_tree : newick.Node or [newick.Node]
        A tree loaded by the Newick package.
    leaf_labels : list
        The label of each leaf in the Newick tree, sorted to align with the gene expression AnnData object filtered to cells at the corresponding time
    leaf_time : float (default None)
        The time of sampling of the leaves. If unspecified, the root of the tree is assigned time 0.
    root_label : str (default 'root')
        The label of the root node of the tree
    unlabeled_nodes_added : int (default 0)
        The number of previously-unlabeled nodes that have already been added to the tree. Leave as 0 for any top-level use of the function.
    at_global_root : bool (default True)
        Whether the function is being called to convert a full tree or a subtree.
    Returns
    -------
    tree : Networkx DiGraph
        The saved tree, in LineageOT's format.
        Each node is annotated with 'time_to_parent' and 'time' (which indicates either the time of sampling (for observed cells) or the time of division (for unobserved ancestors)).
        Edges are directed from parent to child and are annotated with 'time' equal to the child node's 'time_to_parent'.
        Observed node indices correspond to their index in leaf_labels.
    """
    assert(type(newick_tree) == newick.Node)
    tree = nx.DiGraph()
    if root_label == None:
        # node was unlabeled in original Newick tree
        # give it a new unique name
        root_label = "Unlabeled_" + str(unlabeled_nodes_added)
        unlabeled_nodes_added = unlabeled_nodes_added + 1
    root_name = root_label
    if len(newick_tree.descendants) == 0:
        # at a leaf, use index of label as name
        if not (root_label in leaf_labels):
            raise ValueError("One of the leaves in the Newick tree does not have a corresponding label in leaf_labels")
        root_label = leaf_labels.index(root_label)
        
    tree.add_node(root_label, name = root_name)
    
    for child in newick_tree.descendants:
        # convert tree for child
        subtree, unlabeled_nodes_added, new_child_label = convert_newick_to_networkx(child, leaf_labels, leaf_time = leaf_time, root_label = child.name, unlabeled_nodes_added = unlabeled_nodes_added, at_global_root = False)
        
        # add to existing tree
        tree = nx.union(tree, subtree) # not sure this is efficient
        
        # annotate edge times
        tree.add_edge(root_label, new_child_label, time=child.length)
        tree.nodes[new_child_label]['time_to_parent'] = child.length

    # annotate node times

    if leaf_time == None:
        # reference time is global root, do nothing in recursive case
        if at_global_root:
            tree.nodes[root_label]['time'] = 0
            add_node_times_from_division_times(tree, current_node = root_label) 
    else:
        if len(newick_tree.descendants) == 0:
            # at leaf
            tree.nodes[root_label]['time'] = leaf_time
        else:
            # add time annotations backwards
            times_from_children = np.array([tree.nodes[child]['time'] - tree.nodes[child]['time_to_parent'] for child in tree.successors(root_label)])
            if np.all(times_from_children == times_from_children[0]):
                # time labels from different branches are consistent
                tree.nodes[root_label]['time'] = times_from_children[0]
            else:
                raise ValueError("Edge length annotations in Newick tree are not consistent with a single sampling time for all leaves. Consider using the global root as a reference time.")

    if at_global_root:
        return tree
    else:
        return tree, unlabeled_nodes_added, root_label


def tree_to_ete3(tree):
    """
    Converts a tree to ete3 format.
    Useful for calculating Robinson-Foulds distance.
    """
    if type(tree) == ete3.coretype.tree.TreeNode:
        return tree
    elif type(tree) == nx.classes.digraph.DiGraph:
        if 'root' in tree.nodes():
            ete3_tree = subtree_to_ete3(tree, 'root')
        else:
            # pick smallest-label leaf as root
            leaves = get_leaves(tree)
            ete3_tree = subtree_to_ete3(tree, leaves[0])
    else:
        raise(NotImplementedError, 'tree_to_ete3 only converts from networkx DiGraph')

    return ete3_tree


def subtree_to_ete3(tree, current_root):
    """
    Converts the subtree from current_root to ete3 format
    """
    assert(type(tree) == nx.classes.digraph.DiGraph)
    
    ete3_tree = ete3.Tree()
    ete3_tree.name = current_root

    for subtree_root in tree.successors(current_root):
        ete3_subtree = subtree_to_ete3(tree, subtree_root)
        if 'time_to_parent' in tree.nodes[subtree_root]:
            d = tree.nodes[subtree_root]['time_to_parent']
        else:
            d = None

        ete3_tree.add_child(ete3_subtree, dist = d)

    return ete3_tree




def cvxopt_qp_from_numpy(P, q, G, h):
    """
    Converts arguments to cvxopt matrices and runs
    cvxopt's quadratic programming solver
    """

    P = cvxopt_matrix(P)
    q = cvxopt_matrix(q)
    G = cvxopt_matrix(G)
    h = cvxopt_matrix(np.squeeze(h))

    sol = cvxopt_qp(P, q, G, h)
    return np.array(sol['x'])




def compute_tree_distances(tree):
    """
    Computes the matrix of pairwise distances between leaves of the tree
    """
    num_leaves = len(get_leaves(tree)) - 1
    distances = np.zeros([num_leaves, num_leaves])
    for leaf in range(num_leaves):
        distance_dictionary, tmp = nx.multi_source_dijkstra(tree.to_undirected(), [leaf], weight = 'time')
        for target_leaf in range(num_leaves):
            distances[leaf, target_leaf] = distance_dictionary[target_leaf]
    return distances
