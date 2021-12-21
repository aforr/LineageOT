
import pytest
import lineageot.inference
import lineageot.simulation

import anndata
import copy
import numpy as np
import networkx as nx
import newick




def assert_tree_equality(tree_1, tree_2):
    num_nodes = len(tree_1.nodes)
    # check isomorphism (ignoring annotation)
    assert(len(nx.algorithms.isomorphism.rooted_tree_isomorphism(tree_1, 'root', tree_2, 'root')) == num_nodes)
    # check node sets match
    for node in tree_1.nodes:
        assert(node in tree_2.nodes)
    # check node annotations match
    for node in tree_1.nodes:
        for key in tree_1.nodes[node]:
            print("Node: ", node)
            print("Key: ", key)
            print(tree_2.nodes[node])
            if type(tree_1.nodes[node][key]) == float:
                assert(np.isclose(tree_1.nodes[node][key], tree_2.nodes[node][key], atol = 0))
            else:
                assert(tree_1.nodes[node][key] == tree_2.nodes[node][key])
    # check edge annotations match
    for edge in tree_1.edges:
        for key in tree_1.edges[edge]:
            print("Edge: ", edge)
            print("Key: ", key)
            if type(tree_1.edges[edge][key]) == float:
                assert(np.isclose(tree_1.edges[edge][key], tree_2.edges[edge][key], atol = 0))
            else:
                assert(tree_1.edges[edge][key] == tree_2.edges[edge][key])



class Test_Simple_Units():
    """
    Collection of tests for single simple functions
    """

    def make_small_tree(self):
        t = nx.DiGraph()
        t.add_nodes_from([1,2,3,4,5,6,7,8,9,10])
        t.add_edges_from([(1,2), (2,3), (3,4),
                          (3,5), (2,6), (6,7),
                          (6,8), (8,9), (9,10),])
        return t


    def test_get_leaves_of_two_node_tree(self):
        t = nx.DiGraph()
        t.add_node(1)
        t.add_node(2)
        t.add_edge(1,2)
        assert(lineageot.inference.get_leaves(t) == [1,2])

    def test_get_leaves_of_three_node_tree(self):
        t = nx.DiGraph()
        t.add_nodes_from([1, 2, 3])
        t.add_edges_from([(1, 2), (2, 3)])
        assert(lineageot.inference.get_leaves(t) == [1, 3])

    def test_leaves_and_non_leaves_are_all_nodes(self):
        t = self.make_small_tree()
        assert(t.number_of_nodes() == len(lineageot.inference.get_leaves(t)
                                        + lineageot.inference.get_internal_nodes(t)))












class Test_Neighbor_Join_Simple_Tree():
    """
    Testing whether neighbor-joining correctly reconstructs
    a simple tree
    """
    def setup_method(self):
        self.d = np.array([
                [0, 2, 4, 8, 7],
                [2, 0, 4, 8, 7],
                [4, 4, 0, 6, 5],
                [8, 8, 6, 0, 3],
                [7, 7, 5, 3, 0]
                ]).astype(float)

        self.correct_nodes = {'root' : {},
                              0 : {'time_to_parent': 1},
                              1 : {'time_to_parent': 1},
                              2 : {'time_to_parent': 1},
                              3 : {'time_to_parent': 2},
                              -1: {'time_to_parent': 1},
                              -2: {'time_to_parent': 2},
                              -3: {'time_to_parent': 3}
                              }
                              
        self.correct_edges = {('root', -1),
                              (-1, 3),
                              (-1, -3),
                              (-3, 2),
                              (-3, -2),
                              (-2, 0),
                              (-2, 1)}

        self.T = lineageot.inference.neighbor_join(self.d)
        return

    def test_has_correct_nodes(self):
        assert(set(self.correct_nodes.keys()) == set( self.T.nodes))

    def test_has_correct_edges(self):
        assert(self.correct_edges == set(self.T.edges))

    def test_has_correct_times(self):
        for node in self.correct_nodes:
            if not node is 'root':
                assert(self.correct_nodes[node]['time_to_parent'] ==
                       self.T.nodes[node]['time_to_parent'])







class Test_Tree_Annotation():
    """
    Testing adding time annotations to trees
    """
    

    def test_one_internal_node_time_least_squares(self):
        distances = np.array([[ 0.0, 1, 1],
                              [ 1.0, 0, 1],
                              [ 1.0, 1, 0]])
        barcodes = np.array([[1, 0, 3, 3],
                             [2, 0, 3, 3]])
        mutation_rates = np.ones(4)*0.5

        tree = lineageot.inference.neighbor_join(distances)
        lineageot.inference.add_leaf_barcodes(tree, barcodes)
        lineageot.inference.add_leaf_times(tree, 2)
        lineageot.inference.annotate_tree(tree, mutation_rates, time_inference_method = 'least_squares')

        for node in tree.nodes:
            print(tree.nodes[node])
        np.testing.assert_approx_equal(tree.nodes[-1]['time_to_parent'], 1)
        
        
    









class Test_Tree_Discrepancy():
    """
    Testing calculation of discrepancy between trees
    """

    def test_error_with_mismatched_leaf_sets(self):
        t1 = nx.DiGraph()
        t1.add_node(0)
        t2 = nx.DiGraph()
        t2.add_nodes_from([0, 1])
        t2.add_edge(0, 1)
        with pytest.raises(AssertionError):
            lineageot.inference.tree_discrepancy(t1, t2)
        return


    def test_zero_distance_to_identical_tree(self):
        t = nx.DiGraph()
        t.add_node(0)
        assert(lineageot.inference.tree_discrepancy(t, t) == 0)

    def test_distance_one_between_small_trees(self):
        t1 = nx.DiGraph()
        t1.add_nodes_from([0, 1, 2, 3, 10, 23])
        t1.add_edges_from([(0, 10), (10, 1), (10, 23),
                           (23, 2), (23, 3)])

        t2 = nx.DiGraph()
        t2.add_nodes_from([0, 1, 2, 3, 30, 12])
        t2.add_edges_from([(0, 30), (30, 3), (30, 12),
                           (12, 1), (12, 2)])
        
        assert(lineageot.inference.tree_discrepancy(t1, t2) == 1)

                
        


        
        
    





class Test_Tree_Manipulation():
    """
    Tests for functions modifying trees
    """

    def setup_method(self):
        np.random.seed(1)
        self.params = lineageot.simulation.SimulationParameters(division_time_std = 0.2,
                                          flow_type = None,
                                          diffusion_constant = 0.1,
                                          num_genes = 1,
                                          keep_tree = True,
                                          barcode_length = 15,
                                          alphabet_size = 200,
                                          relative_mutation_likelihoods = np.ones(200),
                                          mutation_rate = 1,
                                          mean_division_time = 1)
        self.initial_cell = lineageot.simulation.sample_cell(self.params)
        self.final_time = 5
        self.sample = lineageot.simulation.sample_descendants(self.initial_cell.deepcopy(),
                                             self.final_time,
                                             self.params)
        self.tree = lineageot.inference.list_tree_to_digraph(self.sample)
        return
        

    def test_tree_truncation_correct_seeds(self):
        """
        Tests whether a truncated tree has the correct seeds on its nodes
        """

        early_tree = lineageot.inference.truncate_tree(self.tree, 
                                         self.final_time - 1.5*self.params.mean_division_time,
                                         self.params)

        leaves = lineageot.inference.get_leaves(early_tree)[:-1] # exclude root
        non_leaves = lineageot.inference.get_internal_nodes(early_tree) + ['root']

        for node in non_leaves:
            early_seed = early_tree.nodes[node]['cell'].seed
            late_seed  =  self.tree.nodes[node]['cell'].seed
            assert(early_seed == late_seed)

        for node in leaves:
            # Plenty of redundancy here; we check all sibling leaves together
            parent = next(early_tree.predecessors(node))
            early_children = early_tree.successors(parent)
            early_seeds = set([early_tree.nodes[child]['cell'].seed
                               for child in early_children])

            late_children = self.tree.successors(parent)
            late_seeds = set([self.tree.nodes[child]['cell'].seed
                              for child in late_children])

            assert(early_seeds == late_seeds)

        return



    def test_truncated_shorter_max_distance(self):
        """
        Tests that the maximum distance in a truncated tree
        is smaller than the maximum distance in the original tree
        """
        early_tree = lineageot.inference.truncate_tree(self.tree, 
                                         self.final_time - 0.8,
                                         self.params)

        d1 = lineageot.inference.compute_tree_distances(early_tree)
        d2 = lineageot.inference.compute_tree_distances(self.tree)

        assert(np.max(d1) < np.max(d2))


    def test_add_nodes_at_time(self):
        """
        Tests whether nodes are added in the correct places
        """

        lineageot.inference.add_node_times_from_division_times(self.tree)
        old_tree = copy.deepcopy(self.tree)
        split_time =  self.final_time - 0.8
        num_nodes_added = lineageot.inference.add_nodes_at_time(self.tree, split_time)

        for i in range(num_nodes_added):
            n = (split_time, i)
            assert(self.tree.has_node(n))
            assert(self.tree.nodes[n]['time'] == split_time)
        

        # Check that adding the nodes didn't mess up total time to leaves
        num_leaves = len(lineageot.inference.get_leaves(old_tree)) - 1
        assert((lineageot.inference.compute_leaf_times(old_tree, num_leaves) == lineageot.inference.compute_leaf_times(self.tree, num_leaves)).all()) 
        print(lineageot.inference.tree_to_ete3(self.tree))
            




    def test_extract_ancestor_data_vs_extract_data(self):
        """
        Tests whether the extracted expression of cells matches their
        'ancestors' expression at the same time.
        """
        e1, b1 = lineageot.inference.extract_data_arrays(self.tree)
        e2, b2 = lineageot.inference.extract_ancestor_data_arrays(self.tree, self.final_time, self.params)

        assert((e1 == e2).all())
        assert((b1 == b2).all())

    def test_split_connected_tree(self):
        """
        Tests whether get_components correctly returns all of
        a connected tree
        """
        
        components = lineageot.inference.get_components(self.tree)
        assert_tree_equality(self.tree, components[0])
        assert len(components) == 1


    def test_split_small_tree(self):
        """
        Tests whether a tree is split into two components
        """
        tree = nx.DiGraph()
        tree.add_edges_from([('root', 1, {"time" : 5}), (1,2, {"time" : np.inf})])

        components = lineageot.inference.get_components(tree)
        comp_1 = nx.DiGraph()
        comp_1.add_edges_from([('root', 1, {"time" : 5})])

        assert_tree_equality(comp_1, components[0])
        assert [n for n in components[1].nodes()] == [2]
        assert len(components[1].edges()) == 0
        

        




class Test_Tree_Fitting():
    """
    Collection of tests for tree fitting
    """
    def make_minimal_dynamic_adata(self):
        rng = np.random.default_rng()
        self.t1 = 5;
        self.t2 = 10;
        n_cells_1 = 5;
        n_cells_2 = 10;
        n_cells = n_cells_1 + n_cells_2;

        n_genes = 5;

        barcode_length = 10;

        self.dynamic_adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                                obs = {"time" : np.concatenate([self.t1*np.ones(n_cells_1), self.t2*np.ones(n_cells_2)])},
                                obsm = {"barcodes" : rng.integers(low = -1, high = 10, size = (n_cells, barcode_length))}
                                )

    def make_minimal_static_adata(self):
        self.t1 = 5
        self.t2 = 10
        self.n_cells_1 = 2
        self.n_cells_2 = 4
        n_cells = self.n_cells_1 + self.n_cells_2;
        n_genes = 5
        self.clone_times = np.array([0,0])
        
        clones = np.concatenate([np.identity(2), np.kron(np.identity(2), np.ones((2, 1)))])
        self.static_adata = _adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                                obs = {"time" : np.concatenate([self.t1*np.ones(self.n_cells_1), self.t2*np.ones(self.n_cells_2)])},
                                obsm = {"X_clone" : clones}
                                )

    def make_nested_static_adata(self):
        self.t1 = 5
        self.t2 = 10
        self.n_cells_1 = 2
        self.n_cells_2 = 8
        n_cells = self.n_cells_1 + self.n_cells_2;
        n_genes = 5
        self.clone_times = np.array([0, 0, 7, 7, 7, 7])

        day_0_clones = np.concatenate([np.identity(2), np.kron(np.identity(2), np.ones((4, 1)))])
        day_7_clones = np.concatenate([np.zeros((2,4)), np.kron(np.identity(4), np.ones((2, 1)))])

        clones = np.concatenate([day_0_clones, day_7_clones], 1)


        self.static_adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                                obs = {"time" : np.concatenate([self.t1*np.ones(self.n_cells_1), self.t2*np.ones(self.n_cells_2)])},
                                obsm = {"X_clone" : clones}
                                )

        

    def test_default_tree_fit(self):
        self.make_minimal_dynamic_adata()
        lineage_tree_t2 = lineageot.fit_tree(self.dynamic_adata[self.dynamic_adata.obs['time'] == self.t2], self.t2)
        # mainly just checking no errors were thrown
        assert(len(lineage_tree_t2.nodes) == 20)

    def test_unavailable_fitting_method(self):
        self.make_minimal_dynamic_adata()
        with pytest.raises(ValueError, match="'fake method' is not an available method for fitting trees"):
            lineage_tree_t2 = lineageot.fit_tree(self.dynamic_adata[self.dynamic_adata.obs['time'] == self.t2], self.t2, method = "fake method")

    def test_clone_tree_fit(self):
        self.make_minimal_static_adata()
        lineage_tree_t2 = lineageot.fit_tree(self.static_adata[self.static_adata.obs['time'] == self.t2], self.t2, method = "non-nested clones")

        correct_tree = nx.DiGraph()
        correct_tree.add_nodes_from([0, 1, 2, 3], time = 10, time_to_parent = 10)
        correct_tree.add_nodes_from([-1, -2], time = 0, time_to_parent = 10000)
        correct_tree.add_node('root', time = -10000)

        correct_tree.add_edges_from([(-1, 0), (-1, 1), (-2, 2), (-2, 3)], time = 10)
        correct_tree.add_edges_from([('root', -1), ('root', -2)], time = 10000)

        assert_tree_equality(correct_tree, lineage_tree_t2)

    def test_nested_clone_needing_parameters(self):
        self.make_minimal_static_adata()
        with pytest.raises(ValueError, match = "clone_times must be specified"):
            lineage_tree_t2 = lineageot.fit_tree(self.static_adata[self.static_adata.obs['time'] == self.t2], self.t2, method = "clones")


    def test_nested_clone_tree_fit_1(self):
        # testing on nonnested clones
        self.make_minimal_static_adata()
        lineage_tree_t2 = lineageot.fit_tree(self.static_adata[self.static_adata.obs['time'] == self.t2], self.t2, clone_times = self.clone_times, method = "clones")

        correct_tree = nx.DiGraph()
        correct_tree.add_nodes_from([0, 1, 2, 3], time = 10, time_to_parent = 10)
        correct_tree.add_nodes_from(['clone_0', 'clone_1'], time = 0, time_to_parent = np.inf)
        correct_tree.add_node('root', time = -np.inf)

        correct_tree.add_edges_from([('clone_0', 0), ('clone_0', 1), ('clone_1', 2), ('clone_1', 3)], time = 10)
        correct_tree.add_edges_from([('root', 'clone_0'), ('root', 'clone_1')], time = np.inf)

        assert_tree_equality(correct_tree, lineage_tree_t2)


    def test_nested_clone_tree_fit_2(self):
        # testing on simply nested clones
        self.make_nested_static_adata()
        lineage_tree_t2 = lineageot.fit_tree(self.static_adata[self.static_adata.obs['time'] == self.t2], self.t2, clone_times = self.clone_times, method = "clones")

        correct_tree = nx.DiGraph()
        correct_tree.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7], time = 10, time_to_parent = 3)
        correct_tree.add_nodes_from(['clone_0', 'clone_1'], time = 0, time_to_parent = np.inf) # day 0 clones
        correct_tree.add_nodes_from(['clone_2', 'clone_3', 'clone_4', 'clone_5'], time = 7, time_to_parent = 7) # day 7 clones
        correct_tree.add_node('root', time = -np.inf)

        correct_tree.add_edges_from([('clone_0', 'clone_2'), ('clone_0', 'clone_3'), ('clone_1', 'clone_4'), ('clone_1', 'clone_5')], time = 7)
        correct_tree.add_edges_from([('clone_2', 0), ('clone_2', 1), ('clone_3', 2), ('clone_3', 3), ('clone_4', 4), ('clone_4', 5), ('clone_5', 6), ('clone_5', 7)], time = 3)
        correct_tree.add_edges_from([('root', 'clone_0'), ('root', 'clone_1')], time = np.inf)
        assert_tree_equality(correct_tree, lineage_tree_t2)




class Test_Tree_Loading():
    """
    Tests for importing and annotating trees created elsewhere
    """

    def test_convert_newick(self):
        
        newick_tree = newick.loads('(((One:0.2,Two:0.3):0.3,(Three:0.5,Four:0.3):0.2):0.3,Five:0.7):0.0;')[0]
        leaf_labels = ['One', 'Two', 'Three', 'Four', 'Five']
        lineageOT_tree = lineageot.inference.convert_newick_to_networkx(newick_tree, leaf_labels)

        correct_tree = nx.DiGraph()
        correct_tree.add_node('root', time = 0)
        correct_tree.add_node(0, name = 'One', time = 0.8, time_to_parent = 0.2)
        correct_tree.add_node(1, name = 'Two', time = 0.9, time_to_parent = 0.3)
        correct_tree.add_node(2, name = 'Three', time = 1, time_to_parent = 0.5)
        correct_tree.add_node(3, name = 'Four', time = 0.8, time_to_parent = 0.3)
        correct_tree.add_node(4, name = 'Five', time = 0.7, time_to_parent = 0.7)

        correct_tree.add_node("Unlabeled_0", name = "Unlabeled_0", time = 0.3, time_to_parent = 0.3)
        correct_tree.add_node("Unlabeled_1", name = "Unlabeled_1", time = 0.6, time_to_parent = 0.3)
        correct_tree.add_node("Unlabeled_2", name = "Unlabeled_2", time = 0.5, time_to_parent = 0.2)

        correct_tree.add_edge("root", 4, time = 0.7)
        correct_tree.add_edge("root", "Unlabeled_0", time = 0.3)
        correct_tree.add_edge("Unlabeled_0", "Unlabeled_1", time = 0.3)
        correct_tree.add_edge("Unlabeled_0", "Unlabeled_2", time = 0.2)
        correct_tree.add_edge("Unlabeled_1", 0, time = 0.2)
        correct_tree.add_edge("Unlabeled_1", 1, time = 0.3)
        correct_tree.add_edge("Unlabeled_2", 2, time = 0.5)
        correct_tree.add_edge("Unlabeled_2", 3, time = 0.3)

        assert_tree_equality(correct_tree, lineageOT_tree)
