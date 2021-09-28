
import pytest
import lineageot.inference
import lineageot.simulation

import copy
import numpy as np
import networkx as nx



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

                
        




class Test_Tree_Likelihood():
    """
    Testing calculation of tree likelihood
    """


    def test_likelihood_returns_float(self):
        distances = np.array([[ 0.0, 1, 1],
                              [ 1.0, 0, 1],
                              [ 1.0, 1, 0]])
        barcodes = np.array([[1, 0, 3, 3],
                             [2, 0, 3, 3]])
        mutation_rates = np.ones(4)*0.5
        relative_mutation_rates = [np.array([0,1,1,1])/3 for _ in range(4)]

        tree = lineageot.inference.neighbor_join(distances)
        lineageot.inference.add_leaf_barcodes(tree, barcodes)
        lineageot.inference.add_leaf_times(tree, 2)
        lineageot.inference.annotate_tree(tree, mutation_rates, time_inference_method = 'least_squares')

        
        assert(type(lineageot.inference.tree_log_likelihood(tree, mutation_rates, relative_mutation_rates))
               is np.double)
        
        
    





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
