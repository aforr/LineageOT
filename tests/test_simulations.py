
import pytest
import lineageot.simulation as sim
import numpy as np



class Test_Simple_Units():
    """
    Collection of tests for single simple functions
    """
    def setup_method(self):
        self.params = sim.SimulationParameters()

    def test_splitting_infinite_samples(self):
        assert(sim.split_targets_between_daughters(10, np.inf, self.params) == (np.inf, np.inf))

    def test_numpy_poisson_reproducibility(self):
        """
        This is a test of how numpy's Poisson sampling works, not of any code in this project.
        If it fails, test_barcode_evolution_reproducibility() may also fail
        """
        num_trials = 10**4
        seed = np.random.randint(2**32 + 1 - num_trials)
        for i in range(num_trials):
            # Note that for rates greater than 10, numpy uses a faster algorithm
            # for which this test would fail
            rate = 10*np.random.rand()
            smaller_rate = rate*np.random.rand()
            np.random.seed(seed + i)
            n1 = np.random.poisson(rate)
            np.random.seed(seed + i)
            n2 = np.random.poisson(smaller_rate)
            assert(n1 >= n2)


    def test_barcode_evolution_reproducibility(self):
        """
        Tests whether barcode evolution is consistently reproducible with shorter
        evolution times
        """
        params = sim.SimulationParameters()
        params.enforce_barcode_reproducibility = True
        params.back_mutations = False

        params.mean_division_time = 1
        params.mutation_rate = 1

        b = sim.sample_barcode(params)
        seed = np.random.randint(2**32)
        num_trials = 10**4
        for i in range(num_trials):
            t2 = 10*params.mean_division_time*np.random.rand()
            t1 = t2*np.random.rand()

            np.random.seed(seed+i)
            b1 = sim.evolve_b(b, t1, params)

            np.random.seed(seed+i)
            b2 = sim.evolve_b(b, t2, params)
            
            # Where b1 is mutated, b2 has an identical mutation
            # Where b2 is not mutated, b1 is not mutated
            comparison_indices = (b1 >= 0) | (b2 == 0)
            assert((b1[comparison_indices] == b2[comparison_indices]).all())


class Test_Simulation_Output_Properties():
    """
    Collection of tests for whether the results of simulations look
    the way they should
    """
    def setup_method(self):
        self.params = sim.SimulationParameters()

    

    


    def test_getting_at_most_target_number_of_cells(self):
        self.params.target_num_cells = 100
        self.params.mean_division_time = 1

        initial_cell = sim.sample_cell(self.params)
        sample = sim.sample_descendants(initial_cell, 10, self.params)

        assert(len(sample) <= self.params.target_num_cells)


    def test_getting_exactly_target_number_of_cells(self):
        self.params.target_num_cells = 230
        self.params.mean_division_time = 1
        self.params.keep_tree = False

        initial_cell = sim.sample_cell(self.params)
        sample = sim.sample_descendants(initial_cell, 10, self.params)

        assert(len(sample) == self.params.target_num_cells)



    def test_subsample_size_after_simulation_with_tree(self):
        self.params.mean_division_time = 1
        self.params.keep_tree = True

        initial_cell = sim.sample_cell(self.params)
        sample = sim.sample_descendants(initial_cell, 10, self.params)

        subsample = sim.subsample_pop(sample, 123, self.params)
        # Check that we get the right number of cells at the end
        assert(len(sim.flatten_list_of_lists(subsample)) == 123)


    def test_subsample_division_time_changes(self):
        self.params.mean_division_time = 1
        self.params.division_time_std = 0
        self.params.keep_tree = True

        initial_cell = sim.sample_cell(self.params)
        sample = sim.sample_descendants(initial_cell, 1.5, self.params)

        subsample = sim.subsample_pop(sample, 1, self.params)
        
        assert(subsample[0][1] == 1.5)








    def test_negative_remaining_time_bug(self):
        # Testing for a bug that could occur
        # leading to trying to simulate for negative time
        # Passes if this runs without errors

        # 
        np.random.seed(1)
        sim_params = sim.SimulationParameters(division_time_std = 0.1,
                                               target_num_cells = 1000,
                                               flow_type = None,
                                               diffusion_constant = 0.1,
                                               num_genes = 1,
                                               keep_tree = True,
                                               barcode_length = 15,
                                               alphabet_size = 200,
                                               relative_mutation_likelihoods = np.ones(200),
                                               mutation_rate = 1,
                                               mean_division_time = 1)

        initial_cell = sim.sample_cell(sim_params)
        evolution_time = 15

        sample = sim.sample_descendants(initial_cell.deepcopy(), evolution_time, sim_params)

        assert(True)
