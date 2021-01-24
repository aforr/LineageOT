# Main file for creating simulated data

import numpy as np
import warnings
import copy


class SimulationParameters:
    """
    Storing the parameters for simulated data
    """
    
    def __init__(self,
                 timestep = 0.01,                              # Time step for Euler integration of SDE
                 diffusion_constant = 0.001,                   # Diffusion constant
                 mean_division_time = 10,                      # Mean time before cell division
                 division_time_distribution = "normal",        # Distribution of cell division times
                 division_time_std = 0,                        # Standard deviation of division times
                 target_num_cells = np.inf,                    # Upper bound on number of observed cells
                 mutation_rate = 1,                            # Rate at which barcodes mutate
                 flow_type = 'bifurcation',                    # Type of flow field simulated
                 x0_speed = 1,                                 # Speed at which cells go through transition
                 barcode_length = 15,                          # Number of elements of the barcode
                 back_mutations = False,                       # Whether barcode elements can mutate multiple times
                 num_genes = 3,                                # Number of genes defining cell state
                 initial_distribution_std = 0,                 # Standard deviation of initial cell distribution
                 alphabet_size = 200,                          # Number of possible mutations for a single barcode element
                 relative_mutation_likelihoods = np.ones(200), # Relative likelihood of each mutation
                 keep_tree = True,                             # Whether simulation output includes tree structure
                 enforce_barcode_reproducibility = True,       # Whether to use reproducible_poisson sampling
                 keep_cell_seeds = True                        # Whether to store seeds to reproduce cell trajectories individually
                 ):

        self.timestep = timestep
        self.diffusion_constant = diffusion_constant
        if flow_type in {'mismatched_clusters', 'convergent', 'partial_convergent'}:
            self.diffusion_matrix = np.diag(np.sqrt(diffusion_constant)*np.ones(num_genes))
            self.diffusion_matrix[0, 0] = 0.005*np.sqrt(x0_speed) #small diffusion in 'time' dimension
        else:
            self.diffusion_matrix = np.sqrt(diffusion_constant)

        self.mean_division_time = mean_division_time
        self.division_time_distribution = division_time_distribution
        self.division_time_std = division_time_std
        self.target_num_cells = target_num_cells

        self.mutation_rate = mutation_rate
        self.flow_type = flow_type
        self.x0_speed = x0_speed
        
        self.barcode_length = barcode_length
        self.back_mutations = back_mutations
        self.num_genes = num_genes
        self.initial_distribution_std = initial_distribution_std
        self.alphabet_size = alphabet_size
        self.keep_tree = keep_tree
        self.enforce_barcode_reproducibility = enforce_barcode_reproducibility
        self.keep_cell_seeds = keep_cell_seeds

        if len(relative_mutation_likelihoods) > alphabet_size:
            warnings.warn('relative_mutation_likelihoods too long: ignoring extra entries')
        elif len(relative_mutation_likelihoods) < alphabet_size:
            raise ValueError('relative_mutation_likelihods must be as long as alphabet_size')
        
        self.relative_mutation_likelihoods = relative_mutation_likelihoods[0:alphabet_size]
        if not self.back_mutations:
            self.relative_mutation_likelihoods[0] = 0

        self.mutation_likelihoods = relative_mutation_likelihoods/sum(relative_mutation_likelihoods)


class Cell:
    """
    Wrapper for (rna expression, barcode) arrays
    """
    def __init__(self, x, barcode, seed = None):
        self.x = x
        self.barcode = barcode

        # Note: to recover a cell's trajectory you
        # need both the seed and initial condition
        #
        # This only stores the seed
        if seed != None:
            self.seed = seed
        else:
            self.seed = np.random.randint(2**32)
        return

    def reset_seed(self):
        self.seed = np.random.randint(2**32)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return '<Cell(%s, %s)>' % (self.x, self.barcode)
    def __str__(self):
        return 'Cell with x: %s\tBarcode: %s' % ( self.x, self.barcode)






def single_bifurcation_flow(x):
    v = np.zeros(x.shape)
    v[0] = 1
    v[1] = -x[1]**3 + x[0]*x[1]
    return v


def mismatched_clusters_flow(x, params):
    """
    Single bifurcation followed by bifurcation of each cluster
    """

    if x[0] < 2:
        v = np.zeros(x.shape)
        v[0] = params.x0_speed
        v[1] = -x[1]**3 + x[0]*x[1]
        v[2] = -x[2]
        return v
    elif x[0] < 4:
        v = np.zeros(x.shape)
        v[0] = params.x0_speed

        stationary_spots = np.array([(x[0] - 1)**(-1),(2 - (x[0]-1)**(-1)) , 1])*1.25
        stationary_spots = np.concatenate([stationary_spots, [0], -stationary_spots])
        v[1] = (-1)**(np.sum(x[1] > stationary_spots))
        

        #add split in 3rd dimension
        v[2] = v[0]*2*np.sign(x[1])
        return v
    else:
        v = np.zeros(x.shape)
        v[0] = params.x0_speed

        
        stationary_spots = np.array([-0.1, 2/3 , 1.1])*np.sqrt(2)
        if x[2] < 0:
            stationary_spots = -stationary_spots

        v[1] = -np.prod(x[1] - stationary_spots)/2

        v[2] = -np.prod(x[2] - np.array([-3, 0, 3]))/1000
        return v


def partial_convergent_flow(x, params):
    """
    Single bifurcation followed by bifurcation of each cluster,
    where two of the new clusters subsequently merge
    """

    if x[0] < 2:
        # First bifurcation
        v = np.zeros(x.shape)
        v[0] = params.x0_speed
        v[1] = -x[1]**3 + x[0]*x[1]
        return v
    elif x[0] < 4:
        # Second bifurcation
        v = np.zeros(x.shape)
        v[0] = params.x0_speed

        stationary_spots = np.array([(x[0] - 1)**(-1),(2 - (x[0]-1)**(-1)) , 1])*1.25
        stationary_spots = np.concatenate([stationary_spots, [0], -stationary_spots])
        v[1] = (-1)**(np.sum(x[1] > stationary_spots))


        return v
    else:
        # Convergence
        v = np.zeros(x.shape)
        v[0] = params.x0_speed


        stationary_spots = np.array([-1.1, -2/3, 0, 2/3, 1.1])*np.sqrt(2)


        v[1] = -np.prod(x[1] - stationary_spots)/(2)
        return v




def convergent_flow(x, params):
    """
    Single bifurcation followed by convergence of the two clusters
    """

    v = np.zeros(x.shape)
    v[0] = params.x0_speed

    if x[0] < 2:
        # First bifurcation
        v[1] = -x[1]**3 + x[0]*x[1]
        return v
    else:
        # Convergence
        v[1] = -0.6*x[1]*params.x0_speed
        return v







def vector_field(x, params):
    """
    Selects a vector field and returns its value at x
    """

    if params.flow_type == "bifurcation":
        return single_bifurcation_flow(x)
    elif params.flow_type == "mismatched_clusters":
        return mismatched_clusters_flow(x, params)
    elif params.flow_type == "partial_convergent":
        return partial_convergent_flow(x, params)
    elif params.flow_type == "convergent":
        return convergent_flow(x, params)
    else:
        warnings.warn("Unknown vector field option. Vector field set to zero.")
        return np.zeros(x.shape)








def center(barcode, params):
    """
    Returns the center of the distribution p(x0|barcode)
    """
    c = np.zeros(params.num_genes)
    l = min(params.num_genes, params.barcode_length)
    c[0:l] = barcode[0:l]
    return c


def sample_barcode(params):
    """
    Samples an initial barcode
    """
    if params.back_mutations:
        barcode = np.random.randint(0,params.alphabet_size,params.barcode_length)
    else:
        barcode = np.zeros(params.barcode_length)
    return barcode

def sample_x0(barcode, params):
    """
    Samples the initial position in gene expression space
    """
    return center(barcode, params) + params.initial_distribution_std*np.random.randn(params.num_genes)

def sample_cell(params):
    """
    Samples an initial cell
    """
    barcode = sample_barcode(params)
    x0 = sample_x0(barcode, params)
    return Cell(x0, barcode)


def evolve_x(initial_x, time, params):
    """
    Returns a sample from Langevin dynamics following potential_gradient
    """
    current_x = initial_x
    if params.flow_type != None:
        current_time = 0
        #print("Evolving x for time %f\n"% time)
        while current_time < time:
            current_x = (current_x 
                         + params.timestep*vector_field(current_x, params)
                         + np.sqrt(params.timestep)*np.dot(params.diffusion_matrix,
                                                           np.random.randn(initial_x.size)))
            current_time = current_time + params.timestep
        #print("Finished evolving x\n")
    return current_x


def sample_division_time(params):
    """
    Samples the time until a cell divides
    """
    t = -1
    for _ in range(10): # try 10 times to get a positive time
        if params.division_time_distribution == 'normal':
            t = np.random.normal(params.mean_division_time, params.division_time_std)
        elif params.division_time_distribution == 'exponential':
            t = np.random.exponential(params.mean_division_time)
        else:
            t = params.mean_division_time
        if t > 0:
            break
    return t
    
#

def evolve_b(initial_barcode, time, params):
    """
    Returns the new barcode after mutations have occurred for some time
    """
    # Setting the seed for reproducibility, specifically to make the
    # process consistent with different time intervals.
    # If the random seed has been set before evolve_b is called, then
    # we want the mutations that occur in time t1 to be a subset of the
    # mutations that occur in time t2 if t2 > t1.

    mutation_seed = np.random.randint(2**32)

    rate = time*params.mutation_rate
    if (rate > 10) & params.enforce_barcode_reproducibility:
        warnings.warn("Large number of mutations expected on one edge. " +
                      "Falling back on slow non-numpy Poisson sampling.")
        num_mutations = reproducible_poisson(rate)
    else:
        num_mutations = np.random.poisson(rate)

    # need to reset the seed after the Poisson sample because the state of the
    # RNG after Poisson sampling depends on the parameter of the Poisson distribution
    np.random.seed(mutation_seed)
    for m in range(num_mutations):
        mutate_barcode(initial_barcode, params)
    return initial_barcode

def mutate_barcode(barcode, params):
    """
    Randomly changes one entry of the barcode
    """
    changed_entry = np.random.randint(0, len(barcode)) # not modeling different mutation rates across cut sites
    if params.back_mutations | (barcode[changed_entry] == 0):
        # if no back mutations, only change unmutated sites
        barcode[changed_entry] = np.random.choice(range(params.alphabet_size), p = params.mutation_likelihoods)
    return barcode

def evolve_cell(initial_cell, time, params):
    """
    Returns a new cell after both barcode and x have evolved for some time
    """
    np.random.seed(initial_cell.seed) # allows reproducibility of individual trajectories
    new_b = evolve_b(initial_cell.barcode, time, params)

    np.random.seed(initial_cell.seed + 1)
    new_x = evolve_x(initial_cell.x, time, params)

    if params.keep_cell_seeds:
        return Cell(new_x, new_b, seed = initial_cell.seed)
    else:
        return Cell(new_x, new_b)


def mask_barcode(barcode, p):
    """
    Replaces a subset of the entries of barcode with -1 to simulate missing data

    Entries are masked independently with probability p 

    Also works for an array of barcodes
    """
    mask = np.random.rand(*barcode.shape) < p
    barcode[mask] = -1
    return barcode







def sample_descendants(initial_cell, time, params, target_num_cells = None):
    """
    Samples the descendants of an initial cell
    """
    assert(isinstance(initial_cell, Cell))

    if target_num_cells == None:
        target_num_cells = params.target_num_cells

    next_division_time = sample_division_time(params)
    if (next_division_time > time) | (target_num_cells == 1):
        #print("In final division stage\n")
        if params.keep_tree:
            return[(evolve_cell(initial_cell, time, params), time)]
        else:
            return [evolve_cell(initial_cell, time, params)]
    else:
        time_remaining = time - next_division_time
        target_num_cells_1, target_num_cells_2 = split_targets_between_daughters(time_remaining, target_num_cells, params)

        while (target_num_cells_1 == 0) | (target_num_cells_2 == 0):
            # wait until we get a division where both daughters have observed descendants
            t = sample_division_time(params)
            if t > time_remaining:
                # In rare cases, you can have:
                # 1) time_remaining greater than mean_division_time
                # 2) split_targets assigning 0 cells to one branch so we skip to the next division and
                # 3) sample_division_time greater than time_remaining
                # leading to negative time_remaining and an error
                #
                # If mean_division_time >> division_time_std, this will only happen at the last division
                # which is the case this handles
                if target_num_cells == 2:
                    # In this case, since the branch we chose doesn't divide before the end of the simulation,
                    # we can assign one of its samples to the other branch
                    target_num_cells_1 = 1
                    target_num_cells_2 = 1
                    break
                else:
                    raise(ValueError)
                
            else:
                next_division_time = next_division_time + t
                time_remaining = time - next_division_time
                target_num_cells_1, target_num_cells_2 = split_targets_between_daughters(time_remaining, target_num_cells, params)

        cell_at_division = evolve_cell(initial_cell, next_division_time, params)
        assert(cell_at_division.seed >= 0)
        daughter_1 = cell_at_division.deepcopy()
        daughter_2 = cell_at_division.deepcopy()

        daughter_1.reset_seed()
        daughter_2.reset_seed()
        
        if params.keep_tree:
            # store as list of lists that keeps tree structure, ancestors, and ancestor times
            return  [sample_descendants(daughter_1, time_remaining, params, target_num_cells_1), 
                     sample_descendants(daughter_2, time_remaining, params, target_num_cells_2),
                     (cell_at_division, next_division_time)]
        else:
            # store all cells as a flat list, ignoring ancestors
            return (sample_descendants(daughter_1, time_remaining, params, target_num_cells_1) +
                    sample_descendants(daughter_2, time_remaining, params, target_num_cells_2) )
    return


def split_targets_between_daughters(time_remaining, target_num_cells, params):
    """
    Given a target number of cells to sample, divides the samples between daughters
    assuming both have the expected number of descendants at the sampling time
    """
    num_future_generations = np.floor(time_remaining/params.mean_division_time)
    num_descendants = 2**num_future_generations
    
    if target_num_cells > 2*num_descendants:
        # we expect to sample all the cells
        target_num_cells_1 = np.floor(target_num_cells/2)
    else:
        target_num_cells_1 = np.random.hypergeometric(num_descendants, num_descendants, target_num_cells)

    if target_num_cells == np.inf:
        target_num_cells_2 = np.inf
    else:
        target_num_cells_2 = target_num_cells - target_num_cells_1

    return target_num_cells_1, target_num_cells_2



def sample_population_descendants(pop, time, params):
    """
    Samples the descendants of each cell in a population
    pop: list of (expression, barcode) tuples
    """
    sampled_population = []
    num_descendants = np.zeros((len(pop))) # the number of descendants of each cell
    for cell in range(len(pop)):
        descendants = sample_descendants(pop[cell], time, params)
        num_descendants[cell] = len(descendants)
        sampled_population = sampled_population + descendants

    return sampled_population, num_descendants

def flatten_list_of_lists(tree_data):
    """
    Converts a dataset of cells with their ancestral tree structure to a list of cells
    (with ancestor and time information dropped)
    """
    assert isinstance(tree_data, list)
    if len(tree_data) == 1:
        assert(isinstance(tree_data[0][0], Cell))
        return [tree_data[0][0]]
    else:
        assert len(tree_data) == 3 # assuming this is a binary tree
        # the third entry of tree_data should be a (cell, time) tuple
        assert isinstance(tree_data[2][0], Cell)
        return flatten_list_of_lists(tree_data[0]) + flatten_list_of_lists(tree_data[1])
    return

def convert_data_to_arrays(data):
    """
    Converts a list of cells to two ndarrays,
    one for expression and one for barcodes
    """
    expressions = np.array([cell.x for cell in data])
    barcodes = np.array([cell.barcode for cell in data])
    return expressions, barcodes



def sample_pop(num_initial_cells, time, params):
    """
    Samples a population after some intervening time

    num_initial_cells:                          Number of cells in the population at time 0
    time:                                       Time when population is measured
    params:                                     Simulation parameters
    """

    if num_initial_cells == 0:
        return np.zeros((0, params.num_genes)), np.zeros((0, params.barcode_length)), np.zeros((0)), []
    
    initial_population = []
    for cell in range(num_initial_cells):
        initial_population = initial_population + [sample_cell(params)]

    sampled_population, num_descendants = sample_population_descendants(initial_population, time, params)

    expressions, barcodes = convert_data_to_arrays(sampled_population)
    return (expressions, barcodes, num_descendants, initial_population)




def subsample_list(sample, target_num_cells):
    """
    Randomly samples target_num_cells from the sample
    
    If there are fewer than target_num_cells in the sample,
    returns the whole sample
    """

    if target_num_cells > len(sample):
        return sample
    else:
        # not using permutation so the order of elements in sample is preserved
        r = np.random.rand(len(sample))
        sorted_indices = np.argsort(r)
        min_dropped_r = r[sorted_indices[target_num_cells]]
        return sample[r < min_dropped_r]





def subsample_pop(sample, target_num_cells, params, num_cells = None):
    """
    Randomly samples target_num_cells from the sample. Subsampling during the simulation by
    setting params.target_num_cells is a more efficient approximation of this.
    
    If there are fewer than target_num_cells in the sample,
    returns the whole sample

    sample should be either:

    - a list of cells, if params.keep_tree is False
    - nested lists of lists of cells encoding the tree structure, if params.keep_tree is True

    (i.e., it should match the output of sample_descendants with the same params)
    """

    if target_num_cells == 0:
        return []
    elif params.keep_tree:
        if num_cells == None:
            sample_list = flatten_list_of_lists(sample)
            num_cells = len(sample_list)

        if num_cells <= target_num_cells:
            return sample
        else:
            daughter_1_subtree = sample[0]
            daughter_2_subtree = sample[1]
            
            # TODO: there is a lot of redundant flattening happening. Should be
            #       redone if it is slow
            num_cells_1 = len(flatten_list_of_lists(daughter_1_subtree))
            num_cells_2 = len(flatten_list_of_lists(daughter_2_subtree))

            target_num_cells_1 = np.random.hypergeometric(num_cells_1, num_cells_2, target_num_cells)
            target_num_cells_2 = target_num_cells - target_num_cells_1

            # If one subtree does not get sampled, return only the other subtree subsampled
            # with its root 'time_to_parent' adjusted
            if target_num_cells_1 == 0:
                daughter_2_subtree[-1] = (daughter_2_subtree[-1][0], daughter_2_subtree[-1][1] + sample[2][1])
                return subsample_pop(daughter_2_subtree, target_num_cells_2, params, num_cells = num_cells_2)
            elif target_num_cells_2 == 0:
                daughter_1_subtree[-1] = (daughter_1_subtree[-1][0], daughter_1_subtree[-1][1] + sample[2][1])
                return subsample_pop(daughter_1_subtree, target_num_cells_1, params, num_cells = num_cells_1)
            else:
                return [subsample_pop(daughter_1_subtree, target_num_cells_1, params, num_cells = num_cells_1),
                        subsample_pop(daughter_2_subtree, target_num_cells_2, params, num_cells = num_cells_2),
                        sample[2]]

            

    else:
        return subsample_list(sample, target_num_cells)

    










def reproducible_poisson(rate):
    """
    Samples a single Poisson random variable, in a way
    that is reproducible, i.e. after
    
    np.random.seed(s)
    a = divisible_poisson(r1)
    np.random.seed(s)
    b = divisible_poisson(r2)
    
    with r1 > r2, b ~ binomial(n = a, p = r2/r1)


    This is the standard numpy Poisson sampling algorithm for rate <= 10.
    
    Note that this is relatively slow, running in O(rate) time.
    """
    T = np.exp(-rate)
    k = -1
    t = 1
    while t >= T:
        t = t*np.random.rand()
        k = k+1
        
    return k
