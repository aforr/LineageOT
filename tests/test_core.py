import pytest

import anndata
import lineageot
import numpy as np


class Test_Fit_Couplings():
    """
    Collection of tests for fitting lineage couplings
    """
    def make_minimal_adata(self, t1 = 5, t2 = 10):
        rng = np.random.default_rng()
        n_cells_1 = 5;
        n_cells_2 = 10;
        n_cells = n_cells_1 + n_cells_2;
        
        n_genes = 5;

        barcode_length = 10;

        self.adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                                obs = {"time" : np.concatenate([t1*np.ones(n_cells_1), t2*np.ones(n_cells_2)])},
                                obsm = {"barcodes" : rng.integers(low = -1, high = 10, size = (n_cells, barcode_length))}
                                )

    def make_minimal_clonal_adata(self, t1 = 5, t2 = 10):
        n_cells_1 = 4;
        n_cells_2 = 8;
        n_cells = n_cells_1 + n_cells_2;
        
        n_genes = 5;
        
        # clones labeled at time 0
        time_0_clones = np.concatenate([np.kron(np.identity(2), np.ones((2,1))),
                                        np.kron(np.identity(2), np.ones((4,1)))])
        # clones labeled at time 7
        time_7_clones = np.concatenate([np.zeros((4,4)),
                                        np.kron(np.identity(4), np.ones((2,1)))])
        clones = np.concatenate([time_0_clones, time_7_clones], 1)
        
        self.clone_times = np.array([0, 0, 7, 7, 7, 7])
        
        self.adata = anndata.AnnData(X = np.random.rand(n_cells, n_genes),
                                obs = {"time" : np.concatenate([t1*np.ones(n_cells_1), t2*np.ones(n_cells_2)])},
                                obsm = {"X_clone" : clones}
                                )


    def test_docs_example(self):
        """
        Checking whether the minimal pipeline example from the docs runs without errors
        """
        t1 = 5;
        t2 = 10;

        self.make_minimal_adata(t1 = t1, t2 = t2)
        lineage_tree_t2 = lineageot.fit_tree(self.adata[self.adata.obs['time'] == t2], t2)
        coupling = lineageot.fit_lineage_coupling(self.adata, t1, t2, lineage_tree_t2)
        assert np.isclose(np.sum(coupling.X), 1, atol = 0, rtol = 10**(-6))

    def test_clonal_docs_example(self):
        """
        Checking whether the minimal pipeline example for clonal data runs without errors
        """
        t1 = 5;
        t2 = 10;
        
        self.make_minimal_clonal_adata(t1 = t1, t2 = t2)
        lineage_tree_t2 = lineageot.fit_tree(self.adata[self.adata.obs['time'] == t2], t2, clone_times = self.clone_times, method = 'clones')
        coupling = lineageot.fit_lineage_coupling(self.adata, t1, t2, lineage_tree_t2)
        assert np.isclose(np.sum(coupling.X), 1, atol = 0, rtol = 10**(-6))

    def test_unbalanced(self):
        """
        Checking whether unbalanced transport runs without errors and is unbalanced
        """
        t1 = 5;
        t2 = 10;

        self.make_minimal_adata()
        lineage_tree_t2 = lineageot.fit_tree(self.adata[self.adata.obs['time'] == t2], t2)
        coupling = lineageot.fit_lineage_coupling(self.adata, t1, t2, lineage_tree_t2, balance_reg = 5)
        assert not (abs(np.sum(coupling.X) - 1) < 10^(-5))


    def test_unbalanced_marginals(self):
        """
        Checking whether unbalanced transport runs without errors and is unbalanced
        with nonuniform marginals
        """
        t1 = 5;
        t2 = 10;

        self.make_minimal_adata()
        lineage_tree_t2 = lineageot.fit_tree(self.adata[self.adata.obs['time'] == t2], t2)

        marginal_1 = np.random.rand(5) + 1
        marginal_2 = np.random.rand(10)/2 + 0.5
        coupling = lineageot.fit_lineage_coupling(self.adata,
                                                  t1,
                                                  t2,
                                                  lineage_tree_t2,
                                                  marginal_1 = marginal_1,
                                                  marginal_2 = marginal_2,
                                                  balance_reg = 5)
        assert not (abs(np.sum(coupling.X) - np.sum(marginal_1)) < 10^(-5))
        assert not (abs(np.sum(coupling.X) - np.sum(marginal_2)) < 10^(-5))

