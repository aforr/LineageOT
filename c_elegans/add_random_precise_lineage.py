# Adding a random precise lineage label to all partially-annotated cells

import numpy as np
import scanpy as sc
import anndata
import pandas as pd
import copy
import networkx as nx
import pickle

seed = 4124
data_path = "data/" # assuming script run from c_elegans folder
cell_annotation_file = "GSE126954/GSE126954_cell_annotation.csv.gz"
supplement_s6_file = "table_s6.csv"
edited_cell_annotation_file = "GSE126954_cell_annotation_with_random_lineage_" + str(seed) + ".csv"


cell_annotations = pd.read_csv(data_path+cell_annotation_file)
cell_annotations.index = [str(i) for i in cell_annotations.index]

with open(data_path + "packer_pickle_lineage_tree.p", 'rb') as file:
    full_reference_tree = pickle.load(file)

paper_table_s6 = pd.read_csv(data_path + supplement_s6_file)

cell_annotations['random_precise_lineage'] = pd.Series([np.nan for i in range(cell_annotations.shape[0])])

def add_random_precise_lineage(index, cell_annotations):
    nonspecific_lineage = cell_annotations['lineage'].iloc[index]
    if nonspecific_lineage is np.nan:
        return
    else:
        cell_annotations['random_precise_lineage'].iloc[index] = randomly_assign_x(nonspecific_lineage)
        return

def randomly_assign_x(nonspecific_lineage, reference_table = paper_table_s6):
    
    candidate_cells = reference_table[reference_table['annotation_name'] == nonspecific_lineage]['cell']
    if len(candidate_cells) == 0:
        # no reference cells have that annotation
        return np.nan
    
    else:
        cell_chosen = np.random.randint(len(candidate_cells))
        return candidate_cells.iloc[cell_chosen]
    
    
np.random.seed(seed)
for i in range(cell_annotations.shape[0]):
    add_random_precise_lineage(i, cell_annotations)

cell_annotations.to_csv(data_path + edited_cell_annotation_file)
