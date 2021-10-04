# Filtering out cells with no lineage information and pickling the resulting anndata object
# for faster loading

import anndata
import pandas as pd
import pickle
import scanpy as sc

data_path = "data/"
counts_file = "GSE126954/GSE126954_gene_by_cell_count_matrix.txt.gz"
gene_annotation_file = "GSE126954/GSE126954_gene_annotation.csv.gz"
cell_annotation_file = "GSE126954_cell_annotation_with_random_lineage_4124.csv"


print('Loading data from mtx')
adata = anndata.read_mtx(data_path+counts_file).transpose()
print('Finished loading mtx')

gene_annotations = pd.read_csv(data_path+gene_annotation_file)
cell_annotations = pd.read_csv(data_path+cell_annotation_file)
gene_annotations.index = [str(i) for i in gene_annotations.index]
cell_annotations.index = [str(i) for i in cell_annotations.index]


adata.obs = cell_annotations
adata.var = gene_annotations

cells_to_keep_mask = ~cell_annotations['random_precise_lineage'].isna()

adata_filtered = adata[cells_to_keep_mask].copy()




with open(data_path + 'pickled_filtered_anndata.p', 'wb') as file:
    pickle.dump(adata_filtered, file)
