# Running LineageOT on _C. elegans_ dataset

This folder contains scripts for reproducing parts of the analysis of _C. elegans_ data in the [LineageOT paper](https://www.nature.com/articles/s41467-021-25133-1).

Our priority with these scripts is reproducibility rather than usability for future analyses. It is a record of what we did, not necessarily what you should do. The code is messy in places. If you would like to do something more complicated than creating identical figures to ours, and you are struggling with the code, please get in touch.

All figures were created on Ubuntu. Adaptations may need to be made for other operating systems.

## Acquiring data

You will need the data from [Packer et al. 2019](https://www.science.org/doi/10.1126/science.aax1971). 

1. Table S6 in the supplementary information. This should be saved as ```c_elegans/data/table_s6.csv```.
2. Expression data and annotations from [GSE126954](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954). The three files ```GSE126954_cell_annotation.csv.gz```, ```GSE126954_gene_annotation.csv.gz```, and ```GSE126954_gene_by_cell_count_matrix.txt.gz``` should be saved in ```c_elegans/data/GSE126954/```.

## Preprocessing data

Run the preprocessing scripts in this order:

1. Run ```make_reference_tree.py``` to create the reference lineage tree.
2. Run ```add_random_precise_lineage.py``` to do random imputation of lineage labels.
3. Run ```filter_cells.py``` to filter out cells with no lineage annotation.

## Creating figures

There are currently two figure creation scripts.

1. ```make_figure_3b-g.ipynb```. Running the entire notebook should create all of the parts of Figure 3 except Figure 3a, as well as Figure S5a, in the folder ```c_elegans/plots/```.
2. ```make_figure_3a.ipynb```. Running the entire notebook should create Figure 3a, Figure S3a, and Figure S4a, in the folder ```c_elegans/plots/```. This script does not run directly on the data; instead, errors with optimal regularization choices for each of the sampling strategies were computed with scripts very similar to ```make_figure_3b-g.ipynb``` and copied directly into the notebook.