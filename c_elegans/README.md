# Running LineageOT on C elegans dataset

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

Currently, this folder has one figure creation script, ```make_figure_2bcd.ipynb```. Running the entire notebook should create Figure 2b, Figure 2c, Figure 2d, and Figure S5a in the folder ```c_elegans/plots/```.