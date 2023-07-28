# Changelog
This file summarizes changes to the package since version 0.1.

## [Unreleased]
### Fixed
- Seeds set for simulations are not out of bounds for int32
## [0.2.0] - 2022-01-12
### Added
- read_newick() to import lineage trees from Newick format
- fit_tree() now can create lineage trees from clone labels without barcode collisions
- fit_lineage_coupling() allows unbalanced transport
- fit lineage_coupling() allows inputting growth rates 
### Fixed
- Fitting a tree with neighbor joining is much more memory efficient
