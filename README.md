# DuallyInnervatedSpines
Detecting dually innervated spines

Dual_Innervation-AllCells.py - program to detect potential dually innervated spines by identifying pairs of spines with mesh overlap

ErrorCaseElimination.py and ErrorCaseElimination.ipynb - program to eliminate error cases/false positives (non-dually innervated spines - specifically, adjacent spines, shafts, double-headed spines, and spines with floating synapses not on the mesh) from the list of potential dually innervated spines

double-spine_test.py - program to check whether a spine is double-headed by comparing paths from synapses to openings of the meshes

shaft_test.py - program to check whether a mesh is a shaft based on number and size of openings

syapse-not-on-mesh.py - program to check whether the synapse is floating not on the mesh
