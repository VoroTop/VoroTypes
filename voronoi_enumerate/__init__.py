"""
voronoi_enumerate: Enumerate all combinatorial types of Voronoi cells
that can arise from perturbations of a crystal lattice.

Given a crystal structure, this package:
1. Identifies degenerate Voronoi vertices (where > 4 atoms are equidistant)
2. Enumerates all regular triangulations (resolutions) at each degenerate vertex
3. Combines compatible resolutions across vertices
4. Computes Weinberg vectors for the resulting Voronoi cell topologies
"""

from .crystal import Crystal
from .voronoi import analyze_voronoi, VoronoiVertex
from .resolution import enumerate_resolutions, Resolution
from .flip_graph import enumerate_regular_triangulations
from .weinberg import weinberg_vector
from .cell import star_to_faces, orient_faces
from .combine import enumerate_cell_types, enumerate_cell_types_v2, classify_neighbors
from .filter import write_filter_file
