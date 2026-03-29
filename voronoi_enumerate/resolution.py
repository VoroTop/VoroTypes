"""
Enumerate resolutions of degenerate Voronoi vertices.

Given k > 4 atoms equidistant from a degenerate Voronoi vertex in 3D,
enumerate all possible topological resolutions arising from perturbation.
Each resolution corresponds to a regular triangulation of the point
configuration; the star of the central atom determines the local Voronoi
cell topology at that vertex.

Default method: deterministic flip-graph BFS over regular triangulations
(guaranteed complete by the GKZ connectivity theorem).

Legacy method: random perturbations + Delaunay triangulation (sampling).
"""

import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, FrozenSet
from .flip_graph import enumerate_regular_triangulations


Simplex = Tuple[int, ...]
Star = FrozenSet[Simplex]


@dataclass
class Resolution:
    """A distinct resolution (star type) for the central vertex.

    Attributes:
        star: Tetrahedra containing the central vertex, sorted.
        neighbors: Vertex indices connected to central by a Delaunay edge.
        face_valences: For each neighbor, number of star tetrahedra containing
            both central and that neighbor.  Equals the number of Voronoi
            edges contributed by this vertex to the face dual to that neighbor.
        n_triangulations: Number of full triangulations that realize this star.
    """
    star: List[Simplex]
    neighbors: List[int]
    face_valences: Dict[int, int]
    n_triangulations: int = 1


def _canonicalize(simplices) -> FrozenSet[Simplex]:
    return frozenset(tuple(sorted(int(v) for v in s)) for s in simplices)


def _star_of(simplices, v) -> Star:
    return frozenset(
        tuple(sorted(int(u) for u in s)) for s in simplices if v in s
    )


def _neighbors_of(star, v) -> List[int]:
    nbrs = set()
    for s in star:
        if v in s:
            nbrs.update(u for u in s if u != v)
    return sorted(nbrs)


def _face_valence(star, central, neighbor) -> int:
    return sum(1 for s in star if central in s and neighbor in s)


def enumerate_resolutions(points, central=0, seed=42, verbose=False):
    """Enumerate distinct star types via deterministic flip-graph BFS.

    Uses the flip graph of regular triangulations to find ALL possible
    resolutions.  Guaranteed complete by the GKZ connectivity theorem.

    Parameters
    ----------
    points : array-like, shape (k, 3)
        Coordinates of the k equidistant atoms.
    central : int
        Index of the central atom whose star we extract.
    seed : int
        Random seed for initial triangulation.
    verbose : bool
        Print progress.

    Returns
    -------
    resolutions : list of Resolution
        Distinct star types, sorted by star.
    triangulations : dict
        Mapping from canonical simplex set to sorted simplex list.
    """
    points = np.asarray(points, dtype=float)
    k = len(points)

    if not (0 <= central < k):
        raise ValueError(
            f"central index {central} out of range for {k} points")

    if k <= 4:
        if k == 4:
            tet = tuple(range(k))
            star = [tet]
            nbrs = [i for i in range(k) if i != central]
            valences = {n: 1 for n in nbrs}
            tri_key = frozenset({tet})
            return (
                [Resolution(star=star, neighbors=nbrs,
                            face_valences=valences, n_triangulations=1)],
                {tri_key: [tet]},
            )
        return [], {}

    # Enumerate all regular triangulations via flip-graph BFS
    all_tris = enumerate_regular_triangulations(points, seed=seed,
                                                verbose=verbose)

    if verbose:
        print(f"  {len(all_tris)} regular triangulations found")

    # Extract distinct stars of the central vertex
    star_to_tris = defaultdict(list)
    triangulations = {}

    for tri in all_tris:
        tri_key = tri
        simplices = sorted(tri)
        triangulations[tri_key] = simplices

        star = frozenset(s for s in simplices if central in s)
        star_to_tris[star].append(tri_key)

    # Build Resolution objects
    resolutions = []
    for star, tri_keys in star_to_tris.items():
        star_list = sorted(star)
        nbrs = _neighbors_of(star_list, central)
        valences = {v: _face_valence(star_list, central, v) for v in nbrs}
        resolutions.append(Resolution(
            star=star_list,
            neighbors=nbrs,
            face_valences=valences,
            n_triangulations=len(tri_keys),
        ))

    resolutions.sort(key=lambda r: r.star)
    return resolutions, triangulations
