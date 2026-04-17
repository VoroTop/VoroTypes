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
from scipy.spatial import Delaunay, ConvexHull
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
    is_primary: bool = True


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


def _classify_primary(points, star_to_tris, all_tris):
    """Classify which star types are primary via the projected GKZ
    secondary polytope.

    The enumeration includes degenerate triangulations (with zero-volume
    tetrahedra from co-planar atom subsets).  These are not proper
    triangulations and only arise under measure-zero perturbations, so
    they are excluded before the GKZ analysis.  Proper triangulations
    of co-spherical points are regular and have unique GKZ vectors
    (GKZ bijection theorem); their projected GKZ vectors are vertices
    of the projected secondary polytope.
    """
    k = len(points)

    # Filter to proper triangulations (all tets have positive volume)
    tri_list = list(all_tris)
    proper = []
    for tri in tri_list:
        if all(abs(np.linalg.det(points[list(s)][1:] - points[list(s)][0]))
               > 1e-12 for s in tri):
            proper.append(tri)

    if not proper:
        return set(star_to_tris.keys())

    n_proper = len(proper)
    gkz = np.zeros((n_proper, k))

    for t_idx, tri in enumerate(proper):
        for simplex in tri:
            idx = list(simplex)
            p = points[idx]
            vol = abs(np.linalg.det(p[1:] - p[0]) / 6.0)
            for i in idx:
                gkz[t_idx, i] += vol

    A = np.column_stack([np.ones(k), points])
    Q, _ = np.linalg.qr(A, mode='reduced')

    projected = gkz - gkz @ Q @ Q.T
    mean = projected.mean(axis=0)
    centered = projected - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    tol = 1e-10 * S[0] if len(S) > 0 and S[0] > 0 else 1e-10
    rank = int(np.sum(S > tol))

    if rank == 0:
        return set(star_to_tris.keys())

    coords = centered @ Vt[:rank].T

    if rank == 1:
        vals = coords[:, 0]
        vmin, vmax = vals.min(), vals.max()
        if abs(vmax - vmin) < 1e-12:
            hull_idx = set(range(n_proper))
        else:
            hull_idx = set(np.where(np.abs(vals - vmin) < 1e-12)[0])
            hull_idx |= set(np.where(np.abs(vals - vmax) < 1e-12)[0])
    else:
        hull = ConvexHull(coords)
        hull_idx = set(hull.vertices)

    tri_to_star = {}
    for star, tri_keys in star_to_tris.items():
        for tk in tri_keys:
            tri_to_star[tk] = star

    primary_stars = set()
    for idx in hull_idx:
        tri_key = proper[idx]
        if tri_key in tri_to_star:
            primary_stars.add(tri_to_star[tri_key])

    return primary_stars


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

    # Classify primary vs secondary via projected GKZ secondary polytope
    primary_stars = _classify_primary(points, star_to_tris,
                                      list(triangulations.keys()))

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
            is_primary=(star in primary_stars),
        ))

    resolutions.sort(key=lambda r: r.star)
    return resolutions, triangulations
