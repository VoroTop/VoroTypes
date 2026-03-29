"""
Voronoi analysis for crystals: compute the Voronoi cell of an atom and
identify degenerate vertices where more than 4 atoms are equidistant.

Strategy:
  1. Build a supercell large enough to enclose the Voronoi cell
  2. Compute Delaunay triangulation of the supercell
  3. Extract the star of the central atom (tetrahedra containing it)
  4. Compute circumcenters of star tetrahedra (= Voronoi vertices)
  5. Cluster coincident circumcenters (= degenerate Voronoi vertices)
  6. At each cluster, find ALL equidistant atoms (not just the 4 per tet)
"""

import numpy as np
from scipy.spatial import Delaunay
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .crystal import Crystal, AtomImage


@dataclass
class VoronoiVertex:
    """A vertex of a Voronoi cell, possibly degenerate."""
    position: np.ndarray
    circumradius: float
    atom_indices: List[int]       # indices in the supercell arrays
    atom_ids: List[AtomImage]     # crystal-level identities
    n_equidistant: int
    delaunay_tets: List[Tuple[int, ...]] = field(default_factory=list)

    @property
    def is_degenerate(self):
        return self.n_equidistant > 4

    def point_config(self, coords, central_supercell_idx):
        """Extract the point configuration for resolution enumeration.

        Parameters
        ----------
        coords : ndarray, shape (M, 3)
            Full supercell coordinates.
        central_supercell_idx : int
            Index of the central atom in the supercell.

        Returns
        -------
        points : ndarray, shape (k, 3)
            Cartesian coordinates of the k equidistant atoms,
            with the central atom at index 0.
        central_local : int
            Always 0 (central atom is placed first).
        atom_map : list of int
            Mapping from local index to supercell index.
        """
        # Put central atom first
        others = [i for i in self.atom_indices if i != central_supercell_idx]
        order = [central_supercell_idx] + others
        points = coords[order]
        return points, 0, order


def circumcenter_3d(pts):
    """Circumcenter and circumradius of a tetrahedron.

    Parameters
    ----------
    pts : array-like, shape (4, 3)

    Returns
    -------
    center : ndarray, shape (3,), or None if degenerate
    radius : float or None
    """
    A = pts[0]
    b = pts[1] - A
    c = pts[2] - A
    d = pts[3] - A

    M = 2 * np.array([b, c, d])
    rhs = np.array([b @ b, c @ c, d @ d])

    try:
        x = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return None, None

    # Check for near-singular (very flat tetrahedron)
    if np.linalg.cond(M) > 1e12:
        return None, None

    center = A + x
    radius = np.linalg.norm(x)
    return center, radius


def analyze_voronoi(crystal, atom_index=0, n_images=3, tol=1e-8,
                    near_gap_threshold=None):
    """Compute Voronoi vertices and detect degeneracies.

    Parameters
    ----------
    crystal : Crystal
    atom_index : int
        Which atom in the unit cell to analyze.
    n_images : int
        Periodic images in each direction for supercell.
    tol : float
        Absolute tolerance for circumcenter clustering and equidistance.
        Only used as a floor; the adaptive algorithm typically computes
        a larger effective tolerance from the gap structure in the data.
    near_gap_threshold : float, optional
        If set, atoms whose relative gap (d - r) / r to a vertex is below
        this threshold are treated as equidistant, even if the adaptive
        gap detection would exclude them.  This captures near-equidistant
        neighbors that finite perturbation would bring into the vertex,
        making the enumeration complete for perturbations up to this scale.
        Typical value: 0.01 (captures atoms within 1% of the circumradius).

    Returns
    -------
    vertices : list of VoronoiVertex
        All Voronoi vertices of the central atom's cell.
    central_idx : int
        Index of the central atom in the supercell.
    coords : ndarray
        Full supercell coordinates (needed for point_config).
    images : list of AtomImage
        Atom identities for the supercell.
    """
    coords, images = crystal.make_supercell(n_images)

    # Find the central atom: atom_index in image (0,0,0)
    central_idx = None
    for i, img in enumerate(images):
        if img.atom_index == atom_index and img.image == (0, 0, 0):
            central_idx = i
            break
    assert central_idx is not None, (
        f"Central atom {atom_index} not found in supercell"
    )

    # Delaunay triangulation
    tri = Delaunay(coords)

    # Star of the central atom
    star_mask = np.any(tri.simplices == central_idx, axis=1)
    star = tri.simplices[star_mask]

    # Circumcenter of each star tetrahedron
    cc_list = []   # (center, radius, simplex_indices)
    for simplex in star:
        center, radius = circumcenter_3d(coords[simplex])
        if center is not None:
            cc_list.append((center, radius, list(int(v) for v in simplex)))

    if not cc_list:
        return [], central_idx, coords, images

    centers = np.array([c for c, r, s in cc_list])
    radii = np.array([r for c, r, s in cc_list])

    # Cluster coincident circumcenters using adaptive gap detection
    groups = _cluster_centers(centers, radii, tol)

    # Build VoronoiVertex for each cluster
    vertices = []

    for group in groups:
        pos = np.mean(centers[group], axis=0)
        rad = np.mean(radii[group])

        # Find ALL equidistant atoms using gap-based detection
        dists = np.linalg.norm(coords - pos, axis=1)
        equidistant = _find_equidistant(dists, rad, tol,
                                         near_gap_threshold=near_gap_threshold)

        # Collect original Delaunay tets at this vertex (in star of central)
        tets = [tuple(sorted(int(v) for v in cc_list[i][2])) for i in group]

        vertices.append(VoronoiVertex(
            position=pos,
            circumradius=rad,
            atom_indices=equidistant,
            atom_ids=[images[i] for i in equidistant],
            n_equidistant=len(equidistant),
            delaunay_tets=tets,
        ))

    return vertices, central_idx, coords, images


def _find_equidistant(dists, radius, tol, near_gap_threshold=None):
    """Find all atoms equidistant from a Voronoi vertex using gap detection.

    Instead of a fixed tolerance, sort distances to the vertex and look for
    the natural gap between the cluster of equidistant atoms and the next
    shell.  This handles floating-point coordinate noise robustly.

    Parameters
    ----------
    dists : ndarray, shape (N,)
        Distance from each atom to the vertex.
    radius : float
        Circumradius (expected distance for equidistant atoms).
    tol : float
        Absolute tolerance floor.  Atoms within this distance of
        the radius are always included.
    near_gap_threshold : float, optional
        If set, after finding the equidistant set by adaptive gap
        detection, extend it to include any additional atoms whose
        gap to the last included atom (in sorted distance order) is
        below this fraction of the circumradius.  This captures atoms
        that are nearly equidistant to the last included atom — i.e.,
        atoms in the same distance shell that the adaptive detection
        split apart.

    Returns
    -------
    equidistant : list of int
        Sorted indices of equidistant atoms.
    """
    # Start with atoms that are definitely close to the circumradius.
    # Use a generous initial window to collect candidates.
    residuals = np.abs(dists - radius)
    # Sort by residual to find the gap
    order = np.argsort(residuals)
    sorted_res = residuals[order]

    # The first 4 atoms (the Delaunay tet) should always be included.
    # Walk through sorted residuals and find the first "large" gap.
    # A gap is large if it exceeds both:
    #   (a) 100x the spread of the current cluster, and
    #   (b) the absolute tolerance floor
    n_min = 4  # at least 4 atoms define the vertex
    n = len(sorted_res)

    # Find the cutoff: include atoms until we hit a gap that's clearly
    # larger than the noise within the equidistant cluster.
    cutoff_idx = n_min
    for k in range(n_min, min(n, 30)):  # don't scan beyond ~30 candidates
        cluster_spread = sorted_res[k - 1] - sorted_res[0]
        gap = sorted_res[k] - sorted_res[k - 1]

        # The gap must be large relative to the cluster spread.
        # For exact coordinates, cluster_spread=0 and any gap works.
        # For noisy coordinates, the cluster has some spread (~1e-6)
        # and the gap to the next shell is much larger (~1 Å).
        noise_scale = max(cluster_spread, tol)
        if gap > 100 * noise_scale:
            cutoff_idx = k
            break
    else:
        # No clear gap found in first 30 — use all atoms within tol
        cutoff_idx = int(np.searchsorted(sorted_res, tol * 10))
        cutoff_idx = max(cutoff_idx, n_min)

    # Extend to include near-equidistant atoms if threshold is set.
    # Check the gap between consecutive atoms in sorted-distance order:
    # if the next atom is very close to the last included atom (relative
    # to the circumradius), include it too — they're in the same shell.
    if near_gap_threshold is not None and radius > 0:
        abs_threshold = near_gap_threshold * radius
        sorted_dists = dists[order]  # actual distances, sorted by residual
        for k in range(cutoff_idx, min(n, 30)):
            gap_to_prev = abs(sorted_dists[k] - sorted_dists[k - 1])
            if gap_to_prev <= abs_threshold:
                cutoff_idx = k + 1
            else:
                break

    return sorted(int(i) for i in order[:cutoff_idx])


def _cluster_centers(centers, radii, tol):
    """Group circumcenters that are within tolerance of each other.

    Uses adaptive gap-based tolerance: computes pairwise distances
    between all circumcenters and uses the gap structure to determine
    which centers are "coincident" vs truly distinct.

    Returns list of groups, each group a list of indices.
    """
    n = len(centers)
    if n == 0:
        return []

    # Compute all pairwise distances between circumcenters
    # For typical star sizes (< 100 tets), this is fast
    from scipy.spatial.distance import pdist, squareform
    if n == 1:
        return [[0]]

    pos_dists = squareform(pdist(centers))
    rad_diffs = np.abs(radii[:, None] - radii[None, :])

    # Determine adaptive clustering tolerance from gap structure.
    # Collect all nonzero pairwise distances and find the gap between
    # "nearly coincident" pairs and "truly distinct" pairs.
    upper_tri = np.triu_indices(n, k=1)
    all_dists = pos_dists[upper_tri]
    all_rdiffs = rad_diffs[upper_tri]
    # Combined metric: position distance + radius difference
    combined = all_dists + all_rdiffs

    if len(combined) == 0:
        return [[0]]

    sorted_combined = np.sort(combined)

    # Find the gap: the first large jump in sorted pairwise distances.
    # Pairs before the gap are "coincident"; pairs after are "distinct".
    cluster_tol = tol  # default to absolute tolerance
    for k in range(len(sorted_combined) - 1):
        gap = sorted_combined[k + 1] - sorted_combined[k]
        spread = sorted_combined[k] - sorted_combined[0] if k > 0 else 0
        noise = max(spread, tol)
        if gap > 100 * noise and sorted_combined[k + 1] > tol:
            # The threshold is the midpoint of the gap
            cluster_tol = (sorted_combined[k] + sorted_combined[k + 1]) / 2
            break
    else:
        # No clear gap — either all coincident or all distinct.
        # If all distances are small (< 1e-4), treat as one cluster.
        # Otherwise, use the absolute tolerance.
        if sorted_combined[-1] < 1e-4:
            cluster_tol = sorted_combined[-1] * 2
        else:
            cluster_tol = tol

    # Standard clustering with the adaptive tolerance
    assigned = [False] * n
    groups = []

    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if combined[_upper_idx(i, j, n)] < cluster_tol:
                group.append(j)
                assigned[j] = True
        groups.append(group)

    return groups


def _upper_idx(i, j, n):
    """Convert (i, j) with i < j to index into condensed distance matrix."""
    return n * i - i * (i + 1) // 2 + j - i - 1


def summarize(vertices, images=None):
    """Print a summary of Voronoi vertex analysis.

    Parameters
    ----------
    vertices : list of VoronoiVertex
    images : list of AtomImage, optional
        If provided, print atom identities.
    """
    n_degen = sum(1 for v in vertices if v.is_degenerate)
    n_generic = len(vertices) - n_degen
    print(f"Voronoi vertices: {len(vertices)} total "
          f"({n_generic} generic, {n_degen} degenerate)")

    # Group degenerate vertices by number of equidistant atoms
    from collections import Counter
    degen_counts = Counter(v.n_equidistant for v in vertices if v.is_degenerate)
    for k in sorted(degen_counts):
        print(f"  {degen_counts[k]} vertices with {k} equidistant atoms")

    if n_generic > 0:
        print(f"  {n_generic} vertices with 4 equidistant atoms (generic)")

    # Detail on degenerate vertices
    for i, v in enumerate(vertices):
        if not v.is_degenerate:
            continue
        print(f"\nDegenerate vertex {i}: {v.n_equidistant} equidistant atoms")
        print(f"  Position: ({v.position[0]:.6f}, "
              f"{v.position[1]:.6f}, {v.position[2]:.6f})")
        print(f"  Circumradius: {v.circumradius:.6f}")
        if images:
            for ai in v.atom_indices:
                img = images[ai]
                print(f"    atom {img.atom_index} image {img.image} "
                      f"at ({img.cart_pos[0]:.4f}, "
                      f"{img.cart_pos[1]:.4f}, {img.cart_pos[2]:.4f})")
