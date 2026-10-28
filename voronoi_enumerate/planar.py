"""
Two-dimensional crystals: enumerate all p-vectors of Voronoi cells that
can arise under generic infinitesimal perturbations.

In two dimensions the topology of a Voronoi cell is recorded by its
p-vector (n, e_1, ..., e_n): the number of edges n of the cell followed
by the edge counts of its n neighbors in cyclic order, canonicalized as
the lexicographically smallest sequence over all rotations and both
orientations.  This is the representation used by VoroTop for
two-dimensional systems.

Method.  The Delaunay subdivision of an unperturbed crystal consists of
convex polygons inscribed in empty circles.  A polygon with m >= 4
vertices is a degenerate Voronoi vertex at which m cells meet.  A small
perturbation subdivides each such polygon by non-crossing diagonals:

  * generic perturbations triangulate it (m-3 diagonals); and
  * non-generic perturbations that preserve some cocircularity
    (--all-types) leave it partially subdivided (any dissection).

For points in convex position every subdivision is regular, so every
dissection is realized by some perturbation, and in the plane distinct
degenerate polygons resolve independently.  The edge count of a cell is
its degree in the resolved Delaunay graph, and its neighbors appear in
angular order around it.  The p-vector of a central cell O therefore
depends only on the resolutions of the degenerate polygons that touch O
or one of its potential neighbors.

Enumeration proceeds in two stages.  Stage 1 enumerates resolutions of
the polygons containing O, which fix O's neighbor sequence.  Stage 2
accumulates, for each neighbor sequence, the edge-count gains that the
remaining polygons contribute to O's neighbors, factoring those
polygons into independent components (polygons that share no neighbor
of O) so that each component is enumerated separately.
"""

import re
import numpy as np
from itertools import product
from functools import lru_cache
from scipy.spatial import Delaunay, cKDTree


# ------------------------------------------------------------------
# Two-dimensional crystal
# ------------------------------------------------------------------

class Crystal2D:
    """A planar crystal: two lattice vectors (rows) and fractional basis."""

    def __init__(self, lattice, frac_coords, species=None):
        self.lattice = np.asarray(lattice, dtype=float).reshape(2, 2)
        self.frac_coords = np.asarray(frac_coords, dtype=float).reshape(-1, 2)
        self.n_atoms = len(self.frac_coords)
        if species is None:
            species = ['A'] * self.n_atoms
        self.species = list(species)
        if abs(np.linalg.det(self.lattice)) < 1e-12:
            raise ValueError("Lattice vectors are linearly dependent.")

    @property
    def cart_coords(self):
        return self.frac_coords @ self.lattice

    def __repr__(self):
        return (f"Crystal2D({self.n_atoms} atoms, "
                f"lattice={self.lattice.tolist()})")

    @classmethod
    def from_file(cls, filename):
        """Read a two-dimensional crystal from a plain-text file.

        Format (blank lines and text after '#' are ignored):

            a1x a1y          first lattice vector (Cartesian)
            a2x a2y          second lattice vector (Cartesian)
            [k]              optional number of basis atoms
            fx fy [label]    fractional coordinates, one atom per line

        The optional label names the species of the atom and is used to
        name the types in the filter file.
        """
        rows = []
        with open(filename) as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if line:
                    rows.append(line.split())
        if len(rows) < 3:
            raise ValueError(f"{filename}: expected two lattice vectors "
                             f"and at least one atom.")
        try:
            lattice = [[float(x) for x in rows[0][:2]],
                       [float(x) for x in rows[1][:2]]]
        except ValueError:
            raise ValueError(f"{filename}: the first two lines must hold "
                             f"the lattice vectors a1 and a2.")
        body = rows[2:]
        if len(body[0]) == 1 and re.fullmatch(r'\d+', body[0][0]):
            k = int(body[0][0])
            body = body[1:]
            if len(body) != k:
                raise ValueError(f"{filename}: header declares {k} atoms "
                                 f"but {len(body)} are listed.")
        frac, species = [], []
        for i, r in enumerate(body):
            if len(r) < 2:
                raise ValueError(f"{filename}: atom line {i + 1} needs "
                                 f"two fractional coordinates.")
            frac.append([float(r[0]), float(r[1])])
            species.append(r[2] if len(r) > 2 else 'A')
        return cls(lattice, frac, species)


_S3 = np.sqrt(3.0)

CRYSTALS_2D = {
    # One atom per cell.
    'square':     lambda: Crystal2D([[1, 0], [0, 1]], [[0, 0]]),
    'triangular': lambda: Crystal2D([[1, 0], [0.5, _S3 / 2]], [[0, 0]]),
    # Graphene: two atoms per cell, each at the center of a triangle.
    'honeycomb':  lambda: Crystal2D([[1, 0], [0.5, _S3 / 2]],
                                    [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]),
    # Midpoints of the edges of the triangular lattice.
    'kagome':     lambda: Crystal2D([[1, 0], [0.5, _S3 / 2]],
                                    [[0, 0], [0.5, 0], [0, 0.5]]),
    # Square lattice plus the midpoints of its edges.
    'lieb':       lambda: Crystal2D([[1, 0], [0, 1]],
                                    [[0, 0], [0.5, 0], [0, 0.5]],
                                    ['corner', 'edge', 'edge']),
}
CRYSTALS_2D['graphene'] = CRYSTALS_2D['honeycomb']
CRYSTALS_2D['hexagonal'] = CRYSTALS_2D['triangular']


# ------------------------------------------------------------------
# Dissections of a convex polygon
# ------------------------------------------------------------------

@lru_cache(maxsize=None)
def dissections(m, triangulations_only=True):
    """All non-crossing diagonal sets of a convex m-gon (vertices 0..m-1).

    Returns a tuple of (diagonals, gains), where diagonals is a tuple of
    pairs (i, j), i < j, and gains[v] is the number of diagonals at v.
    With triangulations_only, only maximal sets (m-3 diagonals) are
    returned: Catalan(m-2) of them.  Otherwise every dissection is
    returned, including the empty one: the little Schroeder numbers
    1, 3, 11, 45, 197, 903, ... for m = 3, 4, 5, ...
    """
    diags = [(i, j) for i in range(m) for j in range(i + 2, m)
             if not (i == 0 and j == m - 1)]

    def crosses(d, e):
        (a, b), (c, f) = d, e
        if len({a, b, c, f}) < 4:
            return False
        return (a < c < b) != (a < f < b)

    out = []

    def extend(start, chosen):
        if not triangulations_only or len(chosen) == m - 3:
            gains = [0] * m
            for a, b in chosen:
                gains[a] += 1
                gains[b] += 1
            out.append((tuple(chosen), tuple(gains)))
        if len(chosen) == m - 3:
            return
        for k in range(start, len(diags)):
            d = diags[k]
            if all(not crosses(d, e) for e in chosen):
                extend(k + 1, chosen + [d])

    extend(0, [])
    return tuple(out)


# ------------------------------------------------------------------
# Delaunay subdivision of the unperturbed crystal
# ------------------------------------------------------------------

def _patch(crystal, n_images):
    """Points of the crystal in (2R+1)^2 cells around the origin cell."""
    pts, ids = [], []
    for tx in range(-n_images, n_images + 1):
        for ty in range(-n_images, n_images + 1):
            for b in range(crystal.n_atoms):
                f = crystal.frac_coords[b] + [tx, ty]
                pts.append(f @ crystal.lattice)
                ids.append((b, tx, ty))
    return np.array(pts), ids


def _circumcircle(p):
    """Circumcenter and radius of a triangle given as a 3x2 array."""
    a, b, c = p
    d = 2 * ((a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1]))
    a2, b2, c2 = a @ a - c @ c, b @ b - c @ c, 0.0
    ux = (a2 * (b[1] - c[1]) - b2 * (a[1] - c[1])) / d
    uy = (b2 * (a[0] - c[0]) - a2 * (b[0] - c[0])) / d
    center = np.array([ux, uy])
    return center, np.linalg.norm(a - center)


class Subdivision:
    """Delaunay subdivision of a patch into empty-circle polygons.

    Attributes
    ----------
    points : (N, 2) array of point positions
    polygons : list of tuples of point indices in counterclockwise order
    centers, radii : circumcircle of each polygon
    gaps : relative gap (d_next - r) / r of each polygon to the nearest
        point outside its circle
    """

    def __init__(self, points, region_center, region_radius, eps):
        self.points = points
        tree = cKDTree(points)
        tri = Delaunay(points)
        seen = {}
        polygons, centers, radii, gaps = [], [], [], []
        for simplex in tri.simplices:
            center, r = _circumcircle(points[simplex])
            if np.linalg.norm(center - region_center) > region_radius:
                continue
            ball = tree.query_ball_point(center, r + eps)
            d = np.linalg.norm(points[ball] - center, axis=1)
            members = [ball[i] for i in np.flatnonzero(np.abs(d - r) <= eps)]
            key = frozenset(members)
            if key in seen:
                continue
            seen[key] = len(polygons)
            ang = np.arctan2(points[members, 1] - center[1],
                             points[members, 0] - center[0])
            polygons.append(tuple(members[i] for i in np.argsort(ang)))
            centers.append(center)
            radii.append(r)
            dists, _ = tree.query(center, k=len(members) + 1)
            gaps.append((dists[-1] - r) / r)
        self.polygons = polygons
        self.centers = centers
        self.radii = radii
        self.gaps = gaps

        # Edges of the subdivision: sides of the polygons.
        self.edges = set()
        for poly in polygons:
            m = len(poly)
            for i in range(m):
                a, b = poly[i], poly[(i + 1) % m]
                self.edges.add((min(a, b), max(a, b)))
        self.fixed_nbrs = {}
        for a, b in self.edges:
            self.fixed_nbrs.setdefault(a, set()).add(b)
            self.fixed_nbrs.setdefault(b, set()).add(a)
        self.polys_at = {}
        for k, poly in enumerate(polygons):
            for v in poly:
                self.polys_at.setdefault(v, []).append(k)


# ------------------------------------------------------------------
# Canonical p-vectors
# ------------------------------------------------------------------

def canonical_pvector(seq):
    """VoroTop's canonical p-vector for a cyclic neighbor edge-count list."""
    n = len(seq)
    rev = seq[::-1]
    best = min(min(tuple(seq[i:] + seq[:i]) for i in range(n)),
               min(tuple(rev[i:] + rev[:i]) for i in range(n)))
    return (n,) + best


def _canonical_rows(rows):
    """Canonicalize many neighbor sequences of the same length at once.

    rows is an (M, n) integer array; returns the set of canonical
    p-vectors.  Each rotation or reflection of a row is packed into an
    integer whose numeric order is the lexicographic order of the row,
    so the canonical form is the minimum over the 2n candidates.
    """
    M, n = rows.shape
    if M == 0:
        return set()
    width = max(int(rows.max()).bit_length(), 1)
    if width * n > 63:
        return {canonical_pvector(list(map(int, r))) for r in rows}
    rows = rows.astype(np.uint64)
    idx = np.arange(n)
    best = None
    for perm in ([(idx + i) % n for i in range(n)] +
                 [(i - idx) % n for i in range(n)]):
        code = np.zeros(M, dtype=np.uint64)
        for j in perm:
            code = (code << np.uint64(width)) | rows[:, j]
        best = code if best is None else np.minimum(best, code)
    mask = np.uint64((1 << width) - 1)
    out = set()
    for code in np.unique(best):
        vals = []
        c = int(code)
        for _ in range(n):
            vals.append(c & int(mask))
            c >>= width
        out.add((n,) + tuple(reversed(vals)))
    return out


# ------------------------------------------------------------------
# Enumeration
# ------------------------------------------------------------------

def _polygon_options(poly, triangulations_only):
    """Resolution options of a polygon: (gains by point, diagonals)."""
    m = len(poly)
    opts = []
    for diags, gains in dissections(m, triangulations_only):
        g = {poly[v]: gains[v] for v in range(m) if gains[v]}
        d = [(poly[a], poly[b]) for a, b in diags]
        opts.append((g, d))
    return opts


def enumerate_pvectors(crystal, atom_index=0, all_types=False,
                       n_images=None, tol=1e-6, verbose=True):
    """Enumerate the p-vectors of one basis atom of a planar crystal.

    Returns (pvectors, info), where pvectors is a set of canonical
    p-vector tuples and info is a dict of diagnostics.
    """
    triangulations_only = not all_types
    pts = crystal.cart_coords
    scale = min(np.linalg.norm(crystal.lattice, axis=1))
    if crystal.n_atoms > 1:
        diffs = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diffs, axis=2)
        d = d[d > 1e-12]
        if d.size:
            scale = min(scale, d.min())

    # Grow the patch until every polygon that can influence the central
    # cell's p-vector lies well inside it.
    R = n_images if n_images is not None else 4
    while True:
        points, ids = _patch(crystal, R)
        # Distance from the origin cell to the edge of the patch.
        area = abs(np.linalg.det(crystal.lattice))
        inner = R * area / max(np.linalg.norm(crystal.lattice, axis=1))
        o = ids.index((atom_index, 0, 0))
        center = points[o]
        sub = Subdivision(points, center, 0.75 * inner, tol * scale)

        tracked = {o}
        for k in sub.polys_at.get(o, []):
            tracked.update(sub.polygons[k])
        relevant = sorted({k for v in tracked for k in sub.polys_at.get(v, [])
                           if len(sub.polygons[k]) > 3})
        reach = 0.0
        for k in set(k for v in tracked for k in sub.polys_at.get(v, [])):
            c, r = sub.centers[k], sub.radii[k]
            reach = max(reach, np.linalg.norm(c - center) + 2 * r)
        if reach < 0.5 * inner:
            break
        R *= 2
        if R > 64:
            raise RuntimeError("Could not find a patch large enough to "
                               "contain the central cell's neighborhood.")

    nbrs_all = sorted(tracked - {o})
    angle = {v: np.arctan2(*(points[v] - center)[::-1]) for v in nbrs_all}
    fixed_deg = {v: len(sub.fixed_nbrs[v]) for v in tracked}

    A = [k for k in relevant if o in sub.polygons[k]]
    B = [k for k in relevant if o not in sub.polygons[k]]
    opts = {k: _polygon_options(sub.polygons[k], triangulations_only)
            for k in relevant}
    min_gap = min(sub.gaps[k] for v in tracked for k in sub.polys_at[v])

    n_configs = 1
    for k in relevant:
        n_configs *= len(opts[k])
    if verbose:
        sizes = sorted((len(sub.polygons[k]) for k in relevant), reverse=True)
        print(f"Central cell: {len(sub.fixed_nbrs[o])} edges unperturbed, "
              f"{len(nbrs_all)} potential neighbors")
        print(f"Degenerate polygons that matter: {len(relevant)} "
              f"(sizes {sizes}); {len(A)} touch the central cell")
        print(f"Resolution combinations: {n_configs:,}")

    # Stage 1: resolutions of the polygons that contain O.
    groups = {}
    for combo in product(*(opts[k] for k in A)):
        nbrs = set(sub.fixed_nbrs[o])
        partial = {}
        for g, diags in combo:
            for a, b in diags:
                if a == o:
                    nbrs.add(b)
                elif b == o:
                    nbrs.add(a)
            for v, x in g.items():
                partial[v] = partial.get(v, 0) + x
        seq = tuple(sorted(nbrs, key=angle.get))
        deg = tuple(fixed_deg[v] + partial.get(v, 0) for v in seq)
        groups.setdefault(seq, set()).add(deg)

    # Stage 2: gains from the remaining polygons, per neighbor sequence.
    pvectors = set()
    for seq, partials in groups.items():
        n = len(seq)
        pos = {v: i for i, v in enumerate(seq)}
        profiles = []
        for k in B:
            prof = set()
            for g, _ in opts[k]:
                prof.add(tuple(sorted((pos[v], x) for v, x in g.items()
                                      if v in pos)))
            if prof != {()}:
                profiles.append((k, prof))

        # Union polygons that share a neighbor position into components.
        parent = list(range(len(profiles)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        owner = {}
        for i, (_, prof) in enumerate(profiles):
            for p in {p for pr in prof for p, _ in pr}:
                if p in owner:
                    parent[find(i)] = find(owner[p])
                else:
                    owner[p] = i
        comps = {}
        for i in range(len(profiles)):
            comps.setdefault(find(i), []).append(profiles[i][1])

        # Reachable gain vectors of each component, then their sums.
        total = np.zeros((1, n), dtype=np.int64)
        for members in comps.values():
            states = {()}
            for prof in members:
                states = {tuple(sorted(_merge(s, pr))) for s in states
                          for pr in prof}
            arr = np.zeros((len(states), n), dtype=np.int64)
            for r, s in enumerate(states):
                for p, x in s:
                    arr[r, p] = x
            total = (total[:, None, :] + arr[None, :, :]).reshape(-1, n)
            total = np.unique(total, axis=0)

        base = np.array(sorted(partials), dtype=np.int64)
        rows = (base[:, None, :] + total[None, :, :]).reshape(-1, n)
        pvectors |= _canonical_rows(np.unique(rows, axis=0))

    info = {
        'n_relevant_polygons': len(relevant),
        'n_configurations': n_configs,
        'n_neighbor_sequences': len(groups),
        'min_gap_rel': min_gap,
        'patch_images': R,
    }
    return pvectors, info


def _merge(state, profile):
    """Add a sparse gain profile to a sparse state (both (pos, x) pairs)."""
    d = dict(state)
    for p, x in profile:
        d[p] = d.get(p, 0) + x
    return d.items()
