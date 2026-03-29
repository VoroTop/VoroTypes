"""
Deterministic enumeration of regular triangulations of a point configuration.

Two methods are provided:

1. _enumerate_exact() — Combinatorial backtracking search over all valid
   triangulations of the convex hull.  Works directly on the (possibly
   degenerate) points without perturbation.  Fast and exact for small k
   (k ≤ ~10).  This is the default.

2. _enumerate_flip_graph() — BFS over the flip graph of regular
   triangulations using a small perturbation.  Used as a fallback for
   larger k.

Important: co-spherical point configurations (the common case for
degenerate Voronoi vertices) can have "degenerate triangulations" that
include zero-volume tetrahedra (4 coplanar points).  These correspond
to valid resolutions under infinitesimal perturbation and MUST be
included.  The exact method handles this correctly; the flip-graph
method with too-large perturbation may miss or add spurious
triangulations.

References:
  - Gel'fand, Kapranov, Zelevinsky, "Discriminants, Resultants, and
    Multidimensional Determinants," 1994.
  - De Loera, Rambau, Santos, "Triangulations," 2010.
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from itertools import combinations
from collections import defaultdict


# Maximum k for exact enumeration; above this, use flip-graph fallback.
_K_EXACT_MAX = 10


def enumerate_regular_triangulations(points, seed=42, verbose=False):
    """Enumerate all regular triangulations of a point configuration.

    Parameters
    ----------
    points : array-like, shape (k, 3)
        Point configuration in R^3.
    seed : int
        Random seed (used only by the flip-graph fallback for k > 10).
    verbose : bool
        Print progress.

    Returns
    -------
    triangulations : list of frozenset
        Each frozenset contains tuples of sorted vertex indices (tetrahedra).
    """
    points = np.asarray(points, dtype=float)
    k = len(points)

    if k <= 4:
        if k == 4:
            vol = abs(_tet_volume(points, (0, 1, 2, 3)))
            if vol < 1e-15:
                return []
            return [frozenset({(0, 1, 2, 3)})]
        return [frozenset()]

    if k <= _K_EXACT_MAX:
        return _enumerate_exact(points, verbose)
    else:
        return _enumerate_flip_graph(points, seed, verbose)


# -----------------------------------------------------------------------
# Exact combinatorial enumeration (for small k)
# -----------------------------------------------------------------------

def _enumerate_exact(points, verbose=False):
    """Enumerate all triangulations via backtracking over face constraints.

    For co-spherical points, all triangulations are regular, so this
    finds all regular triangulations.  Zero-volume (degenerate) tets
    are included — they represent valid resolutions under perturbation.

    The algorithm:
      1. Compute the convex hull to get boundary faces.
      2. Detect coplanar hull face groups — when 4+ hull vertices are
         coplanar, the hull surface triangulation is ambiguous.  All
         possible triangulations of each coplanar polygon are tried.
      3. Generate all C(k,4) potential tetrahedra.
      4. For each boundary face set, backtracking search: find subsets
         of tets where each boundary face appears in exactly 1 tet
         and each internal face appears in exactly 0 or 2 tets.
    """
    k = len(points)

    # Compute convex hull for boundary faces.
    # Use a tiny perturbation only if the exact hull computation fails.
    try:
        hull = ConvexHull(points)
    except Exception:
        rng = np.random.default_rng(12345)
        pts_h = points + rng.standard_normal(points.shape) * 1e-10
        hull = ConvexHull(pts_h)

    hull_vol = hull.volume

    # All possible boundary face sets (accounting for coplanar ambiguity)
    all_boundary_sets = _all_hull_triangulations(points, hull)

    # All possible tetrahedra (including potentially degenerate ones)
    all_tets = list(combinations(range(k), 4))
    n_tets = len(all_tets)

    # Precompute absolute volumes
    tet_vols = []
    for tet in all_tets:
        tet_vols.append(abs(_tet_volume(points, tet)))

    # Precompute faces for each tet
    tet_faces = []
    for tet in all_tets:
        tet_faces.append([tuple(sorted(f)) for f in combinations(tet, 3)])

    # face → list of tet indices containing it
    face_to_tets = defaultdict(list)
    for i, faces in enumerate(tet_faces):
        for f in faces:
            face_to_tets[f].append(i)

    results = set()

    for boundary_faces in all_boundary_sets:
        _backtrack_search(boundary_faces, hull_vol, all_tets, tet_vols,
                          tet_faces, face_to_tets, results)

    result_list = list(results)
    if verbose:
        n_bsets = len(all_boundary_sets)
        extra = (f" ({n_bsets} hull triangulations)"
                 if n_bsets > 1 else "")
        print(f"  Exact enumeration: {len(result_list)} triangulations "
              f"for {k} points ({n_tets} candidate tets){extra}")

    return result_list


def _backtrack_search(boundary_faces, hull_vol, all_tets, tet_vols,
                      tet_faces, face_to_tets, results):
    """Run backtracking search for one boundary face set."""

    def _find_open_face(face_counts):
        for f in boundary_faces:
            if face_counts.get(f, 0) == 0:
                return f
        for f, c in face_counts.items():
            if f not in boundary_faces and c == 1:
                return f
        return None

    def backtrack(used, face_counts, vol):
        target = _find_open_face(face_counts)
        if target is None:
            results.add(frozenset(all_tets[i] for i in used))
            return

        for tet_idx in face_to_tets[target]:
            if tet_idx in used:
                continue

            ok = True
            for f in tet_faces[tet_idx]:
                new_c = face_counts.get(f, 0) + 1
                if f in boundary_faces:
                    if new_c > 1:
                        ok = False
                        break
                else:
                    if new_c > 2:
                        ok = False
                        break
            if not ok:
                continue

            new_vol = vol + tet_vols[tet_idx]
            if new_vol > hull_vol * 1.001 + 1e-12:
                continue

            new_used = used | {tet_idx}
            new_fc = dict(face_counts)
            for f in tet_faces[tet_idx]:
                new_fc[f] = new_fc.get(f, 0) + 1

            backtrack(new_used, new_fc, new_vol)

    backtrack(frozenset(), {}, 0.0)


def _all_hull_triangulations(points, hull):
    """Enumerate all triangulations of the hull surface.

    When 4+ hull vertices are coplanar, the hull surface has a
    polygonal face that can be triangulated in multiple ways.  Each
    triangulation gives a different set of boundary faces for the
    backtracking search.

    Returns a list of boundary face sets (each a set of sorted tuples).
    """
    hull_faces = set(tuple(sorted(int(v) for v in s))
                     for s in hull.simplices)

    coplanar_groups = _find_coplanar_groups(points, hull)

    if not coplanar_groups:
        return [hull_faces]

    # Separate fixed faces from coplanar group faces
    coplanar_face_set = set()
    for _, face_indices in coplanar_groups:
        for fi in face_indices:
            coplanar_face_set.add(
                tuple(sorted(int(v) for v in hull.simplices[fi]))
            )

    fixed_faces = hull_faces - coplanar_face_set

    # Enumerate polygon triangulations for each coplanar group
    from itertools import product as iproduct
    group_triangulations = []
    for _, face_indices in coplanar_groups:
        polygon = _polygon_from_coplanar_faces(hull.simplices, face_indices)
        tris = _enumerate_polygon_triangulations(polygon)
        group_triangulations.append(tris)

    # Cartesian product across all groups
    all_boundary_sets = []
    for combo in iproduct(*group_triangulations):
        bf = set(fixed_faces)
        for tri_faces in combo:
            bf.update(tri_faces)
        all_boundary_sets.append(bf)

    return all_boundary_sets


def _find_coplanar_groups(points, hull):
    """Find groups of coplanar hull faces (4+ coplanar hull vertices).

    Returns list of (group_vertices, face_indices) for each coplanar
    group with more than one face.
    """
    faces = hull.simplices
    n_faces = len(faces)

    hull_center = points[hull.vertices].mean(axis=0)
    normals = []
    dists = []
    for face in faces:
        p0, p1, p2 = points[face]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            normals.append(np.zeros(3))
            dists.append(0.0)
            continue
        n /= norm
        if np.dot(n, points[face[0]] - hull_center) < 0:
            n = -n
        normals.append(n)
        dists.append(np.dot(n, points[face[0]]))

    used = [False] * n_faces
    groups = []
    for i in range(n_faces):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, n_faces):
            if used[j]:
                continue
            if (abs(np.dot(normals[i], normals[j]) - 1.0) < 1e-8
                    and abs(dists[i] - dists[j]) < 1e-8):
                group.append(j)
                used[j] = True
        if len(group) > 1:
            verts = set()
            for fi in group:
                verts.update(int(v) for v in faces[fi])
            groups.append((verts, group))

    return groups


def _polygon_from_coplanar_faces(hull_simplices, face_indices):
    """Extract the ordered boundary polygon from a coplanar face group.

    Returns a list of vertex indices in cyclic order around the polygon.
    """
    edge_count = {}
    for fi in face_indices:
        face = hull_simplices[fi]
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_count[e] = edge_count.get(e, 0) + 1

    # Boundary edges appear in exactly 1 face
    boundary_edges = {e for e, c in edge_count.items() if c == 1}

    adj = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = min(adj.keys())
    ordered = [start]
    prev = None
    current = start
    while True:
        nbrs = [v for v in adj[current] if v != prev]
        if not nbrs or nbrs[0] == start:
            break
        ordered.append(nbrs[0])
        prev = current
        current = nbrs[0]

    return ordered


def _enumerate_polygon_triangulations(vertices):
    """Enumerate all triangulations of a convex polygon.

    Parameters
    ----------
    vertices : list of int
        Vertex indices in cyclic order.

    Returns
    -------
    list of set of tuple
        Each element is a set of sorted triangle tuples.
    """
    n = len(vertices)
    if n < 3:
        return [set()]
    if n == 3:
        return [{tuple(sorted(vertices))}]

    def _recurse(verts):
        if len(verts) < 3:
            return [set()]
        if len(verts) == 3:
            return [{tuple(sorted(verts))}]

        all_tris = []
        for k in range(1, len(verts) - 1):
            tri = tuple(sorted([verts[0], verts[k], verts[-1]]))
            left = _recurse(verts[:k + 1]) if k >= 2 else [set()]
            right = _recurse(verts[k:]) if len(verts) - k >= 3 else [set()]
            for lt in left:
                for rt in right:
                    all_tris.append({tri} | lt | rt)
        return all_tris

    return _recurse(vertices)


# -----------------------------------------------------------------------
# Flip-graph BFS (fallback for large k)
# -----------------------------------------------------------------------

def _enumerate_flip_graph(points, seed=42, verbose=False):
    """Enumerate regular triangulations via flip-graph BFS.

    Uses a small perturbation to break degeneracies.  May miss or add
    spurious triangulations for co-spherical configurations if the
    perturbation is too large.
    """
    k = len(points)
    rng = np.random.default_rng(seed)
    perturbed = points + rng.standard_normal(points.shape) * 1e-6

    hull_vol = ConvexHull(perturbed).volume

    initial = _initial_triangulation(perturbed, seed)
    if initial is None:
        return []

    seen = {initial}
    queue = [initial]
    all_tris = [initial]

    while queue:
        tri = queue.pop(0)
        for flipped in _all_flips(tri, perturbed):
            if flipped not in seen:
                seen.add(flipped)
                if _is_valid_volume(flipped, perturbed, hull_vol):
                    queue.append(flipped)
                    all_tris.append(flipped)

    if verbose:
        print(f"  Flip-graph BFS: {len(all_tris)} regular triangulations "
              f"(explored {len(seen)} candidates)")

    return all_tris


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _initial_triangulation(points, seed=42):
    """Get one triangulation via Delaunay of (already perturbed) points."""
    try:
        tri = Delaunay(points)
        return frozenset(
            tuple(sorted(int(v) for v in s)) for s in tri.simplices
        )
    except Exception:
        rng = np.random.default_rng(seed + 999)
        for attempt in range(5):
            eps = 1e-6 * (10 ** attempt)
            extra = points + rng.standard_normal(points.shape) * eps
            try:
                tri = Delaunay(extra)
                return frozenset(
                    tuple(sorted(int(v) for v in s)) for s in tri.simplices
                )
            except Exception:
                continue
    return None


def _all_flips(tri, points):
    """Generate all possible bistellar flips of a triangulation."""
    simplices = list(tri)
    simplex_set = set(simplices)
    results = []

    face_to_tets = defaultdict(list)
    edge_to_tets = defaultdict(list)
    for s in simplices:
        for face in combinations(s, 3):
            face_to_tets[face].append(s)
        for edge in combinations(s, 2):
            edge_to_tets[edge].append(s)

    for face, tets in face_to_tets.items():
        if len(tets) == 2:
            flipped = _flip_23(simplex_set, face, tets[0], tets[1], points)
            if flipped is not None:
                results.append(flipped)

    for edge, tets in edge_to_tets.items():
        if len(tets) == 3:
            flipped = _flip_32(simplex_set, edge, tets, points)
            if flipped is not None:
                results.append(flipped)

    return results


def _flip_23(simplex_set, face, t1, t2, points):
    """2-3 flip: replace 2 tets sharing a triangle with 3 tets sharing an edge."""
    d = [v for v in t1 if v not in face][0]
    e = [v for v in t2 if v not in face][0]
    face_verts = list(face)

    new_tets = []
    for i in range(3):
        for j in range(i + 1, 3):
            tet = tuple(sorted([face_verts[i], face_verts[j], d, e]))
            new_tets.append(tet)

    for tet in new_tets:
        if abs(_tet_volume(points, tet)) < 1e-15:
            return None

    return frozenset((simplex_set - {t1, t2}) | set(new_tets))


def _flip_32(simplex_set, edge, tets, points):
    """3-2 flip: replace 3 tets sharing an edge with 2 tets sharing a triangle."""
    outer = set()
    for t in tets:
        for v in t:
            if v not in edge:
                outer.add(v)

    if len(outer) != 3:
        return None

    outer = sorted(outer)
    e = list(edge)

    new_tets = [
        tuple(sorted(outer + [e[0]])),
        tuple(sorted(outer + [e[1]])),
    ]

    for tet in new_tets:
        if abs(_tet_volume(points, tet)) < 1e-15:
            return None

    old_set = set(tets)
    return frozenset((simplex_set - old_set) | set(new_tets))


def _tet_volume(points, tet):
    """Signed volume of a tetrahedron (det / 6)."""
    idx = list(tet)
    p = points[idx]
    M = p[1:] - p[0]
    return np.linalg.det(M) / 6.0


def _is_valid_volume(tri, points, hull_volume, tol=0.01):
    """Check that tet volumes sum to convex hull volume (no overlaps/gaps)."""
    total = sum(abs(_tet_volume(points, tet)) for tet in tri)
    return abs(total - hull_volume) / hull_volume < tol
