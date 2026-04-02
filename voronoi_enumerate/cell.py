"""
Convert a Delaunay star to Voronoi cell face structure.

The Delaunay-Voronoi duality maps:
  - Delaunay tetrahedron  ->  Voronoi vertex
  - Delaunay triangle     ->  Voronoi edge
  - Delaunay edge         ->  Voronoi face

Given the star of atom p0 (all tetrahedra containing p0), we reconstruct
the combinatorial structure of p0's Voronoi cell: its faces, each as an
ordered cycle of Voronoi vertices.
"""

from collections import defaultdict, deque
from typing import List, Tuple, Dict


def star_to_faces(star, central=0, tet_groups=None):
    """Build Voronoi cell faces from a Delaunay star.

    Parameters
    ----------
    star : list of tuple
        Tetrahedra (4-tuples of vertex indices) containing the central atom.
        This must be the complete star (from the fully resolved triangulation),
        not a partial star from a single degenerate vertex.
    central : int
        Index of the central atom.
    tet_groups : dict, optional
        Maps tet (tuple) -> group_id.  Tets with the same group_id are
        treated as a single Voronoi vertex (for unresolved degenerate
        vertices).  If None, each tet is its own vertex.

    Returns
    -------
    faces : list of tuple
        Each face is a tuple of Voronoi-vertex IDs (group IDs or tet indices)
        in cyclic order.
    face_neighbors : list of int
        The neighbor atom corresponding to each face.
    """
    # Index Voronoi vertices by position in star list
    star = [tuple(s) for s in star]

    # Build group mapping: tet -> vertex id
    if tet_groups is None:
        tet_gid = {s: i for i, s in enumerate(star)}
    else:
        tet_gid = dict(tet_groups)

    # Identify all neighbors of central
    neighbors = set()
    for s in star:
        for v in s:
            if v != central:
                neighbors.add(v)
    neighbors = sorted(neighbors)

    faces = []
    face_neighbors = []

    for a in neighbors:
        # Voronoi vertices on face F_a = tets containing both central and a
        face_tets = [s for s in star if central in s and a in s]

        if len(face_tets) < 3:
            # Face with < 3 vertices: only possible for incomplete stars
            # (partial resolution of a single degenerate vertex).
            # Skip for now; the full cell will have valid faces.
            continue

        # Adjacency: two tets are adjacent on face F_a iff they share
        # a triangle {central, a, b} for some b.
        adj = defaultdict(set)
        for i, t1 in enumerate(face_tets):
            for j, t2 in enumerate(face_tets):
                if i < j:
                    common = set(t1) & set(t2)
                    if central in common and a in common and len(common) == 3:
                        adj[i].add(j)
                        adj[j].add(i)

        # Trace the cycle.  Each vertex on the face has exactly 2 neighbors
        # on that face (it's a polygon boundary).
        cycle = _trace_cycle(adj, len(face_tets))
        if cycle is None:
            continue

        # Map cycle to group IDs
        group_cycle = [tet_gid[face_tets[i]] for i in cycle]

        # Collapse consecutive same-group entries (for merged vertices)
        if tet_groups is not None:
            group_cycle = _collapse_cycle(group_cycle)
            if len(group_cycle) < 3:
                continue  # face degenerated after merging

        faces.append(tuple(group_cycle))
        face_neighbors.append(a)

    return faces, face_neighbors


def _trace_cycle(adj, n):
    """Trace a Hamiltonian cycle through adjacency of n nodes.

    Each node must have exactly degree 2.  Returns the cycle as a list
    of node indices, or None if the graph is not a single cycle.
    """
    if n < 3:
        return None
    for node in range(n):
        if len(adj[node]) != 2:
            return None

    # Start: pick node 0, go to its first neighbor
    first_nbrs = sorted(adj[0])
    cycle = [0, first_nbrs[0]]
    prev = 0
    current = first_nbrs[0]

    for _ in range(n - 2):
        nbrs = adj[current]
        nxt = [x for x in nbrs if x != prev]
        if len(nxt) != 1:
            return None
        prev = current
        current = nxt[0]
        cycle.append(current)

    # Verify closure: last node must connect back to first
    if 0 not in adj[current]:
        return None

    return cycle


def _collapse_cycle(cycle):
    """Remove consecutive duplicate entries from a circular sequence.

    When tets at the same degenerate vertex are grouped, adjacent entries
    with the same group ID represent a single Voronoi vertex and should
    be collapsed.
    """
    if len(cycle) <= 1:
        return cycle
    result = [cycle[0]]
    for i in range(1, len(cycle)):
        if cycle[i] != result[-1]:
            result.append(cycle[i])
    # Check wrap-around
    while len(result) > 1 and result[-1] == result[0]:
        result.pop()
    return result


def orient_faces(faces):
    """Orient face cycles consistently using BFS.

    Two faces sharing an edge must traverse it in opposite directions.
    Face 0 is kept as-is; all others are oriented to be consistent.

    Parameters
    ----------
    faces : list of tuple
        Face cycles with possibly inconsistent orientation.

    Returns
    -------
    list of tuple
        Face cycles with consistent orientation.
    """
    if not faces:
        return faces

    n = len(faces)
    faces = [list(f) for f in faces]

    # For each undirected edge, record which faces use it and in which direction
    edge_faces = defaultdict(list)  # (min,max) -> [(face_idx, u, v), ...]
    for i, face in enumerate(faces):
        for j in range(len(face)):
            u, v = face[j], face[(j + 1) % len(face)]
            key = (min(u, v), max(u, v))
            edge_faces[key].append((i, u, v))

    # Build adjacency with orientation info
    # same_dir=True means both faces traverse the shared edge u->v (inconsistent)
    adj = defaultdict(list)
    for entries in edge_faces.values():
        if len(entries) == 2:
            (fi, u1, v1), (fj, u2, v2) = entries
            same_dir = (u1 == u2)  # both start from the same vertex
            adj[fi].append((fj, same_dir))
            adj[fj].append((fi, same_dir))

    # BFS: face 0 keeps its orientation; flip others as needed
    flipped = [False] * n
    visited = [False] * n
    visited[0] = True
    queue = deque([0])

    while queue:
        fi = queue.popleft()
        for fj, same_dir in adj[fi]:
            if visited[fj]:
                continue
            visited[fj] = True
            if same_dir:
                flipped[fj] = not flipped[fi]
            else:
                flipped[fj] = flipped[fi]
            queue.append(fj)

    result = []
    for i in range(n):
        if flipped[i]:
            result.append(tuple(reversed(faces[i])))
        else:
            result.append(tuple(faces[i]))

    return result
