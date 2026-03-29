"""
Compute Weinberg vectors for convex polyhedra.

The Weinberg vector is a canonical encoding of the combinatorial type of a
convex polyhedron, based on a spanning traversal of its surface.  Two polyhedra
have the same Weinberg vector iff they are combinatorially equivalent.

Algorithm (Weinberg, 1966):
  1. Represent the polyhedron as a combinatorial map: a set of darts (directed
     edges) with a face permutation sigma and edge involution tau.
  2. For each starting dart, perform a DFS-like traversal that visits every
     dart exactly once, labeling vertices in order of discovery.
  3. The Weinberg vector is the lexicographically smallest code over all
     starting darts and both surface orientations.

Reference:
  L. Weinberg, "A simple and efficient algorithm for determining isomorphism
  of planar triply connected graphs," IEEE Trans. Circuit Theory, 1966.
"""

from typing import List, Tuple, Dict, Optional


def weinberg_vector(faces: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Compute the Weinberg vector of a convex polyhedron.

    Parameters
    ----------
    faces : list of tuple of int
        Each face is a cycle of vertex indices in consistent cyclic order
        (all counterclockwise when viewed from outside, or all clockwise).
        Vertex indices need not be contiguous.

    Returns
    -------
    tuple of int
        The canonical Weinberg vector (lexicographically smallest code
        over all starting darts and both orientations).
    """
    best = None

    # Try both orientations of the surface
    for oriented_faces in [faces, [f[::-1] for f in faces]]:
        try:
            sigma, dart_face_size = _build_sigma(oriented_faces)
        except ValueError:
            return None  # Invalid cell structure (duplicate darts)
        # Try darts on smaller faces first: the canonical (lex-min) code
        # almost always starts on a smallest face, so getting a good
        # upper bound early lets early termination prune most darts.
        sorted_darts = sorted(sigma, key=lambda d: dart_face_size[d])
        for d0 in sorted_darts:
            try:
                code = _traverse(d0, sigma, len(sigma), best)
            except KeyError:
                return None  # Incomplete cell (missing edges)
            if code is not None and (best is None or code < best):
                best = code

    return best


def _build_sigma(faces):
    """Build the face-next permutation from oriented faces.

    For each dart (u, v) lying on a face boundary, sigma maps it to
    the next dart along that face.

    Returns
    -------
    sigma : dict
        Dart -> next dart permutation.
    dart_face_size : dict
        Dart -> size of the face containing it.
    """
    sigma = {}
    dart_face_size = {}
    for face in faces:
        n = len(face)
        for i in range(n):
            dart = (face[i], face[(i + 1) % n])
            nxt = (face[(i + 1) % n], face[(i + 2) % n])
            if dart in sigma:
                raise ValueError(
                    f"Dart {dart} appears in multiple faces. "
                    "Check that faces have consistent orientation."
                )
            sigma[dart] = nxt
            dart_face_size[dart] = n
    return sigma, dart_face_size


def _traverse(d0, sigma, n_darts, best=None):
    """Compute the Weinberg code starting from dart d0.

    Implements Weinberg's (1966) surface traversal:
      - Follow sigma (face boundary) when the target vertex is new.
      - When the target vertex is already labeled, scan around it using
        the vertex permutation phi = sigma . tau for the first unvisited
        outgoing dart.
      - Terminate when all outgoing darts at the target vertex are visited.

    The resulting code has length 2E + 1 (one label per dart, plus the
    initial source vertex).

    If *best* is provided, the traversal terminates early and returns None
    as soon as the code being built is lexicographically larger than *best*.
    """
    code = []
    label = {}
    next_label = 0
    visited = set()
    # Track whether the code built so far is still a prefix of best.
    # Once it's strictly less we can stop comparing; once strictly greater
    # we can abort immediately.
    tied = best is not None
    best_len = len(best) if best is not None else 0
    pos = 0

    def assign(v):
        nonlocal next_label
        if v not in label:
            next_label += 1
            label[v] = next_label
        return label[v]

    def emit(val):
        """Append *val* to code; return False if code already exceeds best."""
        nonlocal tied, pos
        code.append(val)
        if tied:
            if pos < best_len:
                if val > best[pos]:
                    return False
                elif val < best[pos]:
                    tied = False
        pos += 1
        return True

    # Visit the starting dart and emit its source vertex
    visited.add(d0)
    if not emit(assign(d0[0])):
        return None

    d = d0
    while True:
        next_v = d[1]
        is_new = next_v not in label

        if not emit(assign(next_v)):
            return None

        if is_new:
            # New vertex: advance along the face (sigma)
            d = sigma[d]
            visited.add(d)
        else:
            # Already-visited vertex: scan around it for the first
            # unvisited outgoing dart, starting from the incoming
            # direction and rotating via the vertex permutation
            # phi(v,w) = sigma[(w,v)].
            start = (next_v, d[0])          # tau(d)
            scan = start
            found = False
            while True:
                if scan not in visited:
                    d = scan
                    visited.add(d)
                    found = True
                    break
                nxt = sigma[(scan[1], scan[0])]   # phi: next dart at vertex
                if nxt == start:
                    break                         # full cycle — all visited
                scan = nxt

            if not found:
                break       # traversal complete

    return tuple(code)


def p_vector(faces: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Compute the p-vector (sorted face-size sequence) of a polyhedron.

    This is a coarser invariant than the Weinberg vector: it records
    only how many edges each face has, sorted in non-decreasing order.

    Parameters
    ----------
    faces : list of tuple of int
        Face cycles.

    Returns
    -------
    tuple of int
        Sorted tuple of face sizes.
    """
    return tuple(sorted(len(f) for f in faces))
