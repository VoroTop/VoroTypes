"""
Combine per-vertex resolutions into global Voronoi cell types.

Two algorithms are provided:

1. enumerate_cell_types_v2() — enumerate resolutions of the full point
   configuration at each vertex, then use incremental star construction
   with deduplication to find all distinct global stars efficiently.

2. enumerate_cell_types() — LEGACY: iterate over the Cartesian product
   of all pre-computed resolutions.  Kept for backward compatibility.
"""

from itertools import product
from collections import defaultdict
from math import prod
from .resolution import enumerate_resolutions
from .cell import star_to_faces, orient_faces
from .weinberg import weinberg_vector, p_vector


# -----------------------------------------------------------------------
# Neighbor classification
# -----------------------------------------------------------------------

def classify_neighbors(vertices, central_idx):
    """Classify neighbors of the central atom as full or partial.

    Full neighbors appear in at least one generic (non-degenerate)
    tetrahedron — they always share a Voronoi face with the central
    atom regardless of how degenerate vertices resolve.

    Partial neighbors appear only at degenerate vertices — whether
    they share a face depends on the resolution.

    Parameters
    ----------
    vertices : list of VoronoiVertex
    central_idx : int
        Supercell index of the central atom.

    Returns
    -------
    full : list of int
        Sorted supercell indices of full neighbors.
    partial : list of int
        Sorted supercell indices of partial neighbors.
    """
    full = set()
    all_atoms = set()

    for v in vertices:
        for ai in v.atom_indices:
            if ai != central_idx:
                all_atoms.add(ai)
        if not v.is_degenerate:
            for ai in v.atom_indices:
                if ai != central_idx:
                    full.add(ai)

    partial = sorted(all_atoms - full)
    return sorted(full), partial


# -----------------------------------------------------------------------
# Validity pruning: pairwise compatibility of adjacent degenerate vertices
# -----------------------------------------------------------------------

def _precompute_compatibility(generic_tets, vertex_options, central_idx):
    """Precompute pairwise compatibility for adjacent degenerate vertices.

    Two degenerate vertices are adjacent if they share atoms (besides the
    central atom).  For each adjacent pair and each pair of resolutions,
    we check whether the combined tets on shared faces produce a valid
    partial structure (no tet with degree > 2 in the face adjacency
    graph).  Degree > 2 is a permanent failure: adding tets from future
    vertices can only increase degree, never reduce it.

    Returns
    -------
    adj : list of list of int
        adj[vi] = sorted list of adjacent vertex indices.
    compat : dict
        compat[(vi, vj)] is a list of lists:
        compat[(vi, vj)][ri][rj] = True if compatible.
    has_coupling : bool
        True if any pair of vertices is adjacent.
    """
    d = len(vertex_options)

    # Atoms (besides central) appearing in each vertex's tets
    vertex_atoms = []
    for opts in vertex_options:
        atoms = set()
        for tets, _ in opts:
            for tet in tets:
                atoms.update(tet)
        atoms.discard(central_idx)
        vertex_atoms.append(atoms)

    # Adjacency
    adj = [[] for _ in range(d)]
    pairs = []
    for vi in range(d):
        for vj in range(vi + 1, d):
            if vertex_atoms[vi] & vertex_atoms[vj]:
                adj[vi].append(vj)
                adj[vj].append(vi)
                pairs.append((vi, vj))

    has_coupling = len(pairs) > 0
    if not has_coupling:
        return adj, {}, False

    # Generic tets indexed by neighbor atom
    generic_by_nbr = {}
    for tet in generic_tets:
        if central_idx in tet:
            for a in tet:
                if a != central_idx:
                    generic_by_nbr.setdefault(a, []).append(tet)

    compat = {}
    for vi, vj in pairs:
        shared = vertex_atoms[vi] & vertex_atoms[vj]
        n_ri = len(vertex_options[vi])
        n_rj = len(vertex_options[vj])
        mat = [[True] * n_rj for _ in range(n_ri)]

        for ri in range(n_ri):
            tets_i = vertex_options[vi][ri][0]
            for rj in range(n_rj):
                tets_j = vertex_options[vj][rj][0]

                for a in shared:
                    # Collect tets containing {central, a}.
                    # Use a set to deduplicate: two vertices may share
                    # the same equidistant atom set and produce identical
                    # tets.  The actual star construction uses frozenset
                    # union which deduplicates, so the check must too.
                    face_tets_set = set(generic_by_nbr.get(a, []))
                    for tet in tets_i:
                        if central_idx in tet and a in tet:
                            face_tets_set.add(tet)
                    for tet in tets_j:
                        if central_idx in tet and a in tet:
                            face_tets_set.add(tet)
                    face_tets = list(face_tets_set)

                    if len(face_tets) < 2:
                        continue

                    # Degree check: any tet adjacent to > 2 others?
                    n = len(face_tets)
                    deg = [0] * n
                    for ii in range(n):
                        for jj in range(ii + 1, n):
                            common = set(face_tets[ii]) & set(face_tets[jj])
                            if (central_idx in common and a in common
                                    and len(common) == 3):
                                deg[ii] += 1
                                deg[jj] += 1

                    if any(x > 2 for x in deg):
                        mat[ri][rj] = False
                        break

        compat[(vi, vj)] = mat
        # Transpose for (vj, vi) lookups
        compat[(vj, vi)] = [[mat[ri][rj] for ri in range(n_ri)]
                            for rj in range(n_rj)]

    return adj, compat, has_coupling


def _make_prune_fn(adj, compat):
    """Build a prune callback for orderly_generate.

    The returned function checks pairwise compatibility between the
    most recently assigned vertex and all previously assigned adjacent
    vertices.  Cost: O(|adj[v]|) table lookups per call.
    """
    def prune(combo, depth):
        vk = depth - 1
        rk = combo[vk]
        for vj in adj[vk]:
            if vj >= depth:
                continue  # not yet assigned
            if not compat[(vk, vj)][rk][combo[vj]]:
                return True
        return False
    return prune


# -----------------------------------------------------------------------
# Classification helper
# -----------------------------------------------------------------------

def _classify_combo(combo, orbit_size, generic_tets, vertex_options,
                    central_idx):
    """Build the star for a combo and classify it.

    Returns (weinberg_vector, info_dict) or None if invalid.
    """
    star = generic_tets
    has_unresolved = False
    for vi, ri in enumerate(combo):
        tets, is_unres = vertex_options[vi][ri]
        star = star | tets
        if is_unres:
            has_unresolved = True

    global_star = list(star)

    tet_groups = None
    if has_unresolved:
        tet_groups = {}
        next_gid = 0
        for t in generic_tets:
            if t not in tet_groups:
                tet_groups[t] = next_gid
                next_gid += 1
        for vi, ri in enumerate(combo):
            tets, is_unres = vertex_options[vi][ri]
            if is_unres:
                gid = next_gid
                next_gid += 1
                for t in tets:
                    if t not in tet_groups:
                        tet_groups[t] = gid
            else:
                for t in tets:
                    if t not in tet_groups:
                        tet_groups[t] = next_gid
                        next_gid += 1

    faces, face_nbrs = star_to_faces(global_star, central=central_idx,
                                     tet_groups=tet_groups)
    if not faces:
        return None

    faces = orient_faces(faces)
    pv = p_vector(faces)
    wv = weinberg_vector(faces)

    if wv is None:
        return None

    return wv, {
        'p_vector': pv,
        'n_faces': len(faces),
        'count': orbit_size,
        'face_neighbors': face_nbrs,
    }


# -----------------------------------------------------------------------
# New algorithm: incremental star construction with deduplication
# -----------------------------------------------------------------------

def enumerate_cell_types_v2(vertices, central_idx, coords, verbose=True,
                            include_degenerate=False, crystal=None,
                            n_workers=1):
    """Enumerate all Voronoi cell types via incremental star construction.

    Algorithm:
      1. At each degenerate vertex, enumerate ALL resolutions of the full
         point configuration (all equidistant atoms present).
      2. Build vertex_options: for each degenerate vertex, a list of
         (frozenset_of_tets, is_unresolved) pairs.
      3. Incrementally construct global stars one vertex at a time,
         merging each vertex's resolution choices with existing stars
         via frozenset union, and deduplicating identical stars at
         each step.
      4. (Optional) If the crystal structure is provided, use site
         symmetry to group equivalent stars and classify only one
         representative per orbit.

    Note: atoms are NOT removed from the point configuration.  The
    presence of an atom constrains the triangulation space even when
    it is not in the star of the central atom.

    Parameters
    ----------
    vertices : list of VoronoiVertex
    central_idx : int
    coords : ndarray, shape (M, 3)
    verbose : bool
    include_degenerate : bool
        If True, include an unresolved option for each degenerate vertex
        (keeping it as a single higher-valence Voronoi vertex).  This
        enumerates all cell types including those with degenerate vertices.
    crystal : Crystal, optional
        If provided, site symmetry is computed and used to reduce the
        number of stars that need classification.

    Returns
    -------
    cell_types : dict
    """
    generic = [v for v in vertices if not v.is_degenerate]
    degen = [v for v in vertices if v.is_degenerate]

    generic_tets = frozenset(tuple(sorted(v.atom_indices)) for v in generic)

    # ------------------------------------------------------------------
    # Step 1: Build vertex_options for each degenerate vertex
    # ------------------------------------------------------------------
    vertex_options = []

    for v in degen:
        order = [central_idx] + [a for a in v.atom_indices
                                 if a != central_idx]
        pts = coords[order]
        resolutions, _ = enumerate_resolutions(pts, central=0)

        opts = []
        for res in resolutions:
            mapped = frozenset(
                tuple(sorted(order[j] for j in tet))
                for tet in res.star
            )
            opts.append((mapped, False))

        # Add unresolved option for --all-types
        if include_degenerate and resolutions:
            all_local_nbrs = set(range(1, len(order)))
            best_star = None
            for res in resolutions:
                if set(res.neighbors) == all_local_nbrs:
                    best_star = frozenset(
                        tuple(sorted(order[j] for j in tet))
                        for tet in res.star
                    )
                    break
            if best_star is None:
                best_res = max(resolutions, key=lambda r: len(r.neighbors))
                best_star = frozenset(
                    tuple(sorted(order[j] for j in tet))
                    for tet in best_res.star
                )
            opts.append((best_star, True))

        vertex_options.append(opts)

    if verbose:
        n_res = [len(opts) for opts in vertex_options]
        print(f"Degenerate vertices: {len(degen)}")
        print(f"Resolutions per vertex: {n_res}"
              f"{' (includes unresolved)' if include_degenerate else ''}")
        print(f"Generic tetrahedra: {len(generic_tets)}")
        total_unfiltered = prod(n_res) if n_res else 1
        print(f"Unfiltered Cartesian product: {total_unfiltered}")

    # ------------------------------------------------------------------
    # Step 1b: Compute site symmetry (if crystal provided)
    # ------------------------------------------------------------------
    symmetry = None
    if crystal is not None and degen:
        from .symmetry import compute_site_symmetry
        # Compute symmetry on resolved options only (unresolved stars
        # can duplicate resolved stars, confusing the index matching)
        vertex_stars = []
        for vopts in vertex_options:
            stars = [tets for tets, is_unres in vopts if not is_unres]
            vertex_stars.append(stars)
        symmetry = compute_site_symmetry(
            crystal, central_idx, coords, degen, vertex_stars,
            verbose=verbose)
        # Extend res_perms: unresolved option always maps to unresolved
        if symmetry is not None and include_degenerate:
            for gi in range(symmetry.order):
                for vi in range(len(degen)):
                    symmetry.res_perms[gi][vi].append(len(vertex_stars[vi]))

    # ------------------------------------------------------------------
    # Step 1c: Precompute pairwise compatibility for validity pruning
    # ------------------------------------------------------------------
    adj, compat, has_coupling = _precompute_compatibility(
        generic_tets, vertex_options, central_idx)

    # ------------------------------------------------------------------
    # Step 2: Enumerate using the best available strategy
    # ------------------------------------------------------------------
    # Choose enumeration strategy.  Orderly generation uses O(depth)
    # memory (DFS), while incremental construction builds a dictionary
    # of all unique intermediate stars that can blow up memory.
    # Always prefer orderly; when no symmetry was found, build a trivial
    # SiteSymmetry so we can use the orderly DFS path, which has
    # O(depth) memory instead of O(product) memory.
    if symmetry is None and vertex_options:
        from .symmetry import SiteSymmetry
        d = len(vertex_options)
        identity_vperm = list(range(d))
        identity_rperm = [list(range(len(opts))) for opts in vertex_options]
        all_star_atoms = set()
        for tets, _ in ((t, u) for opts in vertex_options for t, u in opts):
            for tet in tets:
                all_star_atoms.update(tet)
        for tet in generic_tets:
            all_star_atoms.update(tet)
        identity_aperm = {a: a for a in all_star_atoms}
        symmetry = SiteSymmetry([identity_vperm], [identity_rperm],
                                [identity_aperm])

    if symmetry is not None:
        # Orderly generation: branch-and-bound DFS producing only canonical
        # combos.  Strictly faster than exhaustive iteration for symmetry
        # groups of moderate order (e.g. 2.4x for FCC with |Oh|=48).
        if n_workers > 1:
            cell_types = _enumerate_orderly_parallel(
                generic_tets, vertex_options, central_idx, symmetry,
                verbose, n_workers, adj, compat, has_coupling)
        else:
            cell_types = _enumerate_orderly(
                generic_tets, vertex_options, central_idx, symmetry,
                verbose, adj, compat, has_coupling)
    elif include_degenerate:
        cell_types = _enumerate_incremental_degenerate(
            generic_tets, vertex_options, central_idx, verbose)
    else:
        cell_types = _enumerate_incremental(
            generic_tets, vertex_options, central_idx, verbose)

    return cell_types


def _enumerate_incremental(generic_tets, vertex_options, central_idx,
                           verbose, symmetry=None):
    """Incremental star construction for the standard (fully resolved) case.

    Builds global stars one vertex at a time, merging each vertex's
    resolution choices via frozenset union, and deduplicating identical
    stars at each step.

    If a SiteSymmetry object is provided, symmetry-equivalent stars are
    grouped after the incremental construction, and only one representative
    per orbit is classified.
    """
    # State: frozenset of tets -> count
    current_stars = {generic_tets: 1}

    for vi, opts in enumerate(vertex_options):
        next_stars = {}
        for star, count in current_stars.items():
            for tets, _ in opts:
                new_star = star | tets
                next_stars[new_star] = next_stars.get(new_star, 0) + count
        current_stars = next_stars
        if verbose:
            print(f"  After vertex {vi+1}/{len(vertex_options)}: "
                  f"{len(current_stars)} unique stars")

    # Group by symmetry equivalence if available
    if symmetry is not None:
        canon_groups = {}  # canonical_star -> (representative_star, total_count)
        for star, count in current_stars.items():
            canon = symmetry.canonical_star(star)
            if canon in canon_groups:
                _, prev_count = canon_groups[canon]
                canon_groups[canon] = (canon_groups[canon][0],
                                       prev_count + count)
            else:
                canon_groups[canon] = (star, count)
        stars_to_classify = {s: c for s, c in canon_groups.values()}
        if verbose:
            print(f"  Symmetry reduction: {len(current_stars)} -> "
                  f"{len(stars_to_classify)} orbits "
                  f"(symmetry group order {symmetry.order})")
    else:
        stars_to_classify = current_stars

    # Classify each unique star (or orbit representative)
    cell_types = {}
    for star, count in stars_to_classify.items():
        global_star = list(star)

        faces, face_nbrs = star_to_faces(global_star,
                                         central=central_idx)
        if not faces:
            continue

        faces = orient_faces(faces)
        pv = p_vector(faces)
        wv = weinberg_vector(faces)

        if wv is None:
            continue

        if wv not in cell_types:
            cell_types[wv] = {
                'p_vector': pv,
                'n_faces': len(faces),
                'count': count,
                'face_neighbors': face_nbrs,
            }
        else:
            cell_types[wv]['count'] += count

    if verbose:
        print(f"\nDone: {len(current_stars)} unique stars, "
              f"{len(cell_types)} distinct cell types")

    return cell_types


def _enumerate_orderly(generic_tets, vertex_options, central_idx,
                       symmetry, verbose,
                       adj=None, compat=None, has_coupling=None):
    """Enumerate using orderly generation: branch-and-bound DFS.

    Instead of iterating over all combinations and checking canonicality,
    generates only canonical representatives by pruning non-canonical
    branches early.  This extends the reach of symmetry reduction to
    product spaces too large for exhaustive iteration.

    Handles both resolved and unresolved (degenerate) options.
    """
    n_res = [len(opts) for opts in vertex_options]
    total = prod(n_res) if n_res else 1

    # Validity pruning for coupled vertices
    if has_coupling is None:
        adj, compat, has_coupling = _precompute_compatibility(
            generic_tets, vertex_options, central_idx)
    prune_fn = _make_prune_fn(adj, compat) if has_coupling else None
    if verbose and has_coupling:
        print(f"  Validity pruning: {sum(len(a) for a in adj) // 2} "
              f"adjacent vertex pairs")

    cell_types = {}
    n_canonical = 0

    for combo, orbit_size in symmetry.orderly_generate(
            n_res, prune=prune_fn):
        n_canonical += 1

        result = _classify_combo(combo, orbit_size, generic_tets,
                                 vertex_options, central_idx)
        if result is not None:
            wv, info = result
            if wv not in cell_types:
                cell_types[wv] = info
            else:
                cell_types[wv]['count'] += info['count']

        if verbose and n_canonical % 500 == 0:
            print(f"  {n_canonical} canonical, "
                  f"{len(cell_types)} types so far")

    if verbose:
        print(f"\nDone: {n_canonical} canonical out of {total} "
              f"(symmetry order {symmetry.order}, "
              f"orderly generation), "
              f"{len(cell_types)} distinct cell types")

    return cell_types


def _classify_subtree(args):
    """Worker function for parallel orderly generation.

    Processes one subtree of the orderly DFS and returns a partial
    cell_types dict with the count of canonical combos processed.
    Must be a top-level function for multiprocessing pickling.
    """
    symmetry, n_res, prefix, active, generic_tets, vertex_options, \
        central_idx, adj, compat, has_coupling = args

    prune_fn = _make_prune_fn(adj, compat) if has_coupling else None

    cell_types = {}
    n_canonical = 0

    for combo, orbit_size in symmetry.orderly_generate(
            n_res, prefix=prefix, active=active, prune=prune_fn):
        n_canonical += 1
        result = _classify_combo(combo, orbit_size, generic_tets,
                                 vertex_options, central_idx)
        if result is not None:
            wv, info = result
            if wv not in cell_types:
                cell_types[wv] = info
            else:
                cell_types[wv]['count'] += info['count']

    return cell_types, n_canonical


def _enumerate_orderly_parallel(generic_tets, vertex_options, central_idx,
                                symmetry, verbose, n_workers,
                                adj=None, compat=None, has_coupling=None):
    """Parallel orderly generation: split DFS into subtrees across workers.

    The orderly DFS tree is expanded to a configurable split depth,
    producing independent (prefix, active_set) subtree specifications.
    Each subtree is dispatched to a worker process that runs the
    remaining DFS and classifies canonical combos independently.
    Results are merged at the end.
    """
    import multiprocessing

    n_res = [len(opts) for opts in vertex_options]
    total = prod(n_res) if n_res else 1
    d = len(n_res)

    # Validity pruning for coupled vertices
    if has_coupling is None:
        adj, compat, has_coupling = _precompute_compatibility(
            generic_tets, vertex_options, central_idx)
    if verbose and has_coupling:
        print(f"  Validity pruning: {sum(len(a) for a in adj) // 2} "
              f"adjacent vertex pairs")

    # Choose split depth: want >= 4*n_workers subtrees for load balancing
    split_depth = 0
    n_subtrees_est = 1
    while split_depth < d and n_subtrees_est < 4 * n_workers:
        n_subtrees_est *= n_res[split_depth]
        split_depth += 1

    subtrees = list(symmetry.generate_subtrees(n_res, split_depth))

    if verbose:
        print(f"  Parallel: {n_workers} workers, split depth {split_depth}, "
              f"{len(subtrees)} subtrees (from {n_subtrees_est} candidates)")

    worker_args = [
        (symmetry, n_res, prefix, active, generic_tets, vertex_options,
         central_idx, adj, compat, has_coupling)
        for prefix, active in subtrees
    ]

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_classify_subtree, worker_args)

    # Merge results
    cell_types = {}
    n_canonical = 0
    for partial_types, partial_count in results:
        n_canonical += partial_count
        for wv, info in partial_types.items():
            if wv not in cell_types:
                cell_types[wv] = info
            else:
                cell_types[wv]['count'] += info['count']

    if verbose:
        print(f"\nDone: {n_canonical} canonical out of {total} "
              f"(symmetry order {symmetry.order}, "
              f"parallel orderly, {n_workers} workers), "
              f"{len(cell_types)} distinct cell types")

    return cell_types


def _enumerate_incremental_degenerate(generic_tets, vertex_options,
                                      central_idx, verbose,
                                      symmetry=None):
    """Incremental construction with unresolved vertex support.

    State includes both the frozenset of tets and a frozenset of
    tet-group frozensets (for unresolved degenerate vertices whose
    tets should be treated as a single Voronoi vertex).

    If a SiteSymmetry object is provided, symmetry-equivalent states are
    grouped after the incremental construction.
    """
    # State: (frozenset of tets, frozenset of frozenset groups) -> count
    initial_state = (generic_tets, frozenset())
    current = {initial_state: 1}

    for vi, opts in enumerate(vertex_options):
        next_states = {}
        for (star, groups), count in current.items():
            for tets, is_unresolved in opts:
                new_star = star | tets
                if is_unresolved:
                    new_groups = groups | {tets}
                else:
                    new_groups = groups
                key = (new_star, new_groups)
                next_states[key] = next_states.get(key, 0) + count
        current = next_states
        if verbose:
            print(f"  After vertex {vi+1}/{len(vertex_options)}: "
                  f"{len(current)} unique states")

    # Group by symmetry equivalence if available
    # (use star-only canonical form; group structure is determined by star)
    if symmetry is not None:
        canon_groups = {}
        for (star, groups), count in current.items():
            canon = symmetry.canonical_star(star)
            if canon in canon_groups:
                prev_state, prev_count = canon_groups[canon]
                canon_groups[canon] = (prev_state, prev_count + count)
            else:
                canon_groups[canon] = ((star, groups), count)
        states_to_classify = {s: c for s, c in canon_groups.values()}
        if verbose:
            print(f"  Symmetry reduction: {len(current)} -> "
                  f"{len(states_to_classify)} orbits "
                  f"(symmetry group order {symmetry.order})")
    else:
        states_to_classify = current

    # Classify each unique state (or orbit representative)
    cell_types = {}
    for (star, groups), count in states_to_classify.items():
        global_star = list(star)

        tet_groups = None
        if groups:
            tet_groups = {}
            next_gid = 0
            # Assign individual group IDs to non-grouped tets
            for t in star:
                grouped = False
                for g in groups:
                    if t in g:
                        grouped = True
                        break
                if not grouped and t not in tet_groups:
                    tet_groups[t] = next_gid
                    next_gid += 1
            # Assign shared group IDs to grouped tets
            for g in groups:
                gid = next_gid
                next_gid += 1
                for t in g:
                    if t not in tet_groups:
                        tet_groups[t] = gid

        faces, face_nbrs = star_to_faces(global_star,
                                         central=central_idx,
                                         tet_groups=tet_groups)
        if not faces:
            continue

        faces = orient_faces(faces)
        pv = p_vector(faces)
        wv = weinberg_vector(faces)

        if wv is None:
            continue

        if wv not in cell_types:
            cell_types[wv] = {
                'p_vector': pv,
                'n_faces': len(faces),
                'count': count,
                'face_neighbors': face_nbrs,
            }
        else:
            cell_types[wv]['count'] += count

    if verbose:
        print(f"\nDone: {len(current)} unique states, "
              f"{len(cell_types)} distinct cell types")

    return cell_types


# -----------------------------------------------------------------------
# Legacy algorithm: Cartesian product of pre-computed resolutions
# -----------------------------------------------------------------------

def enumerate_cell_types(vertex_resolutions, generic_tets, central_idx,
                         verbose=True):
    """Enumerate all Voronoi cell types from per-vertex resolutions.

    LEGACY: iterates over the Cartesian product of all resolution types.
    Use enumerate_cell_types_v2() for the improved algorithm.

    Parameters
    ----------
    vertex_resolutions : list of (list_of_Resolution, atom_map)
    generic_tets : list of tuple
    central_idx : int
    verbose : bool

    Returns
    -------
    cell_types : dict
    """
    n_verts = len(vertex_resolutions)
    n_res = [len(res) for res, _ in vertex_resolutions]
    total = prod(n_res)

    if verbose:
        expr = " x ".join(str(n) for n in n_res)
        print(f"Degenerate vertices: {n_verts}")
        print(f"Resolutions per vertex: {n_res}")
        print(f"Total combinations: {expr} = {total}")

    cell_types = {}
    done = 0

    for combo in product(*[range(n) for n in n_res]):
        global_star = list(generic_tets)
        for vi, ri in enumerate(combo):
            resolutions, atom_map = vertex_resolutions[vi]
            for local_tet in resolutions[ri].star:
                global_tet = tuple(sorted(atom_map[j] for j in local_tet))
                global_star.append(global_tet)

        faces, face_nbrs = star_to_faces(global_star, central=central_idx)
        if not faces:
            done += 1
            continue

        faces = orient_faces(faces)
        pv = p_vector(faces)
        wv = weinberg_vector(faces)

        if wv is None:
            done += 1
            continue

        if wv not in cell_types:
            cell_types[wv] = {
                'p_vector': pv,
                'n_faces': len(faces),
                'count': 1,
                'example_combo': combo,
                'face_neighbors': face_nbrs,
            }
        else:
            cell_types[wv]['count'] += 1

        done += 1
        if verbose and done % 10000 == 0:
            print(f"  {done}/{total}: {len(cell_types)} distinct types so far")

    if verbose:
        print(f"  {done}/{total}: {len(cell_types)} distinct types (done)")

    return cell_types


def summarize_cell_types(cell_types):
    """Print a summary of the enumerated cell types."""
    print(f"\nDistinct Voronoi cell types: {len(cell_types)}")

    by_pv = defaultdict(list)
    for wv, info in cell_types.items():
        by_pv[info['p_vector']].append((wv, info))

    print(f"Distinct p-vectors: {len(by_pv)}")

    for pv in sorted(by_pv):
        entries = by_pv[pv]
        total_count = sum(info['count'] for _, info in entries)
        n_faces = entries[0][1]['n_faces']
        print(f"\n  p-vector {pv}  ({n_faces} faces)")
        print(f"    {len(entries)} Weinberg type(s), "
              f"{total_count} combination(s) total")
        for wv, info in entries:
            print(f"      WV length {len(wv)}, "
                  f"realized by {info['count']} combo(s)")
