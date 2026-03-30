"""
Site symmetry for Voronoi cell enumeration.

Computes the point group of a crystallographic site and its action on
degenerate Voronoi vertices, resolution types, and supercell atoms.
Used to reduce the enumeration space by identifying symmetry-equivalent
resolution combinations.

Requires pymatgen for space group analysis.
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class SiteSymmetry:
    """Site symmetry group and its action on Voronoi structures.

    Attributes
    ----------
    order : int
        Number of symmetry operations.
    vertex_perms : list of list of int
        For each group element, the permutation of degenerate vertex indices.
    res_perms : list of list of list of int
        For each group element g and vertex vi, res_perms[g][vi][ri] is
        the resolution index at vertex vperm[vi] corresponding to
        resolution ri at vertex vi.
    atom_perms : list of dict
        For each group element, mapping from supercell atom index to
        supercell atom index.
    """

    def __init__(self, vertex_perms, res_perms, atom_perms):
        self.vertex_perms = vertex_perms
        self.res_perms = res_perms
        self.atom_perms = atom_perms
        self.order = len(vertex_perms)

    def canonical_star(self, star):
        """Compute the canonical form of a star under the symmetry group.

        Parameters
        ----------
        star : frozenset of tuple
            A global star (frozenset of sorted tet tuples with supercell
            atom indices).

        Returns
        -------
        canonical : tuple of tuple
            The lexicographically smallest image of the star under all
            group elements, as a sorted tuple of sorted tet tuples.
        """
        best = tuple(sorted(star))
        for atom_perm in self.atom_perms:
            mapped = tuple(sorted(
                tuple(sorted(atom_perm[a] for a in tet))
                for tet in star
            ))
            if mapped < best:
                best = mapped
        return best

    def apply_combo(self, combo, gi):
        """Apply group element gi to a resolution combination.

        Parameters
        ----------
        combo : tuple of int
            Resolution indices (r0, r1, ..., r_{d-1}).
        gi : int
            Index of the group element.

        Returns
        -------
        mapped : tuple of int
            The transformed combination.
        """
        vperm = self.vertex_perms[gi]
        rperm = self.res_perms[gi]
        d = len(combo)
        # g maps vertex vi -> vperm[vi], and resolution combo[vi] at vi
        # to rperm[vi][combo[vi]] at vperm[vi].
        # So the new combo at vertex vj is rperm[vi][combo[vi]]
        # where vi = vperm_inv[vj].
        result = [0] * d
        for vi in range(d):
            vj = vperm[vi]
            result[vj] = rperm[vi][combo[vi]]
        return tuple(result)

    def canonical_combo(self, combo):
        """Compute the canonical form and orbit size of a combination.

        Parameters
        ----------
        combo : tuple of int
            Resolution indices.

        Returns
        -------
        canonical : tuple of int
            Lexicographically smallest image under the group.
        orbit_size : int
            Size of the orbit (|G| / |stabilizer|).
        """
        best = combo
        stab = 0
        for gi in range(self.order):
            mapped = self.apply_combo(combo, gi)
            if mapped < best:
                best = mapped
            if mapped == combo:
                stab += 1
        return best, self.order // stab

    def _vperm_inv(self, d):
        """Precompute inverse vertex permutations."""
        vperm_inv = []
        for gi in range(self.order):
            inv = [0] * d
            for i in range(d):
                inv[self.vertex_perms[gi][i]] = i
            vperm_inv.append(inv)
        return vperm_inv

    def orderly_generate(self, n_res, prefix=(), active=None, prune=None):
        """Generate canonical combinations via branch-and-bound DFS.

        Builds resolution combinations position by position, pruning any
        partial assignment where a group element maps it to a
        lexicographically smaller image.  Only canonical representatives
        (lex-min in their orbit) reach the leaves.

        For each active group element, we track a *resume position*: the
        first position whose comparison hasn't been resolved yet.
        Positions before it are known to match the combo exactly.  When
        a new position is assigned, previously unresolved comparisons may
        become resolvable (their source vertex is now assigned).

        Parameters
        ----------
        n_res : list of int
            Number of resolution options at each vertex.
        prefix : tuple of int, optional
            Already-assigned positions (for resuming from a subtree).
        active : list of (int, int), optional
            Active group elements with resume positions.  If None,
            starts with all group elements.
        prune : callable(combo_list, depth) -> bool, optional
            Called after canonicality check passes.  combo_list is the
            mutable internal list (read-only!); depth is the number of
            assigned positions.  Return True to prune the subtree.
            Used for validity pruning of geometrically incompatible
            resolution combinations.

        Yields
        ------
        combo : tuple of int
            Canonical resolution combination.
        orbit_size : int
            Size of the orbit (|G| / |stabilizer|).
        """
        d = len(n_res)
        if d == 0:
            yield ((), self.order)
            return

        start = len(prefix)
        if active is None:
            active = [(gi, 0) for gi in range(self.order)]

        vperm_inv = self._vperm_inv(d)
        combo = list(prefix) + [0] * (d - start)
        rperm = self.res_perms

        def _dfs(depth, active):
            if depth == d:
                yield (tuple(combo), self.order // len(active))
                return

            for r in range(n_res[depth]):
                combo[depth] = r
                new_active = []
                pruned = False

                for gi, resume in active:
                    status = None
                    new_resume = depth + 1

                    for j in range(resume, depth + 1):
                        src = vperm_inv[gi][j]
                        if src > depth:
                            status = 'undecided'
                            new_resume = j
                            break
                        img_j = rperm[gi][src][combo[src]]
                        cj = combo[j]
                        if img_j < cj:
                            status = 'smaller'
                            break
                        elif img_j > cj:
                            status = 'larger'
                            break

                    if status is None:
                        status = 'equal'

                    if status == 'smaller':
                        pruned = True
                        break
                    elif status != 'larger':
                        new_active.append((gi, new_resume))

                if not pruned:
                    if prune is not None and prune(combo, depth + 1):
                        continue
                    yield from _dfs(depth + 1, new_active)

        yield from _dfs(start, active)

    def generate_subtrees(self, n_res, split_depth):
        """Generate independent subtree specs for parallel processing.

        Runs the orderly DFS to ``split_depth``, yielding the prefix
        and active set for each surviving branch.  Each (prefix, active)
        pair defines an independent subtree that can be processed by a
        separate worker via ``orderly_generate(n_res, prefix, active)``.

        Parameters
        ----------
        n_res : list of int
            Number of resolution options at each vertex.
        split_depth : int
            Depth at which to split.

        Yields
        ------
        prefix : tuple of int
            Assigned positions (length ``split_depth``).
        active : list of (int, int)
            Active group elements with resume positions.
        """
        d = len(n_res)
        split_depth = min(split_depth, d)

        vperm_inv = self._vperm_inv(d)
        combo = [0] * d
        rperm = self.res_perms

        def _dfs(depth, active):
            if depth == split_depth:
                yield (tuple(combo[:split_depth]), list(active))
                return

            for r in range(n_res[depth]):
                combo[depth] = r
                new_active = []
                pruned = False

                for gi, resume in active:
                    status = None
                    new_resume = depth + 1

                    for j in range(resume, depth + 1):
                        src = vperm_inv[gi][j]
                        if src > depth:
                            status = 'undecided'
                            new_resume = j
                            break
                        img_j = rperm[gi][src][combo[src]]
                        cj = combo[j]
                        if img_j < cj:
                            status = 'smaller'
                            break
                        elif img_j > cj:
                            status = 'larger'
                            break

                    if status is None:
                        status = 'equal'

                    if status == 'smaller':
                        pruned = True
                        break
                    elif status != 'larger':
                        new_active.append((gi, new_resume))

                if not pruned:
                    yield from _dfs(depth + 1, new_active)

        yield from _dfs(0, [(gi, 0) for gi in range(self.order)])


def compute_site_symmetry(crystal, central_idx, coords, degen_vertices,
                          vertex_stars, verbose=False):
    """Compute site symmetry from a crystal structure.

    Uses pymatgen's SpacegroupAnalyzer to get symmetry operations,
    then computes their action on degenerate vertices and resolution
    types.

    Parameters
    ----------
    crystal : Crystal
        The crystal structure.
    central_idx : int
        Supercell index of the central atom.
    coords : ndarray, shape (M, 3)
        Supercell atom coordinates.
    degen_vertices : list of VoronoiVertex
        Degenerate Voronoi vertices.
    vertex_stars : list of list of frozenset
        For each degenerate vertex, the list of resolution stars
        (each a frozenset of sorted tet tuples in supercell indices).
    verbose : bool

    Returns
    -------
    SiteSymmetry or None
        None if pymatgen is not available or symmetry computation fails.
    """
    if not degen_vertices:
        return None

    # Build pymatgen Structure
    lattice = Lattice(crystal.lattice)
    species = crystal.species if crystal.species else ['X'] * crystal.n_atoms
    struct = Structure(lattice, species, crystal.frac_coords)

    sga = SpacegroupAnalyzer(struct)
    ops = sga.get_symmetry_operations(cartesian=True)

    # Get degenerate vertex positions
    d = len(degen_vertices)
    degen_pos = np.array([v.position for v in degen_vertices])

    # Collect all supercell atom indices appearing in any star
    all_star_atoms = {central_idx}
    for v in degen_vertices:
        all_star_atoms.update(v.atom_indices)
    all_star_atoms = sorted(all_star_atoms)

    # For each symmetry operation, check if it:
    # 1. Fixes the central atom
    # 2. Permutes degenerate vertices among themselves
    # 3. Induces a valid permutation of resolution types
    valid_vertex_perms = []
    valid_res_perms = []
    valid_atom_perms = []

    central_pos = coords[central_idx]

    for op in ops:
        R = op.rotation_matrix

        # Check: does this operation fix the central atom?
        rotated_central = R @ central_pos
        if np.linalg.norm(rotated_central - central_pos) > 1e-6:
            continue

        # Compute vertex permutation
        vperm = []
        ok = True
        for pos in degen_pos:
            rotated = R @ pos
            dists = np.linalg.norm(degen_pos - rotated, axis=1)
            j = int(np.argmin(dists))
            if dists[j] > 1e-6:
                ok = False
                break
            vperm.append(j)
        if not ok:
            continue

        # Compute atom permutation for star atoms
        atom_perm = {}
        for ai in all_star_atoms:
            rotated = R @ coords[ai]
            dists = np.linalg.norm(coords - rotated, axis=1)
            best = int(np.argmin(dists))
            if dists[best] > 1e-6:
                ok = False
                break
            atom_perm[ai] = best
        if not ok:
            continue

        # Compute resolution type permutation
        rperm = []  # rperm[vi][ri] = rj at vperm[vi]
        for vi in range(d):
            vj = vperm[vi]
            ri_to_rj = []
            for ri, star_i in enumerate(vertex_stars[vi]):
                mapped_star = frozenset(
                    tuple(sorted(atom_perm[a] for a in tet))
                    for tet in star_i
                )
                found = False
                for rj, star_j in enumerate(vertex_stars[vj]):
                    if mapped_star == star_j:
                        ri_to_rj.append(rj)
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if not ok:
                break
            rperm.append(ri_to_rj)
        if not ok:
            continue

        valid_vertex_perms.append(vperm)
        valid_res_perms.append(rperm)
        valid_atom_perms.append(atom_perm)

    if not valid_vertex_perms:
        if verbose:
            print("No valid symmetry operations found")
        return None

    if verbose:
        print(f"Site symmetry group order: {len(valid_vertex_perms)}")

    return SiteSymmetry(valid_vertex_perms, valid_res_perms,
                        valid_atom_perms)
