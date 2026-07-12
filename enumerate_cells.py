"""
Enumerate all possible Voronoi cell types for a crystal under perturbation.

General-purpose script: works for any crystal structure.
Usage:
    python enumerate_cells.py fcc
    python enumerate_cells.py a15 --atoms 0 2
    python enumerate_cells.py structure.cif
    python enumerate_cells.py POSCAR --atoms 0
    python enumerate_cells.py structure.cif --max-memory 8
"""

import os
import sys
import time
import resource
from math import prod
import numpy as np
from voronoi_enumerate.crystal import Crystal
from voronoi_enumerate.voronoi import analyze_voronoi
from voronoi_enumerate.resolution import enumerate_resolutions
from voronoi_enumerate.combine import (
    enumerate_cell_types, enumerate_cell_types_v2,
    summarize_cell_types, classify_neighbors,
)
from voronoi_enumerate.filter import write_filter_file


SUMMARY_COLUMNS = [
    'name', 'mode', 'status', 'runtime_sec',
    'n_atoms_basis', 'n_sites_enumerated',
    'n_wv_total', 'max_family',
    'min_gap_delta', 'n_images', 'near_gap_threshold',
    'symmetrized', 'notes',
]


def _vertex_min_gap_rel(v, coords):
    """Relative distance gap (d_next - r) / r at this vertex.

    The nearest non-equidistant atom sits (d_next - r) / r farther
    from the vertex than the equidistant set -- the "X% farther"
    quantity quoted in the papers (omega-Ti 0.34, MgCu2 0.50), and
    the same convention as --near-gap-threshold: an atom is treated
    as borderline when its relative gap is below tau, so raising tau
    above the reported minimum gap brings the nearest such atom into
    the enumeration.

    The conservative safety margin against simultaneous two-sided
    motion (the atom and the central atom each moving sigma toward
    each other) is half this value: sigma/r = gap/2.
    """
    dists = np.linalg.norm(coords - v.position, axis=1)
    eq = set(v.atom_indices)
    non_eq = [d for i, d in enumerate(dists) if i not in eq]
    if not non_eq:
        return float('inf')
    return (min(non_eq) - v.circumradius) / v.circumradius


def _append_summary_row(path, row):
    """Append a row to the summary TSV, creating header if file is new."""
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a') as f:
        if new_file:
            f.write('\t'.join(SUMMARY_COLUMNS) + '\n')
        f.write('\t'.join(str(row.get(c, '')) for c in SUMMARY_COLUMNS) + '\n')


def _a15(a=1.0):
    """A15 (Cr3Si-type): Pm-3n, 8 atoms/cell.
    2a sites (Si/Sn): (0,0,0), (1/2,1/2,1/2)
    6c sites (Cr/Nb): (1/4,0,1/2) + permutations
    """
    lattice = a * np.eye(3)
    frac_coords = [
        [0.0, 0.0, 0.0],          # 2a
        [0.5, 0.5, 0.5],          # 2a
        [0.25, 0.0, 0.5],         # 6c
        [0.75, 0.0, 0.5],         # 6c
        [0.5, 0.25, 0.0],         # 6c
        [0.5, 0.75, 0.0],         # 6c
        [0.0, 0.5, 0.25],         # 6c
        [0.0, 0.5, 0.75],         # 6c
    ]
    species = ['A']*2 + ['B']*6
    return Crystal(lattice, frac_coords, species)


def _fluorite(a=1.0):
    """Fluorite (CaF2): Fm-3m, primitive FCC cell, 3 atoms.
    Ca at (0,0,0), F at (1/4,1/4,1/4) and (3/4,3/4,3/4).
    """
    lattice = (a / 2) * np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
    ], dtype=float)
    frac_coords = [
        [0.0, 0.0, 0.0],       # Ca
        [0.25, 0.25, 0.25],     # F
        [0.75, 0.75, 0.75],     # F
    ]
    return Crystal(lattice, frac_coords, ['Ca', 'F', 'F'])


def _omega(a=1.0):
    """Omega (ω) phase: P6/mmm, 3 atoms/cell.
    1a: (0,0,0), 2d: (1/3,2/3,1/2), (2/3,1/3,1/2).
    Ideal c/a = sqrt(3/8).
    """
    c = a * np.sqrt(3.0 / 8.0)
    lattice = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, c],
    ])
    frac_coords = [
        [0.0, 0.0, 0.0],
        [1.0/3, 2.0/3, 0.5],
        [2.0/3, 1.0/3, 0.5],
    ]
    return Crystal(lattice, frac_coords, ['A', 'A', 'A'])


MAX_MEMORY_GB = 4  # default memory limit (GB)


def set_memory_limit(gb):
    """Set a hard address-space limit.  Allocations beyond this raise MemoryError."""
    limit_bytes = int(gb * 1024 ** 3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, resource.error) as e:
        print(f"Warning: could not set memory limit ({e}). "
              f"Memory will not be capped.", file=sys.stderr)


CRYSTALS = {
    'fcc': lambda: Crystal.fcc(a=1.0),
    'bcc': lambda: Crystal.bcc(a=1.0),
    'sc':  lambda: Crystal.sc(a=1.0),
    'hcp': lambda: Crystal.hcp(a=1.0),
    'a15': lambda: _a15(a=1.0),
    'fluorite': lambda: _fluorite(a=1.0),
    'omega': lambda: _omega(a=1.0),
}


def run(name, atom_indices=None, legacy=False, all_types=False,
        n_workers=1, near_gap_threshold=None, primary_only=False,
        n_images=3, summary_out=None, symmetrized=False, notes='',
        filter_dir='.'):
    t0 = time.time()

    cryst = CRYSTALS[name]()
    print(f"Crystal: {name}")
    print(f"Atoms per unit cell: {cryst.n_atoms}")
    print(f"Species: {cryst.species}")
    print(f"Lattice:\n{cryst.lattice}\n")
    print(f"Supercell images: +/-{n_images} in each direction")

    if atom_indices is None:
        atom_indices = list(range(cryst.n_atoms))

    # Track the smallest relative gap across every vertex of every atom;
    # this is the limiting near-equidistant scale for the filter.
    min_gap_delta = float('inf')

    # Collect cell types from all atoms for filter file output
    # Each entry is (atom_index, cell_types_dict)
    per_atom_types = []

    for atom_idx in atom_indices:
        print("=" * 65)
        print(f"Atom {atom_idx}: Voronoi analysis")
        print("=" * 65)
        vertices, central_idx, coords, images = analyze_voronoi(
            cryst, atom_index=atom_idx,
            n_images=n_images,
            near_gap_threshold=near_gap_threshold,
        )

        for v in vertices:
            gap = _vertex_min_gap_rel(v, coords)
            if gap < min_gap_delta:
                min_gap_delta = gap

        degen = [v for v in vertices if v.is_degenerate]
        generic = [v for v in vertices if not v.is_degenerate]
        print(f"Central atom: supercell index {central_idx}")
        print(f"Voronoi vertices: {len(vertices)} "
              f"({len(generic)} generic, {len(degen)} degenerate)")

        if not degen:
            print("No degenerate vertices — Voronoi cell is unique "
                  "under perturbation.")
            # Build the single cell type from the full Delaunay star
            all_tets = []
            for v in vertices:
                for t in v.delaunay_tets:
                    if t not in all_tets:
                        all_tets.append(t)
            from voronoi_enumerate.cell import star_to_faces, orient_faces
            from voronoi_enumerate.weinberg import weinberg_vector, p_vector
            faces, face_nbrs = star_to_faces(all_tets, central=central_idx)
            faces = orient_faces(faces)
            pv = p_vector(faces)
            wv = weinberg_vector(faces)
            print(f"Single cell type: {len(faces)} faces, p-vector {pv}")
            print(f"Weinberg vector length: {len(wv)}")
            per_atom_types.append((atom_idx, {wv: {
                'p_vector': pv,
                'n_faces': len(faces),
                'count': 1,
                'face_neighbors': face_nbrs,
            }}))
            continue

        # ---------------------------------------------------------------
        # New algorithm (default): neighbor-subset decomposition
        # ---------------------------------------------------------------
        if not legacy:
            print(f"\n{'=' * 65}")
            print(f"Atom {atom_idx}: Neighbor-subset enumeration")
            print("=" * 65)

            cell_types = enumerate_cell_types_v2(
                vertices, central_idx, coords, verbose=True,
                include_degenerate=all_types,
                crystal=cryst,
                n_workers=n_workers,
                primary_only=primary_only,
            )

            print(f"\n{'=' * 65}")
            print(f"Atom {atom_idx}: Results")
            print("=" * 65)
            summarize_cell_types(cell_types)
            per_atom_types.append((atom_idx, cell_types))
            continue

        # ---------------------------------------------------------------
        # Legacy algorithm: pre-compute resolutions, Cartesian product
        # ---------------------------------------------------------------
        generic_tets = [tuple(sorted(v.atom_indices)) for v in generic]
        print(f"Generic tetrahedra: {len(generic_tets)}")

        print(f"\n{'=' * 65}")
        print(f"Atom {atom_idx}: Per-vertex resolution enumeration (legacy)")
        print("=" * 65)

        vertex_resolutions = []
        for i, v in enumerate(degen):
            pts, c_local, atom_map = v.point_config(coords, central_idx)
            print(f"\nVertex {i} at ({v.position[0]:.3f}, "
                  f"{v.position[1]:.3f}, {v.position[2]:.3f}): "
                  f"{v.n_equidistant} equidistant atoms")
            resolutions, _ = enumerate_resolutions(
                pts, central=0, verbose=False
            )
            print(f"  {len(resolutions)} resolution types found")
            for j, res in enumerate(resolutions):
                print(f"    Type {j}: {len(res.star)} tets, "
                      f"neighbors={res.neighbors}, "
                      f"valences={res.face_valences}")
            vertex_resolutions.append((resolutions, atom_map))

        # Combinatorial enumeration
        n_res = [len(res) for res, _ in vertex_resolutions]
        total = prod(n_res)
        print(f"\n{'=' * 65}")
        print(f"Atom {atom_idx}: Combinatorial enumeration "
              f"({total} combinations)")
        print("=" * 65)

        cell_types = enumerate_cell_types(
            vertex_resolutions, generic_tets, central_idx, verbose=True
        )

        print(f"\n{'=' * 65}")
        print(f"Atom {atom_idx}: Results")
        print("=" * 65)
        summarize_cell_types(cell_types)
        per_atom_types.append((atom_idx, cell_types))

    # Write VoroTop filter file
    if filter_dir and filter_dir != '.':
        os.makedirs(filter_dir, exist_ok=True)
    filter_path = os.path.join(filter_dir, f"{name}.filter")
    n_wv, n_groups = write_filter_file(
        filter_path, name, per_atom_types, species=cryst.species,
        min_gap_rel=(min_gap_delta if min_gap_delta != float('inf')
                     else None),
    )
    print(f"\nFilter file written: {filter_path} "
          f"({n_groups} type(s), {n_wv} Weinberg vectors)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f} seconds")

    if summary_out is not None:
        if primary_only:
            mode = 'primary'
        elif all_types:
            mode = 'all'
        else:
            mode = 'default'
        # Largest cell-type family across all enumerated sites
        max_family = 0
        for _, cts in per_atom_types:
            if cts and len(cts) > max_family:
                max_family = len(cts)
        row = {
            'name': name,
            'mode': mode,
            'status': 'OK',
            'runtime_sec': f'{elapsed:.2f}',
            'n_atoms_basis': cryst.n_atoms,
            'n_sites_enumerated': len(atom_indices),
            'n_wv_total': n_wv,
            'max_family': max_family,
            'min_gap_delta': (f'{min_gap_delta:.6e}'
                              if min_gap_delta != float('inf') else 'inf'),
            'n_images': n_images,
            'near_gap_threshold': (near_gap_threshold
                                    if near_gap_threshold is not None
                                    else ''),
            'symmetrized': 'Y' if symmetrized else 'N',
            'notes': notes,
        }
        _append_summary_row(summary_out, row)
        print(f"Summary row appended to {summary_out}")


if __name__ == '__main__':
    import os

    name = sys.argv[1] if len(sys.argv) > 1 else 'fcc'

    # Optional: --atoms 0 2 to run only specific atom indices
    atom_indices = None
    if '--atoms' in sys.argv:
        idx = sys.argv.index('--atoms')
        atom_indices = []
        for a in sys.argv[idx+1:]:
            try:
                atom_indices.append(int(a))
            except ValueError:
                break

    legacy = '--legacy' in sys.argv
    all_types = '--all-types' in sys.argv
    primary_only = '--primary' in sys.argv

    n_workers = 1
    if '-j' in sys.argv:
        idx = sys.argv.index('-j')
        if idx + 1 < len(sys.argv):
            n_workers = int(sys.argv[idx + 1])

    near_gap_threshold = None
    if '--near-gap-threshold' in sys.argv:
        idx = sys.argv.index('--near-gap-threshold')
        if idx + 1 < len(sys.argv):
            near_gap_threshold = float(sys.argv[idx + 1])

    n_images = 3
    if '--n-images' in sys.argv:
        idx = sys.argv.index('--n-images')
        if idx + 1 < len(sys.argv):
            n_images = int(sys.argv[idx + 1])

    summary_out = None
    if '--summary-out' in sys.argv:
        idx = sys.argv.index('--summary-out')
        if idx + 1 < len(sys.argv):
            summary_out = sys.argv[idx + 1]

    symmetrize = '--symmetrize' in sys.argv
    symprec = 0.01
    if '--symprec' in sys.argv:
        idx = sys.argv.index('--symprec')
        if idx + 1 < len(sys.argv):
            symprec = float(sys.argv[idx + 1])

    max_memory = MAX_MEMORY_GB
    if '--max-memory' in sys.argv:
        idx = sys.argv.index('--max-memory')
        if idx + 1 < len(sys.argv):
            max_memory = float(sys.argv[idx + 1])
    set_memory_limit(max_memory)

    def _memory_abort(exc_type, exc, tb):
        if exc_type is MemoryError:
            print(f"\nERROR: memory limit exceeded "
                  f"(--max-memory {max_memory} GB); aborting.",
                  file=sys.stderr)
            sys.exit(1)
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _memory_abort

    filter_dir = '.'
    if '--filter-dir' in sys.argv:
        idx = sys.argv.index('--filter-dir')
        if idx + 1 < len(sys.argv):
            filter_dir = sys.argv[idx + 1]

    if name in CRYSTALS:
        if symmetrize:
            cryst = CRYSTALS[name]().symmetrized(symprec=symprec)
            CRYSTALS[name] = lambda c=cryst: c
        run(name, atom_indices=atom_indices, legacy=legacy,
            all_types=all_types, n_workers=n_workers,
            near_gap_threshold=near_gap_threshold,
            primary_only=primary_only, n_images=n_images,
            summary_out=summary_out, symmetrized=symmetrize,
            filter_dir=filter_dir)
    elif os.path.isfile(name):
        # Load crystal structure from CIF, POSCAR, or other file
        cryst = Crystal.from_file(name)
        n_before = cryst.n_atoms
        if symmetrize:
            cryst = cryst.symmetrized(symprec=symprec)
            print(f"Symmetrized at symprec={symprec}: "
                  f"{n_before} -> {cryst.n_atoms} atoms")
        label = os.path.splitext(os.path.basename(name))[0]
        CRYSTALS[label] = lambda c=cryst: c
        run(label, atom_indices=atom_indices, legacy=legacy,
            all_types=all_types, n_workers=n_workers,
            near_gap_threshold=near_gap_threshold,
            primary_only=primary_only, n_images=n_images,
            summary_out=summary_out, symmetrized=symmetrize,
            filter_dir=filter_dir)
    else:
        print(f"Unknown crystal: {name}")
        print(f"Built-in structures: {', '.join(CRYSTALS)}")
        print(f"Or provide a CIF/POSCAR file path (requires pymatgen).")
        sys.exit(1)
