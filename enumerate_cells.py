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
        n_workers=1, near_gap_threshold=None):
    t0 = time.time()

    cryst = CRYSTALS[name]()
    print(f"Crystal: {name.upper()}")
    print(f"Atoms per unit cell: {cryst.n_atoms}")
    print(f"Species: {cryst.species}")
    print(f"Lattice:\n{cryst.lattice}\n")

    if atom_indices is None:
        atom_indices = list(range(cryst.n_atoms))

    # Collect cell types from all atoms for filter file output
    # Each entry is (atom_index, cell_types_dict)
    per_atom_types = []

    for atom_idx in atom_indices:
        print("=" * 65)
        print(f"Atom {atom_idx}: Voronoi analysis")
        print("=" * 65)
        vertices, central_idx, coords, images = analyze_voronoi(
            cryst, atom_index=atom_idx,
            near_gap_threshold=near_gap_threshold,
        )

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
    filter_path = f"{name.upper()}.filter"
    n_wv, n_groups = write_filter_file(
        filter_path, name, per_atom_types, species=cryst.species
    )
    print(f"\nFilter file written: {filter_path} "
          f"({n_groups} type(s), {n_wv} Weinberg vectors)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f} seconds")


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

    max_memory = MAX_MEMORY_GB
    if '--max-memory' in sys.argv:
        idx = sys.argv.index('--max-memory')
        if idx + 1 < len(sys.argv):
            max_memory = float(sys.argv[idx + 1])
    set_memory_limit(max_memory)

    if name in CRYSTALS:
        run(name, atom_indices=atom_indices, legacy=legacy,
            all_types=all_types, n_workers=n_workers,
            near_gap_threshold=near_gap_threshold)
    elif os.path.isfile(name):
        # Load crystal structure from CIF, POSCAR, or other file
        cryst = Crystal.from_file(name)
        label = os.path.splitext(os.path.basename(name))[0]
        CRYSTALS[label] = lambda c=cryst: c
        run(label, atom_indices=atom_indices, legacy=legacy,
            all_types=all_types, n_workers=n_workers,
            near_gap_threshold=near_gap_threshold)
    else:
        print(f"Unknown crystal: {name}")
        print(f"Built-in structures: {', '.join(CRYSTALS)}")
        print(f"Or provide a CIF/POSCAR file path (requires pymatgen).")
        sys.exit(1)
