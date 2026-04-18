"""
Export enumeration data to binary file for C++ voronoi_core.

Usage:
    python3 export_enum.py sc --primary -o sc_data.bin
    python3 export_enum.py fcc --primary -o fcc_data.bin
    python3 export_enum.py structure.cif --primary -o data.bin
"""

import sys
import struct
import numpy as np
from voronoi_enumerate.crystal import Crystal
from voronoi_enumerate.voronoi import analyze_voronoi
from voronoi_enumerate.resolution import enumerate_resolutions
from voronoi_enumerate.symmetry import compute_site_symmetry

CRYSTALS = {
    'fcc': lambda: Crystal.fcc(a=1.0),
    'bcc': lambda: Crystal.bcc(a=1.0),
    'sc':  lambda: Crystal.sc(a=1.0),
    'hcp': lambda: Crystal.hcp(a=1.0),
}


def export_data(name, output_path, atom_index=0, primary_only=False):
    if name in CRYSTALS:
        cryst = CRYSTALS[name]()
    else:
        import os
        if os.path.isfile(name):
            cryst = Crystal.from_file(name)
        else:
            print(f"Unknown crystal: {name}")
            sys.exit(1)

    print(f"Crystal: {name}")
    print(f"Atoms per unit cell: {cryst.n_atoms}")

    vertices, central_idx, coords, images = analyze_voronoi(
        cryst, atom_index=atom_index)

    degen = [v for v in vertices if v.is_degenerate]
    generic = [v for v in vertices if not v.is_degenerate]

    print(f"Central atom: {central_idx}")
    print(f"Vertices: {len(vertices)} ({len(generic)} generic, "
          f"{len(degen)} degenerate)")

    # Generic tetrahedra
    generic_tets = []
    for v in generic:
        t = tuple(sorted(v.atom_indices))
        if t not in generic_tets:
            generic_tets.append(t)

    # Enumerate resolutions for each degenerate vertex
    vertex_stars = [[] for _ in range(len(degen))]
    vertex_options = []  # [vi][ri] = list of tet tuples (global indices)
    n_res = []

    for i, v in enumerate(degen):
        pts, c_local, atom_map = v.point_config(coords, central_idx)
        res_list, _ = enumerate_resolutions(pts, central=0)

        if primary_only:
            res_list = [r for r in res_list if r.is_primary]

        options = []
        for r in res_list:
            global_tets = [
                tuple(sorted(atom_map[u] for u in s))
                for s in r.star
            ]
            options.append(global_tets)
            vertex_stars[i].append(
                frozenset(tuple(sorted(atom_map[u] for u in s))
                          for s in r.star))
        vertex_options.append(options)
        n_res.append(len(options))

        n_primary = sum(1 for r in enumerate_resolutions(pts, central=0)[0]
                        if r.is_primary) if not primary_only else len(options)
        print(f"  Vertex {i}: {v.n_equidistant} atoms, "
              f"{len(options)} resolutions"
              f"{' (primary only)' if primary_only else ''}")

    # Compute symmetry
    symmetry = compute_site_symmetry(
        cryst, central_idx, coords, degen, vertex_stars, verbose=True)
    print(f"Symmetry order: {symmetry.order}")

    # Now symmetry may have added resolution permutations
    # Re-read n_res after symmetry (it may have reordered)
    for vi in range(len(degen)):
        n_res[vi] = len(vertex_options[vi])

    # Write binary file
    with open(output_path, 'wb') as f:
        f.write(b'VORO')
        f.write(struct.pack('<i', len(degen)))       # n_vertices
        f.write(struct.pack('<i', central_idx))       # central_idx

        # Generic tets
        f.write(struct.pack('<i', len(generic_tets)))
        for t in generic_tets:
            for v in t:
                f.write(struct.pack('<i', v))

        # Vertex options
        for vi in range(len(degen)):
            f.write(struct.pack('<i', n_res[vi]))
            for ri in range(n_res[vi]):
                tets = vertex_options[vi][ri]
                f.write(struct.pack('<i', len(tets)))
                for t in tets:
                    for v in t:
                        f.write(struct.pack('<i', v))

        # Symmetry
        f.write(struct.pack('<i', symmetry.order))

        # Vertex permutations
        for gi in range(symmetry.order):
            for vi in range(len(degen)):
                f.write(struct.pack('<i', symmetry.vertex_perms[gi][vi]))

        # Resolution permutations
        for gi in range(symmetry.order):
            for vi in range(len(degen)):
                for ri in range(n_res[vi]):
                    f.write(struct.pack('<i', symmetry.res_perms[gi][vi][ri]))

    total = 1
    for nr in n_res:
        total *= nr
    print(f"\nExported to {output_path}")
    print(f"Total combinations: {total:.2e}")
    print(f"Estimated canonical combos: ~{total / symmetry.order:.2e}")
    print(f"\nTo build and run:")
    print(f"  g++ -O3 -fopenmp -std=c++17 -o voronoi_core "
          f"voronoi_enumerate/core/voronoi_core.cpp")
    print(f"  ./voronoi_core {output_path} -j 64")


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'fcc'
    output_path = 'enum_data.bin'
    primary_only = '--primary' in sys.argv
    atom_index = 0

    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if '--atoms' in sys.argv:
        idx = sys.argv.index('--atoms')
        if idx + 1 < len(sys.argv):
            atom_index = int(sys.argv[idx + 1])

    export_data(name, output_path, atom_index=atom_index,
                primary_only=primary_only)
