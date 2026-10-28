"""Regression tests for two-dimensional enumeration (enumerate_cells.py -2).

Checks the counts of polygon dissections, the p-vector family sizes of
the built-in lattices, that symmetry-equivalent sites give identical
families, and that the shipped 2D filters match a fresh enumeration.

Usage: python3 test_2d.py
"""

import os
import re
import sys

from voronoi_enumerate.planar import (
    CRYSTALS_2D, Crystal2D, dissections, enumerate_pvectors,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# (lattice, all_types) -> number of distinct p-vectors per basis atom
EXPECTED = {
    ('triangular', False): [1],
    ('square', False): [1261],
    ('square', True): [3959],
    ('lieb', False): [4, 51, 51],
    ('lieb', True): [12, 76, 76],
    ('kagome', False): [176128] * 3,
    ('kagome', True): [233662] * 3,
}

failures = 0


def check(label, ok, detail=''):
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}: {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures += 1


print("=== Dissections of a convex m-gon ===")
catalan = [1, 2, 5, 14, 42, 132]
schroeder = [1, 3, 11, 45, 197, 903]
for m in range(3, 9):
    check(f"m={m}: {catalan[m - 3]} triangulations, "
          f"{schroeder[m - 3]} dissections",
          len(dissections(m, True)) == catalan[m - 3] and
          len(dissections(m, False)) == schroeder[m - 3])

print("=== Family sizes of built-in lattices ===")
families = {}
for (name, all_types), expected in EXPECTED.items():
    cryst = CRYSTALS_2D[name]()
    got = []
    for a in range(cryst.n_atoms):
        pv, _ = enumerate_pvectors(cryst, a, all_types=all_types,
                                   verbose=False)
        got.append(len(pv))
        families[(name, all_types, a)] = pv
    mode = 'all-types' if all_types else 'default'
    check(f"{name} ({mode}): {expected}", got == expected, f"got {got}")

print("=== Equivalent sites give identical families ===")
for name, all_types in [('kagome', False), ('lieb', False)]:
    sites = range(1, 3) if name == 'lieb' else range(3)
    sets = [families[(name, all_types, a)] for a in sites]
    check(f"{name}: sites {list(sites)}", all(s == sets[0] for s in sets))

print("=== Lattice geometry does not change the combinatorics ===")
rect = Crystal2D.from_file(os.path.join(HERE, 'examples', '2d',
                                        'rectangular.txt'))
pv, _ = enumerate_pvectors(rect, 0, verbose=False)
check("rectangular == square", pv == families[('square', False, 0)])

print("=== Shipped filters match a fresh enumeration ===")
for name in ['triangular', 'square', 'lieb', 'kagome']:
    path = os.path.join(HERE, 'filters', '2d', f'{name}.filter')
    shipped = set()
    with open(path) as f:
        for line in f:
            if line[0].isdigit():
                vec = re.search(r'\(([^)]*)\)', line).group(1)
                shipped.add(tuple(int(x) for x in vec.split(',')))
    fresh = set()
    for a in range(CRYSTALS_2D[name]().n_atoms):
        fresh |= families[(name, False, a)]
    check(f"filters/2d/{name}.filter", shipped == fresh,
          f"{len(shipped)} p-vectors")

print(f"\nResults: {'all passed' if failures == 0 else f'{failures} failed'}")
sys.exit(1 if failures else 0)
