"""Regression test for the supercell-sufficiency check in analyze_voronoi.

The FCC lattice is written in its standard primitive basis and in a
strongly sheared but equivalent basis (a3 -> a3 + 6 a1).  With the check
disabled, the +/-3-image supercell truncates the cell in the sheared basis
and the wrong polyhedron is returned; with the check enabled the supercell
is enlarged automatically and the two bases give the same cell.

Usage: python3 test_supercell.py
"""

import io
import contextlib

import numpy as np

from voronoi_enumerate.crystal import Crystal
from voronoi_enumerate.voronoi import analyze_voronoi, required_images


def cell_signature(vertices, coords, central_idx):
    p0 = coords[central_idx]
    dists = sorted(round(float(np.linalg.norm(v.position - p0)), 6)
                   for v in vertices)
    return len(vertices), sorted(v.n_equidistant for v in vertices), dists


def main():
    fcc = Crystal.fcc(a=1.0)
    sheared_lattice = fcc.lattice.copy()
    sheared_lattice[2] += 6 * fcc.lattice[0]
    sheared = Crystal(sheared_lattice, fcc.frac_coords, fcc.species)

    with contextlib.redirect_stdout(io.StringIO()):
        ref = analyze_voronoi(fcc, 0, n_images=3)
        off = analyze_voronoi(sheared, 0, n_images=3, check_supercell=False)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            on = analyze_voronoi(sheared, 0, n_images=3, check_supercell=True)

    sig_ref, sig_off, sig_on = (cell_signature(r[0], r[2], r[1]) for r in (ref, off, on))
    assert sig_ref[0] == 14 and sig_ref[1] == [4] * 8 + [6] * 6, sig_ref
    assert sig_off != sig_ref, "sheared basis should truncate the cell at +/-3"
    assert required_images(sheared, off[0], off[2][off[1]]) > 3
    assert 'rebuilding' in log.getvalue()
    assert sig_on == sig_ref, (sig_on, sig_ref)
    # the standard basis needs no rebuild and satisfies the bound at +/-3
    assert required_images(fcc, ref[0], ref[2][ref[1]]) <= 3
    print("supercell check: OK (sheared FCC rebuilt to the standard cell)")


if __name__ == '__main__':
    main()
