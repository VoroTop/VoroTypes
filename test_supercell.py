"""Regression test for the supercell-sufficiency check in analyze_voronoi.

The FCC lattice is written in its standard primitive basis and in a
strongly sheared but equivalent basis (a3 -> a3 + 3 a1).  With the check
disabled, a +/-3-image supercell truncates the cell in the sheared basis
and the wrong polyhedron is returned; with the check enabled (the
default, starting from +/-1 image) the supercell is enlarged until the
bound holds and the two bases give the same cell.

Usage: python3 test_supercell.py
"""

import numpy as np

from voronoi_enumerate.crystal import Crystal
from voronoi_enumerate.voronoi import analyze_voronoi, required_images


def cell_signature(vertices, coords, central_idx):
    p0 = coords[central_idx]
    dists = sorted(round(float(np.linalg.norm(v.position - p0)), 6)
                   for v in vertices)
    return len(vertices), sorted(v.n_equidistant for v in vertices), dists


def images_used(images):
    return max(abs(k) for img in images for k in img.image)


def main():
    fcc = Crystal.fcc(a=1.0)
    sheared_lattice = fcc.lattice.copy()
    sheared_lattice[2] += 3 * fcc.lattice[0]
    sheared = Crystal(sheared_lattice, fcc.frac_coords, fcc.species)

    ref3 = analyze_voronoi(fcc, 0, n_images=3, check_supercell=False)
    ref = analyze_voronoi(fcc, 0)                       # default: bound decides
    off = analyze_voronoi(sheared, 0, n_images=3, check_supercell=False)
    on = analyze_voronoi(sheared, 0)

    sig_ref3, sig_ref, sig_off, sig_on = (
        cell_signature(r[0], r[2], r[1]) for r in (ref3, ref, off, on))
    assert sig_ref3[0] == 14 and sig_ref3[1] == [4] * 8 + [6] * 6, sig_ref3
    assert sig_ref == sig_ref3, (sig_ref, sig_ref3)
    assert sig_off != sig_ref, "sheared basis should truncate the cell at +/-3"
    assert required_images(sheared, off[0], off[2][off[1]]) > 3
    assert images_used(on[3]) > 3, "sheared basis should be enlarged past +/-3"
    assert sig_on == sig_ref, (sig_on, sig_ref)
    # the standard basis satisfies the bound at +/-3 and is enlarged to it
    assert required_images(fcc, ref[0], ref[2][ref[1]]) <= 3
    assert images_used(ref[3]) == required_images(fcc, ref[0], ref[2][ref[1]])
    print(f"supercell check: OK (standard FCC uses +/-{images_used(ref[3])}, "
          f"sheared FCC enlarged to +/-{images_used(on[3])}, same cell)")


if __name__ == '__main__':
    main()
