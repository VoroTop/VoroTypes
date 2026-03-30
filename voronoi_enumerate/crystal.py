"""
Crystal structure handling: lattice + basis atoms, supercell generation.

Provides factory methods for common structures (FCC, BCC, HCP, SC) and
a from_file loader for CIF files (built-in parser) and POSCAR (via pymatgen).
"""

import re
import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ------------------------------------------------------------------
# CIF parsing helpers (no external dependencies)
# ------------------------------------------------------------------

def _cif_float(s):
    """Parse a CIF numeric value, stripping uncertainty in parentheses.

    Examples: '1.234' -> 1.234, '1.234(5)' -> 1.234, '-0.5' -> -0.5
    """
    return float(re.sub(r'\([^)]*\)', '', s))


def _lattice_from_params(a, b, c, alpha, beta, gamma):
    """Convert (a, b, c, alpha, beta, gamma) to a 3x3 lattice matrix.

    Returns lattice vectors as rows, with a along x.
    """
    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)

    ax = a
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))

    return np.array([
        [ax, 0.0, 0.0],
        [bx, by, 0.0],
        [cx, cy, cz],
    ])


def _parse_symops(text):
    """Extract symmetry operations from CIF text.

    Returns a list of (R, t) tuples where R is 3x3 and t is length-3,
    representing the operation frac' = R @ frac + t.
    """
    # Find the symmetry loop
    symops = []
    # Match loop_ blocks containing _symmetry_equiv_pos_as_xyz
    pattern = re.compile(
        r'loop_\s*\n'
        r'((?:\s*_symmetry_equiv_pos\S*\s*\n)+)'  # column headers
        r'((?:\s*\d*\s*[\'"]?[^_\n].*\n)*)',       # data lines
        re.MULTILINE
    )
    for m in pattern.finditer(text):
        headers = m.group(1).strip().split('\n')
        headers = [h.strip() for h in headers]
        data_block = m.group(2).strip()
        if not data_block:
            continue

        # Find which column has the xyz string
        xyz_col = None
        for i, h in enumerate(headers):
            if 'as_xyz' in h:
                xyz_col = i
                break
        if xyz_col is None:
            continue

        for line in data_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Extract the xyz expression (may be quoted)
            xyz_str = None
            for q in ["'", '"']:
                idx = line.find(q)
                if idx >= 0:
                    end = line.find(q, idx + 1)
                    if end > idx:
                        xyz_str = line[idx + 1:end]
                        break
            if xyz_str is None:
                # Unquoted: split by whitespace, take the right column
                parts = line.split()
                if len(parts) > xyz_col:
                    xyz_str = parts[xyz_col]
            if xyz_str:
                op = _parse_one_symop(xyz_str)
                if op is not None:
                    symops.append(op)

    if not symops:
        # Default: identity only
        symops = [(np.eye(3), np.zeros(3))]
    return symops


def _parse_one_symop(s):
    """Parse a symmetry operation string like 'x+1/2, -y, z' into (R, t)."""
    components = [c.strip() for c in s.split(',')]
    if len(components) != 3:
        return None

    R = np.zeros((3, 3))
    t = np.zeros(3)
    var_map = {'x': 0, 'y': 1, 'z': 2}

    for i, comp in enumerate(components):
        # Remove spaces
        comp = comp.replace(' ', '')
        # Parse tokens: look for x, y, z with optional sign, and fractions
        pos = 0
        sign = 1.0
        while pos < len(comp):
            ch = comp[pos]
            if ch == '+':
                sign = 1.0
                pos += 1
            elif ch == '-':
                sign = -1.0
                pos += 1
            elif ch in var_map:
                R[i, var_map[ch]] = sign
                sign = 1.0
                pos += 1
            elif ch.isdigit():
                # Parse a number or fraction like "1/2" or "0.5"
                num_match = re.match(r'(\d+)/(\d+)', comp[pos:])
                if num_match:
                    val = int(num_match.group(1)) / int(num_match.group(2))
                    t[i] += sign * val
                    sign = 1.0
                    pos += num_match.end()
                else:
                    num_match = re.match(r'[\d.]+', comp[pos:])
                    if num_match:
                        t[i] += sign * float(num_match.group())
                        sign = 1.0
                        pos += num_match.end()
                    else:
                        pos += 1
            else:
                pos += 1

    return (R, t)


def _parse_atom_sites(text):
    """Extract atom species and fractional coordinates from CIF text."""
    species = []
    frac_coords = []

    # Find loop_ blocks with _atom_site columns
    pattern = re.compile(
        r'loop_\s*\n'
        r'((?:\s*_atom_site\S*\s*\n)+)'     # column headers
        r'((?:\s+\S+.*\n)*)',                 # data lines
        re.MULTILINE
    )

    for m in pattern.finditer(text):
        headers = m.group(1).strip().split('\n')
        headers = [h.strip() for h in headers]
        data_block = m.group(2).strip()
        if not data_block:
            continue

        # Find column indices
        col = {}
        for i, h in enumerate(headers):
            if h == '_atom_site_type_symbol':
                col['type'] = i
            elif h == '_atom_site_label':
                col['label'] = i
            elif h == '_atom_site_fract_x':
                col['x'] = i
            elif h == '_atom_site_fract_y':
                col['y'] = i
            elif h == '_atom_site_fract_z':
                col['z'] = i

        if 'x' not in col or 'y' not in col or 'z' not in col:
            continue

        # Species column: prefer type_symbol, fall back to label
        sp_col = col.get('type', col.get('label'))

        for line in data_block.split('\n'):
            parts = line.split()
            if len(parts) < len(headers):
                continue
            x = _cif_float(parts[col['x']])
            y = _cif_float(parts[col['y']])
            z = _cif_float(parts[col['z']])
            sp = parts[sp_col] if sp_col is not None else 'X'
            species.append(sp)
            frac_coords.append([x, y, z])

    return species, frac_coords


def _apply_symops(species, frac_coords, symops):
    """Apply symmetry operations and remove duplicate atoms."""
    all_species = []
    all_coords = []

    for sp, fc in zip(species, frac_coords):
        fc = np.array(fc)
        for R, t in symops:
            new_fc = R @ fc + t
            # Wrap to [0, 1)
            new_fc = new_fc % 1.0
            all_species.append(sp)
            all_coords.append(new_fc)

    # Remove duplicates (atoms closer than tolerance in fractional coords)
    unique_sp = []
    unique_fc = []
    tol = 1e-4
    for sp, fc in zip(all_species, all_coords):
        is_dup = False
        for ufc in unique_fc:
            diff = fc - ufc
            # Periodic distance in fractional coords
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < tol:
                is_dup = True
                break
        if not is_dup:
            unique_sp.append(sp)
            unique_fc.append(fc)

    return unique_sp, unique_fc


# ------------------------------------------------------------------

@dataclass
class AtomImage:
    """An atom in a supercell, identified by unit-cell index + image vector."""
    atom_index: int
    image: Tuple[int, int, int]
    cart_pos: np.ndarray

    def __repr__(self):
        return f"AtomImage(atom={self.atom_index}, image={self.image})"


class Crystal:
    """Crystal structure: lattice vectors + basis atom positions."""

    def __init__(self, lattice, frac_coords, species=None):
        """
        Parameters
        ----------
        lattice : array-like, shape (3, 3)
            Lattice vectors as rows: lattice[i] is the i-th vector.
        frac_coords : array-like, shape (N, 3)
            Fractional coordinates of atoms in the unit cell.
        species : list of str, optional
            Atom species labels.
        """
        self.lattice = np.array(lattice, dtype=float)
        self.frac_coords = np.array(frac_coords, dtype=float)
        self.n_atoms = len(self.frac_coords)
        self.species = species or [f"A{i}" for i in range(self.n_atoms)]

    @property
    def cart_coords(self):
        """Cartesian coordinates of unit cell atoms."""
        return self.frac_coords @ self.lattice

    def make_supercell(self, n_images=3):
        """Generate supercell with periodic images in all directions.

        Parameters
        ----------
        n_images : int
            Include images from -n_images to +n_images in each direction.

        Returns
        -------
        coords : ndarray, shape (M, 3)
            Cartesian positions of all supercell atoms.
        images : list of AtomImage
            Identity of each supercell atom.
        """
        cart_basis = self.cart_coords
        coords = []
        images = []

        for n1, n2, n3 in product(range(-n_images, n_images + 1), repeat=3):
            shift = (n1 * self.lattice[0]
                     + n2 * self.lattice[1]
                     + n3 * self.lattice[2])
            for i in range(self.n_atoms):
                pos = cart_basis[i] + shift
                coords.append(pos)
                images.append(AtomImage(i, (n1, n2, n3), pos.copy()))

        return np.array(coords), images

    # ------------------------------------------------------------------
    # Factory methods for common crystal structures
    # ------------------------------------------------------------------

    @classmethod
    def fcc(cls, a=1.0):
        """FCC with conventional lattice constant a (primitive cell)."""
        lattice = (a / 2) * np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=float)
        return cls(lattice, [[0.0, 0.0, 0.0]])

    @classmethod
    def bcc(cls, a=1.0):
        """BCC with conventional lattice constant a (primitive cell)."""
        lattice = (a / 2) * np.array([
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ], dtype=float)
        return cls(lattice, [[0.0, 0.0, 0.0]])

    @classmethod
    def sc(cls, a=1.0):
        """Simple cubic with lattice constant a."""
        return cls(a * np.eye(3), [[0.0, 0.0, 0.0]])

    @classmethod
    def hcp(cls, a=1.0, c_over_a=None):
        """HCP with lattice constant a (ideal c/a = sqrt(8/3) if not given)."""
        if c_over_a is None:
            c_over_a = np.sqrt(8.0 / 3.0)
        c = a * c_over_a
        lattice = np.array([
            [a, 0, 0],
            [-a / 2, a * np.sqrt(3) / 2, 0],
            [0, 0, c],
        ])
        return cls(lattice,
                   [[0.0, 0.0, 0.0], [1.0/3, 2.0/3, 0.5]],
                   ['A', 'A'])

    @classmethod
    def from_file(cls, filename):
        """Load crystal structure from a file.

        Supports CIF files natively and POSCAR/other formats via pymatgen.
        """
        if filename.lower().endswith('.cif'):
            return cls._from_cif(filename)
        # Non-CIF formats: use pymatgen
        from pymatgen.core import Structure
        struct = Structure.from_file(filename)
        return cls(
            struct.lattice.matrix,
            struct.frac_coords,
            [str(s) for s in struct.species],
        )

    @classmethod
    def _from_cif(cls, filename):
        """Parse a CIF file without external dependencies.

        Handles:
        - Lattice parameters (a, b, c, alpha, beta, gamma)
        - Atom sites from loop_ blocks (_atom_site_fract_x/y/z)
        - Symmetry operations (_symmetry_equiv_pos_as_xyz)
        - Values with uncertainties like 1.234(5)
        """
        with open(filename) as f:
            text = f.read()

        # Parse lattice parameters
        params = {}
        for key in ('_cell_length_a', '_cell_length_b', '_cell_length_c',
                     '_cell_angle_alpha', '_cell_angle_beta',
                     '_cell_angle_gamma'):
            m = re.search(rf'{key}\s+(\S+)', text)
            if m is None:
                raise ValueError(f"CIF missing {key}")
            params[key] = _cif_float(m.group(1))

        a = params['_cell_length_a']
        b = params['_cell_length_b']
        c = params['_cell_length_c']
        alpha = np.radians(params['_cell_angle_alpha'])
        beta = np.radians(params['_cell_angle_beta'])
        gamma = np.radians(params['_cell_angle_gamma'])

        lattice = _lattice_from_params(a, b, c, alpha, beta, gamma)

        # Parse symmetry operations
        symops = _parse_symops(text)

        # Parse atom sites
        species, frac_coords = _parse_atom_sites(text)

        if not species:
            raise ValueError("CIF contains no atom sites")

        # Apply symmetry operations to generate all atoms
        if len(symops) > 1:
            species, frac_coords = _apply_symops(species, frac_coords,
                                                  symops)

        return cls(lattice, frac_coords, species)
