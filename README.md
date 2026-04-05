# VoroTypes: Deterministic Enumeration of Voronoi Cell Topologies

## A tool for enumerating all Voronoi cell types of a crystal under perturbation

*VoroTypes* deterministically enumerates all combinatorial types of
Voronoi cells that can arise from infinitesimal perturbations of a
crystal structure. The output is a
[VoroTop](https://github.com/VoroTop/VoroTop)-compatible filter file
that can be used to classify local structure in atomistic simulations.

For simple structures like BCC, the Voronoi cell is unique under any
perturbation. For structures with degenerate Voronoi vertices (such as
FCC, HCP, and many intermetallics), small perturbations break
degeneracies in different ways, producing multiple distinct cell
topologies. VoroTypes finds all of them.

---

### Table of Contents

* [Installation](#installation)
* [Quick Start](#quick-start)
* [Options](#options)
* [Examples](#examples)
* [Pre-computed Filters](#pre-computed-filters)
* [Filter File Format](#filter-file-format)
* [Limitations](#limitations)
* [Publications](#publications)
* [License](#license)
* [Contact](#contact)

---

### Installation

#### Prerequisites

- Python 3.10+
- [pymatgen](https://pymatgen.org/) (for reading CIF/POSCAR files)

#### Setup

```bash
git clone https://github.com/VoroTop/VoroTypes.git
cd VoroTypes
pip install -r requirements.txt
```

---

### Quick Start

Given a CIF file for a crystal structure (e.g., downloaded from the
[Materials Project](https://materialsproject.org/)):

```bash
python enumerate_cells.py structure.cif
```

This produces a VoroTop filter file (e.g., `STRUCTURE.filter`) listing
all Weinberg vectors for each atom site. The filter can then be used
with VoroTop:

```bash
VoroTop trajectory.xyz -f STRUCTURE.filter
```

---

### Options

```
python enumerate_cells.py <structure.cif> [options]
```

| Option | Description |
|--------|-------------|
| `--atoms 0 2` | Enumerate only the specified atom indices (default: all atoms). |
| `--all-types` | Include partially resolved cell types (cells with some degenerate vertices left unresolved). |
| `-j N` | Use N parallel worker processes. |
| `--near-gap-threshold T` | Include near-equidistant atoms within a relative gap of T (see [Limitations](#limitations)). |
| `--max-memory G` | Set memory limit in GB (default: 4). Aborts with a clear message if exceeded. |
| `--legacy` | Use the legacy Cartesian-product enumeration algorithm. |

---

### Examples

The `examples/` directory contains CIF files for several structures.

#### Aluminum (FCC)

FCC has 6 degenerate Voronoi vertices, each with 6 equidistant atoms.
Perturbations break these degeneracies in 2,815 distinct ways:

```bash
python enumerate_cells.py examples/Al_mp-134.cif
```
```
Filter file written: AL_MP-134.filter (1 type(s), 2815 Weinberg vectors)
Total time: 2.7 seconds
```

#### Iron (BCC)

BCC has no degenerate vertices, so there is exactly one cell type
(the truncated octahedron):

```bash
python enumerate_cells.py examples/Fe_mp-13.cif
```
```
Filter file written: FE_MP-13.filter (1 type(s), 1 Weinberg vectors)
Total time: 0.0 seconds
```

#### Cadmium (HCP)

HCP has degenerate vertices at both atom sites, producing thousands
of cell types:

```bash
python enumerate_cells.py examples/Cd_mp-94.cif
```
```
Filter file written: CD_MP-94.filter (1 type(s), 9556 Weinberg vectors)
Total time: 162.4 seconds
```

#### Titanium nickel (multi-atom cell)

A two-atom-type unit cell (4 atoms, P2_1/m). VoroTypes automatically
assigns separate type indices to inequivalent sites:

```bash
python enumerate_cells.py examples/TiNi_mp-1048.cif
```
```
Filter file written: TINI_MP-1048.filter (2 type(s), 2 Weinberg vectors)
Total time: 0.1 seconds
```

#### Tantalum nitride

An example with near-equidistant neighbors. Some atoms are almost
(but not exactly) equidistant from a Voronoi vertex. The filter file
includes a diagnostic warning:

```bash
python enumerate_cells.py examples/TaN_mp-1279.cif
```

To extend the enumeration to cover perturbations that would bring
near-equidistant atoms into play:

```bash
python enumerate_cells.py examples/TaN_mp-1279.cif --near-gap-threshold 0.01
```

---

### Pre-computed Filters

A library of pre-computed filter files for 779 structures from the
[Materials Project](https://materialsproject.org/) is available at
[vorotop.org/filter-library](https://www.vorotop.org/filter-library.html).

---

### Filter File Format

The output is a VoroTop-compatible filter file:

```
#   AL_MP-134 filter, computed by voronoi_enumerate
#   Total Weinberg types: 2815
*   1   AL_MP-134   atoms [0]
1   (1,2,3,4,5,3,6,7,8,9,10,8,11,12,...)
1   (1,2,3,4,5,6,7,8,9,10,7,11,12,...)
...
```

Lines beginning with `#` are comments. Lines beginning with `*`
define type indices and labels. Remaining lines map each Weinberg
vector to a type index.

For structures with near-equidistant neighbors (relative gap
< 0.01), the filter includes a warning:

```
#   WARNING: Near-equidistant neighbors detected (gap_rel=3.08e-07).
#   For perturbations sigma > 3e-07, supplement with stochastic filter:
#     VoroTop structure.xyz -mf 100000 <sigma>
```

---

### Limitations

**Combinatorial complexity.** Some structures have Voronoi vertices
with high-order degeneracies (many equidistant atoms), leading to a
combinatorial explosion in the number of resolution types. VoroTypes
uses site symmetry to reduce the enumeration space, but structures
where the product of resolutions across all degenerate vertices
exceeds ~10^6 may be intractable. For such structures, use VoroTop's
stochastic filter generation instead:

```bash
VoroTop structure.xyz -mf 100000 0.0001
```

**Near-equidistant neighbors.** When an atom lies very close to (but
not exactly on) a Voronoi vertex's circumsphere, perturbations larger
than this gap can produce cell types not in the analytical enumeration.
The `--near-gap-threshold` option addresses this by extending the
equidistant set at each vertex to include such atoms, at the cost of
increased combinatorial complexity. The filter file header reports the
minimum relative gap to help users assess whether this matters for
their perturbation scale.

**Input sensitivity.** The enumeration depends on the exact atomic
coordinates, not just the space group. Different DFT relaxations of
the same structure can produce different Weinberg vector sets. Ensure
the reference structure matches the one used in your simulation.

---

### Publications

* Lazar, E.A., Han, J., and Srolovitz, D.J., "A Topological Framework
  for Local Structure Analysis in Condensed Matter,"
  [Proc. Natl. Acad. Sci. 112:E5769](https://www.pnas.org/doi/10.1073/pnas.1505788112),
  2015.

* Lazar, E.A., "VoroTop: Voronoi Cell Topology Visualization and
  Analysis Toolkit,"
  [Model. Simul. Mater. Sci. Eng. 26:1](https://iopscience.iop.org/article/10.1088/1361-651X/aa9a01),
  2017.

---

### License

MIT License. See [LICENSE](LICENSE).

---

### Contact

Contributions are welcome. Please be in touch with questions, comments,
and suggestions, as well as about potential research collaborations.
Emails can be sent to "help" at vorotop.org.
