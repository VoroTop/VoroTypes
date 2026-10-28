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

VoroTypes also handles two-dimensional crystals (option `-2`), producing
filters of p-vectors for use with VoroTop's `-2` option; see
[Two-dimensional Systems](#two-dimensional-systems).

---

### Table of Contents

* [Installation](#installation)
* [Quick Start](#quick-start)
* [Options](#options)
* [Examples](#examples)
* [Two-dimensional Systems](#two-dimensional-systems)
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
- [pymatgen](https://pymatgen.org/) (structure input and space-group detection)

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
| `-2` | Treat the input as a two-dimensional crystal (see [Two-dimensional Systems](#two-dimensional-systems)). |
| `--atoms 0 2` | Enumerate only the specified atom indices (default: all atoms). |
| `--primary` | Enumerate only primary resolutions: those realized with positive probability in the limit of vanishing perturbation. Zero-volume tetrahedra are admitted only when their four coplanar atoms form a degenerate edge of the Voronoi cell (all other equidistant atoms on one side of their plane); tetrahedra on coplanar sets with atoms on both sides are secondary and excluded. |
| `--all-types` | Include partially resolved cell types (cells with some degenerate vertices left unresolved). |
| `-j N` | Use N parallel worker processes. |
| `--near-gap-threshold T` | Include near-equidistant atoms within a relative gap of T (see [Limitations](#limitations)). |
| `--max-memory G` | Address-space cap in GB (default: 4); the run aborts with a clear message if it is exceeded. Not supported on macOS, where a warning is printed and the run is uncapped. |
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

#### Tantalum nitride: reading the gap diagnostic

Every filter header records the structure's minimum relative gap --
how much farther (relative to the vertex--atom distance) the nearest
non-equidistant atom sits from a Voronoi vertex than the equidistant
set:

```bash
python enumerate_cells.py examples/TaN_mp-1279.cif
```
```
#   Min relative gap: 1.263186e-01
```

Here the nearest such atom is ~13% farther away, comfortably beyond
small perturbations, so the default filter is complete as computed.
When the reported gap is small (below 1e-2 the header carries an
explicit warning), regenerate with a threshold at least as large as
the reported gap:

```bash
python enumerate_cells.py structure.cif --near-gap-threshold 0.01
```

This treats atoms within the threshold as borderline and enumerates
every subset. For example, triclinic CoWO4 (mp-19092) reports a gap
of 3.4e-03, and rerunning with `--near-gap-threshold 0.005` (the
filter-library setting) grows its family from 6 to 20 Weinberg
types.

---

### Two-dimensional Systems

With the `-2` option, VoroTypes enumerates the Voronoi cell topologies
of a two-dimensional crystal and writes a filter for VoroTop's `-2`
option:

```bash
python enumerate_cells.py -2 square
VoroTop trajectory.dump -2 -f square.filter -l
```

In two dimensions a Voronoi cell's topology is recorded by its
*p-vector* `(n, e1, ..., en)`: the number of edges `n` of the cell,
followed by the numbers of edges of its `n` neighbors in cyclic order
[[Lazar et al. 2024](https://iopscience.iop.org/article/10.1088/1361-651X/ad8ad9)].
This is the representation VoroTop uses for two-dimensional systems.

A degenerate Voronoi vertex in two dimensions is a point where `m > 3`
cells meet; their particles lie on a common circle. A small perturbation
resolves it into one of the triangulations of the `m`-gon formed by
those particles (Catalan(`m`-2) of them: 2 for a square, 14 for a
hexagon). Different degenerate vertices resolve independently, so
the family of a cell is obtained by combining every triangulation of
every degenerate vertex that touches the cell or one of its neighbors.
With `--all-types`, partially resolved vertices (any non-crossing set of
diagonals, not only triangulations) are included as well. In two
dimensions every triangulation is a primary resolution, so `--primary`
gives the same filter as the default mode.

#### Built-in lattices

| Name | Atoms/cell | Degenerate vertices | p-vectors (default) | p-vectors (`--all-types`) | Time |
|------|:-:|------|--:|--:|--:|
| `triangular` | 1 | none | 1 | 1 | <1 s |
| `square` | 1 | 4 cells meet | 1,261 | 3,959 | <1 s |
| `lieb` | 3 | 4 cells meet | 55 (4 corner + 51 edge) | 88 (12 + 76) | <1 s |
| `kagome` | 3 | 6 cells meet | 176,128 | 233,662 | ~10 s |
| `honeycomb` | 2 | 6 cells meet | not tractable (see below) | | |

`graphene` and `hexagonal` are accepted as aliases for `honeycomb` and
`triangular`. Sites related by symmetry (the three kagome sites, the
two Lieb edge sites) produce identical families and share one type in
the filter; inequivalent sites receive separate types.

**Why the honeycomb lattice is not included.** The Voronoi cell of a
honeycomb (graphene) site is a triangle, each of whose corners is
shared by six cells. Under perturbation the cell can have from 3 to 12
edges, and its p-vector depends on how 12 such six-fold vertices
resolve, giving 14^12, or about 5.7 x 10^13, combinations. The
resulting family is far larger than the kagome family (which depends
on 10 six-fold vertices) and too large for a practical filter. For
honeycomb systems, use a stochastic filter instead:

```bash
VoroTop honeycomb.dump -2 -mf 1000000 0.001
```

#### Input file format

A two-dimensional crystal is given as a plain-text file:

```
# Lieb lattice (examples/2d/lieb.txt)
1.0  0.0             <- lattice vector a1 (Cartesian)
0.0  1.0             <- lattice vector a2
0.0  0.0   corner    <- fractional coordinates and optional species label
0.5  0.0   edge
0.0  0.5   edge
```

Text after `#` is ignored, and a line holding the number of atoms may
optionally precede the atom list. Species labels name the types in the
filter (`lieb-corner`, `lieb-edge`).

```bash
python enumerate_cells.py -2 examples/2d/lieb.txt
```

The enumeration depends only on the combinatorics of the crystal: a
rectangular lattice (`examples/2d/rectangular.txt`) gives the same
filter as the square lattice.

Pre-computed filters for the built-in lattices are in `filters/2d/`.

---

### Pre-computed Filters

A library of pre-computed filter files for 1,088 structures from the
[Materials Project](https://materialsproject.org/) is available at
[vorotop.org/filter-library](https://www.vorotop.org/filter-library.html).
Filters for two-dimensional lattices are in `filters/2d/` and in the
two-dimensional section of the filter library.

---

### Filter File Format

The output is a VoroTop-compatible filter file:

```
#   AL_MP-134 filter, computed by voronoi_enumerate
#   Total Weinberg types: 2815
#   Min relative gap: 7.320507e-01
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
#   WARNING: Near-equidistant neighbors detected (gap_rel=3.1e-03).
#   This filter may be incomplete for perturbations that close this gap; consider
#   regenerating with --near-gap-threshold >= 3e-03, or supplement with a
#   stochastic filter: VoroTop structure.xyz -mf 100000 <sigma>
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

**Two-dimensional systems.** The options `--near-gap-threshold`,
`--legacy` and `--symmetrize` apply to three-dimensional crystals only.
Two-dimensional input must list exact (or nearly exact) coordinates:
particles are treated as cocircular when their distances to a vertex
agree to within 10^-6 of the nearest-neighbor distance.

**Input sensitivity.** The enumeration depends on the exact atomic
coordinates, not just the space group. Different DFT relaxations of
the same structure can produce different Weinberg vector sets. Ensure
the reference structure matches the one used in your simulation.

---

### Publications

* Lazar, E.A. and Rycroft, C.H., "VoroTypes: automated enumeration of
  Voronoi cell topologies for structural identification of complex
  crystals in atomistic simulations,"
  [arXiv:2610.21911](https://arxiv.org/abs/2610.21911), 2026.

* Lazar, E.A. and Rycroft, C.H., "Structural identification in general
  crystalline systems via Voronoi cell topology,"
  [arXiv:2610.20498](https://arxiv.org/abs/2610.20498), 2026.

* Lazar, E.A., Lu, J., Rycroft, C.H., and Schwarcz, D., "Characterizing
  structural features of two-dimensional particle systems through
  Voronoi topology,"
  [Model. Simul. Mater. Sci. Eng. 32:085022](https://iopscience.iop.org/article/10.1088/1361-651X/ad8ad9),
  2024.

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
