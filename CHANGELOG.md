# Changelog

## v1.4 (2026-11-15)

### Supercell determined by the bound

* The Voronoi analysis now starts from a supercell of +/-1 image and
  enlarges it until the sufficiency bound holds, instead of starting from
  a fixed +/-3.  No arbitrary image count enters; `--n-images` sets only
  the starting value.
* `required_images` requires the supercell to contain every atom within
  3 R_max of the central atom (the companion paper's 2 R_max for an exact
  cell, plus R_max so that the near-gap diagnostics see every atom whose
  relative gap to a vertex is below 100%).  Reported minimum gaps are
  therefore exact whenever they matter.
* `enumerate_cells.py` prints the image count actually used for each
  atom and records the largest in the summary.
* Verified on all 1,300 library candidates (raw and symmetrized cells):
  Voronoi vertices, equidistant sets, and reported minimum gaps identical
  to the +/-3 results, except for floating-point-level differences at
  vertices that are near-degenerate below 1e-5 (where the Delaunay
  triangulation itself depends on the point set) and one oblique raw cell
  whose gaps the old +/-3 supercell had overstated.  Regenerated filters
  are identical to the deployed library.  Voronoi stage about 2.5x faster.

## v1.3 (2026-11-05)

### Supercell-sufficiency check

* `analyze_voronoi` now evaluates the supercell-sufficiency bound of the
  companion paper (Supplementary Section S1.1), N_j = ceil(2 R_max / h_j),
  on the computed Voronoi cell and enlarges the supercell (at most doubling
  the image count, up to +/-12) when the bound requires it.  A too-small
  supercell can only enlarge the computed cell, so the check is
  conservative.  The default of +/-3 images satisfies the bound for every
  structure in the filter library (largest ratio 2.14), so no library
  result changes.
* New `test_supercell.py`: FCC in a strongly sheared basis is rebuilt to
  the standard cell.
* README: documents `--n-images`.

## v1.2 (2026-10-29)

### Two-dimensional crystals

New option `-2` enumerates the Voronoi cell topologies of a
two-dimensional crystal and writes a filter of p-vectors for VoroTop's
`-2` option.

* `voronoi_enumerate/planar.py`: planar crystals (`Crystal2D`, built-in
  `square`, `triangular`, `honeycomb`, `kagome`, `lieb`, and a plain-text
  input format), polygon dissections, and the two-stage enumeration of
  p-vectors.
* Degenerate vertices are resolved by triangulations (default) or by
  arbitrary dissections (`--all-types`) of the inscribed polygon.
  `--primary` is accepted and equals the default mode.
* `filters/2d/`: pre-computed filters for the triangular (1 p-vector),
  square (1,261), Lieb (55) and kagome (176,128) lattices.
  The honeycomb lattice is documented but not included: its family is
  too large for a practical filter.
* `examples/2d/`: input files for the Lieb and rectangular lattices.
* `test_2d.py`: regression tests for dissection counts, family sizes,
  site equivalence, and the shipped filters.
* `filter.py`: `write_filter_file` takes `dimension=2` to label p-vector
  filters.

Validation: every p-vector observed in perturbed square, Lieb and kagome
crystals (VoroTop `-2 -mf`, and an independent Delaunay sampler) is in
the enumerated family; the square and Lieb families are recovered in
full by sampling; random joint triangulations of all degenerate
vertices in square, Lieb, kagome and honeycomb patches were realized
by explicit perturbations; and VoroTop classifies every particle of
perturbed square, Lieb and kagome crystals using the new filters.

## v1.1 (2026-09-28)

### Corrected primary/secondary classification

Primary and secondary resolutions now follow the definition of Lazar et al.
(PNAS 2015): a resolution is *primary* if it is realized with positive
probability in the limit of vanishing perturbation, and *secondary* if its
probability vanishes with the perturbation magnitude.

Version 1.0 classified every resolution containing a zero-volume tetrahedron
(four coplanar equidistant atoms) as secondary. That is correct only when
other equidistant atoms lie on both sides of the plane of the four (e.g. the
equatorial squares of the FCC octahedral vertex). When all other equidistant
atoms lie on one side, the four atoms form a degenerate edge of the Voronoi
cell, which a perturbation resolves in one of two ways with fixed
probability; such resolutions are primary. Examples: the base of the
omega-Ti 2d square-pyramid vertex and every face of the simple-cubic cube.

Consequences:

* `--primary` output grows for structures with such vertices (omega-Ti 2d:
  13 -> 382 types, now equal to the default mode; simple-cubic vertex:
  32 -> 256 primary resolutions). FCC and HCP are unchanged. About a fifth
  of the 1,300-structure library is affected.
* The default and `--all-types` modes are unchanged.
* The published filter library was regenerated in primary mode: 144 filters
  replaced, 33 structures no longer complete within the 300 s budget
  (1,121 -> 1,088 primary-mode filters; default 1,054 and all 1,025
  unchanged).

Implementation: `voronoi_enumerate/resolution.py`, `_classify_primary`
(`_is_flat`, `_flat_on_hull_facet`). Validated by direct sampling of
perturbed FCC, omega-Ti and simple-cubic vertices.

### Documentation

* README: documented the `--primary` option; library count updated.
* `enumerate_cells.py`: usage lists `--primary` and `--all-types`.

## v1.0 (2026-07-12)

Initial release.
