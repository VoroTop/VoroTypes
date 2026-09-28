# Changelog

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
