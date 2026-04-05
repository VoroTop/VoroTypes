"""Generate VoroTop filter files for a large library of crystal structures.

Downloads structures from the Materials Project, probes each for Voronoi
enumeration complexity, and runs the enumeration on all feasible structures.
Filter files are saved in the filters/ directory.
"""
import sys
import os
import time
import traceback
import json
import resource
import argparse
import numpy as np
from math import prod
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from voronoi_enumerate.crystal import Crystal
from voronoi_enumerate.voronoi import analyze_voronoi
from voronoi_enumerate.resolution import enumerate_resolutions
from voronoi_enumerate.combine import (
    enumerate_cell_types_v2, summarize_cell_types,
)
from voronoi_enumerate.filter import write_filter_file, write_intractable_filter


# ===================================================================
# Configuration
# ===================================================================
MP_API_KEY = os.environ.get("MP_API_KEY", "")
FILTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filters")
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "filters", "mp_structures.json")
MAX_RES_PER_VERTEX = 50   # skip if any vertex has more resolutions
MAX_TOTAL_COMBOS = 1_000_000  # skip if product of resolutions exceeds this
MAX_MEMORY_GB = 4          # default memory limit per structure (GB)
N_WORKERS = 1
TARGET_COUNT = 1000


def download_structures():
    """Download diverse structures from Materials Project.

    Strategy: get stable structures with small primitive cells (1-8 atoms)
    across many space groups, prioritizing diversity of structure types.
    """
    from mp_api.client import MPRester

    print("Downloading structures from Materials Project...")
    sys.stdout.flush()

    mpr = MPRester(MP_API_KEY)

    # Get stable structures with small primitive cells
    # Batch by number of sites for diversity
    all_docs = []
    seen_sg = defaultdict(int)

    for nsites_range in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                         (6, 7), (7, 8), (8, 9), (9, 12), (12, 16)]:
        print(f"  Fetching nsites {nsites_range}...", end=" ")
        sys.stdout.flush()
        try:
            docs = mpr.materials.summary.search(
                num_sites=nsites_range,
                is_stable=True,
                fields=['material_id', 'formula_pretty', 'nsites',
                        'symmetry', 'structure'],
            )
            print(f"{len(docs)} found")
            all_docs.extend(docs)
        except Exception as e:
            print(f"error: {e}")

    print(f"Total stable structures fetched: {len(all_docs)}")

    # Deduplicate by space group + nsites to get diverse set
    # Keep at most ~5 per (spacegroup, nsites) pair
    by_type = defaultdict(list)
    for doc in all_docs:
        key = (doc.symmetry.number, doc.nsites)
        by_type[key].append(doc)

    selected = []
    for key in sorted(by_type.keys()):
        docs = by_type[key]
        # Take up to 3 per type for diversity
        selected.extend(docs[:3])

    # If we have too many, trim; if too few, add more
    if len(selected) > TARGET_COUNT * 1.5:
        # Trim to target, keeping diversity
        by_sg = defaultdict(list)
        for doc in selected:
            by_sg[doc.symmetry.number].append(doc)
        trimmed = []
        for sg in sorted(by_sg.keys()):
            n_keep = max(1, TARGET_COUNT * len(by_sg[sg]) // len(selected))
            trimmed.extend(by_sg[sg][:n_keep])
        selected = trimmed[:TARGET_COUNT + 200]

    print(f"Selected {len(selected)} diverse structures "
          f"({len(by_type)} distinct (SG, nsites) types)")

    # Cache to disk for reuse
    cache = []
    for doc in selected:
        struct = doc.structure
        cache.append({
            'material_id': str(doc.material_id),
            'formula': doc.formula_pretty,
            'nsites': doc.nsites,
            'sg_number': doc.symmetry.number,
            'sg_symbol': doc.symmetry.symbol,
            'lattice': struct.lattice.matrix.tolist(),
            'frac_coords': struct.frac_coords.tolist(),
            'species': [str(s) for s in struct.species],
        })

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    print(f"Cached to {CACHE_FILE}")

    return cache


def load_structures():
    """Load structures from cache or download."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached structures from {CACHE_FILE}")
        with open(CACHE_FILE) as f:
            return json.load(f)
    return download_structures()


def probe_atom(cryst, atom_idx, max_equidistant=7):
    """Probe a single atom for Voronoi complexity.

    Returns None if any vertex has too many equidistant atoms
    (resolution enumeration would be too slow).
    """
    vertices, central_idx, coords, images = analyze_voronoi(
        cryst, atom_index=atom_idx)
    degen = [v for v in vertices if v.is_degenerate]
    generic = [v for v in vertices if not v.is_degenerate]

    # Quick check: skip resolution enumeration for high-order vertices
    for v in degen:
        if v.n_equidistant > max_equidistant:
            return None

    n_res_list = []
    for v in degen:
        order = [central_idx] + [a for a in v.atom_indices
                                 if a != central_idx]
        pts = coords[order]
        resolutions, _ = enumerate_resolutions(pts, central=0)
        n_res_list.append(len(resolutions))

    combos = prod(n_res_list) if n_res_list else 1

    # Compute minimum relative gap to nearest non-equidistant atom
    min_gap_rel = float('inf')
    for v in vertices:
        if v.circumradius <= 0:
            continue
        eq_set = set(v.atom_indices)
        dists = np.linalg.norm(coords - v.position, axis=1)
        max_eq_dist = max(dists[a] for a in v.atom_indices)
        other_dists = sorted(dists[i] for i in range(len(dists))
                             if i not in eq_set)
        if other_dists:
            gap_rel = (other_dists[0] - max_eq_dist) / v.circumradius
            if gap_rel < min_gap_rel:
                min_gap_rel = gap_rel

    return {
        'n_generic': len(generic),
        'n_degen': len(degen),
        'n_res': n_res_list,
        'combos': combos,
        'min_gap_rel': min_gap_rel,
    }


def run_structure(name, cryst, atom_gap_info=None):
    """Run enumeration for all atoms and save filter file.

    Parameters
    ----------
    atom_gap_info : dict, optional
        Maps atom_idx -> min_gap_rel from the probe step.
    """
    per_atom_types = []

    for atom_idx in range(cryst.n_atoms):
        vertices, central_idx, coords, images = analyze_voronoi(
            cryst, atom_index=atom_idx)
        degen = [v for v in vertices if v.is_degenerate]

        if not degen:
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
            per_atom_types.append((atom_idx, {wv: {
                'p_vector': pv,
                'n_faces': len(faces),
                'count': 1,
                'face_neighbors': face_nbrs,
            }}))
            continue

        cell_types = enumerate_cell_types_v2(
            vertices, central_idx, coords, verbose=False,
            crystal=cryst, n_workers=N_WORKERS,
        )
        per_atom_types.append((atom_idx, cell_types))

    # Compute overall minimum gap across all atoms
    min_gap_rel = None
    if atom_gap_info:
        gaps = [g for g in atom_gap_info.values()
                if g < float('inf')]
        if gaps:
            min_gap_rel = min(gaps)

    # Write filter file
    filter_path = os.path.join(FILTER_DIR, f"{name}.filter")
    n_wv, n_groups = write_filter_file(
        filter_path, name, per_atom_types, species=cryst.species,
        min_gap_rel=min_gap_rel)
    return n_wv, n_groups


def get_memory_gb():
    """Return current RSS memory usage in GB."""
    # ru_maxrss is in bytes on Linux, kilobytes on macOS
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return ru.ru_maxrss / (1024 ** 3)
    else:
        return ru.ru_maxrss / (1024 ** 2)


def set_memory_limit(gb):
    """Set a hard address-space limit.  Allocations beyond this raise MemoryError."""
    limit_bytes = int(gb * 1024 ** 3)
    # RLIMIT_AS (address space) is the portable way to cap memory;
    # exceeding it makes malloc return NULL, which Python turns into MemoryError.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, resource.error) as e:
        print(f"Warning: could not set memory limit ({e}). "
              f"Memory will not be capped.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate VoroTop filter files for crystal structures.")
    parser.add_argument('--max-memory', type=float, default=MAX_MEMORY_GB,
                        help=f"Memory limit in GB (default: {MAX_MEMORY_GB})")
    args = parser.parse_args()

    max_mem_gb = args.max_memory
    set_memory_limit(max_mem_gb)

    os.makedirs(FILTER_DIR, exist_ok=True)

    structures = load_structures()
    print(f"\nStructures to process: {len(structures)}")
    print(f"Filter directory: {FILTER_DIR}")
    print(f"Max resolutions per vertex: {MAX_RES_PER_VERTEX}")
    print(f"Memory limit: {max_mem_gb:.1f} GB")
    print(f"Workers: {N_WORKERS}")
    print(f"{'=' * 80}\n")
    sys.stdout.flush()

    results = []
    skipped = []
    failed = []

    for i, entry in enumerate(structures):
        mid = entry['material_id']
        formula = entry['formula']
        sg = entry['sg_symbol']
        nsites = entry['nsites']
        name = f"{mid}_{formula}".replace(" ", "").replace("/", "-")

        filter_path = os.path.join(FILTER_DIR, f"{name}.filter")

        cryst = Crystal(
            np.array(entry['lattice']),
            entry['frac_coords'],
            entry['species'],
        )

        # Probe complexity
        try:
            max_combos = 0
            max_res = 0
            atom_summaries = []
            feasible = True
            for atom_idx in range(cryst.n_atoms):
                info = probe_atom(cryst, atom_idx)
                if info is None:
                    # Too many equidistant atoms — probe would be slow
                    feasible = False
                    atom_summaries.append({'n_generic': 0, 'n_degen': 0,
                                           'n_res': [], 'combos': 0})
                    break
                atom_summaries.append(info)
                max_combos = max(max_combos, info['combos'])
                if info['n_res']:
                    max_res = max(max_res, max(info['n_res']))
                if max_res > MAX_RES_PER_VERTEX:
                    feasible = False
                    break
                if info['combos'] > MAX_TOTAL_COMBOS:
                    feasible = False
                    break
        except Exception as e:
            failed.append((name, formula, sg, str(e)[:80]))
            continue

        if not feasible:
            if max_res > MAX_RES_PER_VERTEX:
                reason = f"too many resolutions per vertex (max {max_res})"
            elif max_combos > MAX_TOTAL_COMBOS:
                reason = (f"too many total combinations "
                          f"(max {max_combos:.1e})")
            else:
                reason = "high vertex degeneracy (>7 equidistant atoms)"
            skipped.append((name, formula, sg, max_combos))
            write_intractable_filter(filter_path, name, reason)
            print(f"[{i+1}/{len(structures)}] {mid} {formula:12s} "
                  f"SG {sg:12s}  INTRACTABLE ({reason})")
            sys.stdout.flush()
            continue

        # Run enumeration
        try:
            t0 = time.time()
            # Collect per-atom gap info from probe results
            atom_gap_info = {}
            for summary in atom_summaries:
                idx = atom_summaries.index(summary)
                if 'min_gap_rel' in summary:
                    atom_gap_info[idx] = summary['min_gap_rel']
            n_wv, n_groups = run_structure(name, cryst,
                                           atom_gap_info=atom_gap_info)
            elapsed = time.time() - t0

            results.append((name, formula, sg, n_groups, n_wv, elapsed))

            mem_gb = get_memory_gb()
            if (i + 1) % 10 == 0 or elapsed > 10:
                degen_info = ", ".join(
                    f"{a['n_degen']}dg" for a in atom_summaries
                    if a['n_degen'] > 0)
                if not degen_info:
                    degen_info = "no degen"
                print(f"[{i+1}/{len(structures)}] {mid} {formula:12s} "
                      f"SG {sg:12s}  {n_groups:2d} grp  {n_wv:5d} WV  "
                      f"{elapsed:6.1f}s  {mem_gb:.1f}GB  ({degen_info})")
                sys.stdout.flush()

        except MemoryError:
            elapsed = time.time() - t0
            mem_gb = get_memory_gb()
            skipped.append((name, formula, sg, max_combos))
            print(f"[{i+1}/{len(structures)}] {mid} {formula:12s} "
                  f"SG {sg:12s}  OUT OF MEMORY after {elapsed:.0f}s "
                  f"({mem_gb:.1f}GB)")
            sys.stdout.flush()
            partial = os.path.join(FILTER_DIR, f"{name}.filter")
            if os.path.exists(partial):
                os.remove(partial)
            write_intractable_filter(
                filter_path, name,
                f"exceeded {max_mem_gb:.0f}GB memory limit "
                f"after {elapsed:.0f}s")
        except Exception as e:
            failed.append((name, formula, sg, str(e)[:80]))
            if len(failed) <= 10:
                print(f"[{i+1}/{len(structures)}] {mid} {formula:12s} "
                      f"SG {sg:12s}  FAILED: {str(e)[:60]}")
                sys.stdout.flush()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Completed: {len(results)}")
    print(f"Skipped (too large): {len(skipped)}")
    print(f"Failed: {len(failed)}")

    total_wv = sum(nwv for _, _, _, _, nwv, _ in results if nwv > 0)
    print(f"Total Weinberg vectors across all filters: {total_wv:,}")

    n_filters = len([f for f in os.listdir(FILTER_DIR) if f.endswith('.filter')])
    print(f"Filter files in {FILTER_DIR}: {n_filters}")

    # Print skipped structures
    if skipped:
        print(f"\nSkipped structures (top 20 by combos):")
        for name, formula, sg, combos in sorted(
                skipped, key=lambda x: -x[3])[:20]:
            print(f"  {formula:15s} SG {sg:12s}  combos={combos:.1e}")

    # Print failures
    if failed:
        print(f"\nFailed structures:")
        for name, formula, sg, err in failed[:20]:
            print(f"  {formula:15s} SG {sg:12s}  {err}")

    sys.stdout.flush()


if __name__ == '__main__':
    main()
