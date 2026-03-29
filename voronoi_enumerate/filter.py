"""
Write VoroTop-compatible filter files.

A filter file maps Weinberg vectors to crystal type indices, allowing
VoroTop to classify particles in atomistic simulations.

Format:
    #<TAB><comment>
    *<TAB><type_index><TAB><type_name>
    <type_index><TAB>(<comma-separated Weinberg vector>)

For multi-atom unit cells, each inequivalent site gets its own type
index.  Atoms that produce identical Weinberg vector sets are merged
into a single type.
"""


def write_filter_file(filepath, crystal_name, per_atom_types, species=None,
                      min_gap_rel=None):
    """Write a VoroTop filter file from enumerated cell types.

    Parameters
    ----------
    filepath : str
        Output file path (typically ending in .filter).
    crystal_name : str
        Crystal structure name (e.g., "FCC", "BCC", "HCP").
    per_atom_types : list of (int, dict)
        Each entry is (atom_index, cell_types_dict) where cell_types_dict
        maps Weinberg vector (tuple) -> info dict.
    species : list of str, optional
        Species labels for all atoms in the unit cell, used for naming
        types in the filter header.
    min_gap_rel : float, optional
        Minimum relative gap to nearest non-equidistant atom across all
        vertices and atoms.  Used to warn about near-equidistant neighbors.
    """
    label = crystal_name.upper()

    # Group atoms that produce identical Weinberg vector sets
    groups = []  # list of (atom_indices, cell_types_dict, wv_frozenset)
    for atom_idx, ct in per_atom_types:
        wv_set = frozenset(ct.keys())
        merged = False
        for g in groups:
            if wv_set == g[2]:
                g[0].append(atom_idx)
                merged = True
                break
        if not merged:
            groups.append(([atom_idx], ct, wv_set))

    # Generate type labels
    if len(groups) == 1:
        type_labels = [label]
    else:
        type_labels = []
        for g_atoms, _, _ in groups:
            if species:
                sp = species[g_atoms[0]]
                type_labels.append(f"{label}-{sp}")
            else:
                type_labels.append(f"{label}-atom{g_atoms[0]}")
        # Disambiguate duplicate labels by appending atom index
        if len(set(type_labels)) < len(type_labels):
            type_labels = []
            for g_atoms, _, _ in groups:
                type_labels.append(f"{label}-atom{g_atoms[0]}")

    # Write filter file
    total_wv = sum(len(g[2]) for g in groups)

    with open(filepath, 'w') as f:
        f.write(f"#\t{label} filter, computed by voronoi_enumerate\n")
        f.write(f"#\tTotal Weinberg types: {total_wv}\n")
        if min_gap_rel is not None:
            f.write(f"#\tMin relative gap: {min_gap_rel:.6e}\n")
            if min_gap_rel < 1e-2:
                f.write(f"#\tWARNING: Near-equidistant neighbors detected "
                        f"(gap_rel={min_gap_rel:.2e}).\n")
                f.write(f"#\tFor perturbations sigma > {min_gap_rel:.0e}, "
                        f"supplement with stochastic filter:\n")
                f.write(f"#\t  VoroTop structure.xyz -mf 100000 <sigma>\n")
        for i, tl in enumerate(type_labels):
            atoms_str = ",".join(str(a) for a in groups[i][0])
            f.write(f"*\t{i + 1}\t{tl}\tatoms [{atoms_str}]\n")
        for i, (_, ct, _) in enumerate(groups):
            for wv in sorted(ct.keys()):
                wv_str = ",".join(str(x) for x in wv)
                f.write(f"{i + 1}\t({wv_str})\n")

    return total_wv, len(groups)


def write_intractable_filter(filepath, crystal_name, reason):
    """Write a filter file indicating that analytical enumeration is intractable.

    Parameters
    ----------
    filepath : str
        Output file path.
    crystal_name : str
        Crystal structure name.
    reason : str
        Brief explanation of why enumeration is intractable.
    """
    label = crystal_name.upper()
    with open(filepath, 'w') as f:
        f.write(f"#\t{label} filter — INTRACTABLE\n")
        f.write(f"#\tAnalytical enumeration was not feasible: {reason}\n")
        f.write(f"#\tUse VoroTop -mf to generate a stochastic filter instead.\n")
        f.write(f"#\tExample: VoroTop structure.xyz -mf 100000 0.0001\n")
