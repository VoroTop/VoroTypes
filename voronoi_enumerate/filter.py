"""
Write VoroTop-compatible filter files.

A filter file maps topology vectors to crystal type indices, allowing
VoroTop to classify particles in atomistic simulations.  Topologies are
Weinberg vectors in three dimensions and p-vectors in two.

Format:
    #<TAB><comment>
    *<TAB><type_index><TAB><type_name>
    <type_index><TAB>(<comma-separated topology vector>)

For multi-atom unit cells, each inequivalent site gets its own type
index.  Atoms that produce identical vector sets are merged into a
single type.
"""


def write_filter_file(filepath, crystal_name, per_atom_types, species=None,
                      min_gap_rel=None, dimension=3):
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
    dimension : int, optional
        3 (default) for Weinberg vectors; 2 for p-vectors of a planar
        crystal, to be used with VoroTop's -2 option.
    """
    vector_name = "Weinberg" if dimension == 3 else "p-vector"
    label = crystal_name

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
        if dimension == 2:
            f.write("#\tTwo-dimensional filter: use with VoroTop -2\n")
        f.write(f"#\tTotal {vector_name} types: {total_wv}\n")
        if min_gap_rel is not None:
            f.write(f"#\tMin relative gap: {min_gap_rel:.6e}\n")
            if min_gap_rel < 1e-2:
                f.write(f"#\tWARNING: Near-equidistant neighbors detected "
                        f"(gap_rel={min_gap_rel:.2e}).\n")
                f.write(f"#\tThis filter may be incomplete for "
                        f"perturbations that close this gap; consider\n")
                if dimension == 3:
                    f.write(f"#\tregenerating with --near-gap-threshold >= "
                            f"{min_gap_rel:.0e}, or supplement with a\n")
                    f.write(f"#\tstochastic filter: "
                            f"VoroTop structure.xyz -mf 100000 <sigma>\n")
                else:
                    f.write(f"#\tsupplementing it with a stochastic filter: "
                            f"VoroTop structure.dump -2 -mf 100000 <sigma>\n")
        for i, tl in enumerate(type_labels):
            atoms_str = ",".join(str(a) for a in groups[i][0])
            f.write(f"*\t{i + 1}\t{tl}\tatoms [{atoms_str}]\n")
        for i, (_, ct, _) in enumerate(groups):
            for wv in sorted(ct.keys()):
                wv_str = ",".join(str(x) for x in wv)
                f.write(f"{i + 1}\t({wv_str})\n")

    return total_wv, len(groups)
