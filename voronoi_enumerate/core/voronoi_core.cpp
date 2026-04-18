/*
 * voronoi_core.cpp — High-performance Voronoi cell type enumeration.
 *
 * Standalone C++ program that reads precomputed vertex options and symmetry
 * data from a binary file, then enumerates all canonical cell types using
 * orderly generation (branch-and-bound DFS) with OpenMP parallelization.
 *
 * Build:
 *   g++ -O3 -fopenmp -std=c++17 -o voronoi_core voronoi_core.cpp
 *
 * Usage:
 *   ./voronoi_core input.bin -j 64 [-o output.bin] [--count-only]
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

// ===================================================================
// Constants and types
// ===================================================================

static constexpr int MAX_V = 128;
static constexpr int MAX_DARTS = MAX_V * MAX_V;
static constexpr int MAX_DEPTH = 16;
static constexpr int MAX_SYM = 192;

using Tet = std::array<int, 4>;

struct ActiveElem {
    int gi;
    int resume;
};

inline int dart_id(int u, int v) { return u * MAX_V + v; }
inline int dart_u(int id) { return id / MAX_V; }
inline int dart_v(int id) { return id % MAX_V; }

// ===================================================================
// Enumeration data (loaded from binary file)
// ===================================================================

struct EnumData {
    int n_vertices;
    int central_idx;
    std::vector<Tet> generic_tets;
    // vertex_options[vi][ri] = list of tets
    std::vector<std::vector<std::vector<Tet>>> vertex_options;
    std::vector<int> n_res;

    // Symmetry
    int sym_order;
    std::vector<std::vector<int>> vertex_perms;   // [gi][vi] -> vj
    std::vector<std::vector<int>> vperm_inv;       // [gi][vj] -> vi
    std::vector<std::vector<std::vector<int>>> res_perms; // [gi][vi][ri] -> rj

    // Flattened lookup tables for hot path (cache-friendly)
    int max_n_res;
    std::vector<int> vperm_inv_flat;   // [gi * n_vertices + vj] -> vi
    std::vector<int> res_perms_flat;   // [gi * n_vertices * max_n_res + vi * max_n_res + ri] -> rj
};

// ===================================================================
// Binary I/O
// ===================================================================

static bool read_int(FILE* f, int& val) {
    int32_t v;
    if (fread(&v, 4, 1, f) != 1) return false;
    val = v;
    return true;
}

static bool write_int(FILE* f, int val) {
    int32_t v = val;
    return fwrite(&v, 4, 1, f) == 1;
}

static bool load_data(const char* path, EnumData& data) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VORO", 4) != 0) {
        fprintf(stderr, "Invalid file format\n"); fclose(f); return false;
    }

    read_int(f, data.n_vertices);
    read_int(f, data.central_idx);

    int n_gen;
    read_int(f, n_gen);
    data.generic_tets.resize(n_gen);
    for (int i = 0; i < n_gen; i++)
        for (int j = 0; j < 4; j++)
            read_int(f, data.generic_tets[i][j]);

    data.vertex_options.resize(data.n_vertices);
    data.n_res.resize(data.n_vertices);
    for (int vi = 0; vi < data.n_vertices; vi++) {
        int nr;
        read_int(f, nr);
        data.n_res[vi] = nr;
        data.vertex_options[vi].resize(nr);
        for (int ri = 0; ri < nr; ri++) {
            int nt;
            read_int(f, nt);
            data.vertex_options[vi][ri].resize(nt);
            for (int ti = 0; ti < nt; ti++)
                for (int j = 0; j < 4; j++)
                    read_int(f, data.vertex_options[vi][ri][ti][j]);
        }
    }

    read_int(f, data.sym_order);
    data.vertex_perms.resize(data.sym_order);
    for (int gi = 0; gi < data.sym_order; gi++) {
        data.vertex_perms[gi].resize(data.n_vertices);
        for (int vi = 0; vi < data.n_vertices; vi++)
            read_int(f, data.vertex_perms[gi][vi]);
    }

    data.res_perms.resize(data.sym_order);
    for (int gi = 0; gi < data.sym_order; gi++) {
        data.res_perms[gi].resize(data.n_vertices);
        for (int vi = 0; vi < data.n_vertices; vi++) {
            data.res_perms[gi][vi].resize(data.n_res[vi]);
            for (int ri = 0; ri < data.n_res[vi]; ri++)
                read_int(f, data.res_perms[gi][vi][ri]);
        }
    }

    // Precompute inverse vertex permutations
    data.vperm_inv.resize(data.sym_order);
    for (int gi = 0; gi < data.sym_order; gi++) {
        data.vperm_inv[gi].resize(data.n_vertices);
        for (int i = 0; i < data.n_vertices; i++)
            data.vperm_inv[gi][data.vertex_perms[gi][i]] = i;
    }

    // Flatten lookup tables for cache-friendly access in hot path
    int nv = data.n_vertices;
    data.max_n_res = *std::max_element(data.n_res.begin(), data.n_res.end());
    int mnr = data.max_n_res;

    data.vperm_inv_flat.resize(data.sym_order * nv);
    for (int gi = 0; gi < data.sym_order; gi++)
        for (int vi = 0; vi < nv; vi++)
            data.vperm_inv_flat[gi * nv + vi] = data.vperm_inv[gi][vi];

    data.res_perms_flat.resize(data.sym_order * nv * mnr);
    for (int gi = 0; gi < data.sym_order; gi++)
        for (int vi = 0; vi < nv; vi++)
            for (int ri = 0; ri < data.n_res[vi]; ri++)
                data.res_perms_flat[gi * nv * mnr + vi * mnr + ri] =
                    data.res_perms[gi][vi][ri];

    fclose(f);
    return true;
}

// ===================================================================
// Fixed-size structures for zero-allocation hot path
// ===================================================================

static constexpr int MAX_TETS = 128;
static constexpr int MAX_FACES = 32;
static constexpr int MAX_FACE_TETS = 32;
static constexpr int MAX_NEIGHBORS = 64;

struct StarFixed {
    Tet tets[MAX_TETS];
    int n;
};

struct FaceFixed {
    int verts[MAX_FACE_TETS];
    int n;
};

struct FacesFixed {
    FaceFixed faces[MAX_FACES];
    int n;
};

// ===================================================================
// star_to_faces (zero-allocation version)
// ===================================================================

static void star_to_faces_fast(const StarFixed& star, int central,
                               FacesFixed& result) {
    result.n = 0;
    int n_tets = star.n;

    // Collect unique neighbors
    int neighbors[MAX_NEIGHBORS];
    int n_nbrs = 0;
    for (int i = 0; i < n_tets; i++)
        for (int j = 0; j < 4; j++) {
            int v = star.tets[i][j];
            if (v == central) continue;
            bool found = false;
            for (int k = 0; k < n_nbrs; k++)
                if (neighbors[k] == v) { found = true; break; }
            if (!found) neighbors[n_nbrs++] = v;
        }

    // Sort neighbors
    std::sort(neighbors, neighbors + n_nbrs);

    for (int ni = 0; ni < n_nbrs; ni++) {
        int a = neighbors[ni];
        // Find tets containing both central and a
        int fidx[MAX_FACE_TETS];
        int nf = 0;
        for (int i = 0; i < n_tets; i++) {
            bool hc = false, ha = false;
            for (int j = 0; j < 4; j++) {
                if (star.tets[i][j] == central) hc = true;
                if (star.tets[i][j] == a) ha = true;
            }
            if (hc && ha) fidx[nf++] = i;
        }
        if (nf < 3) continue;

        // Adjacency (each node has exactly degree 2 in a valid face)
        int adj[MAX_FACE_TETS][2];
        int nadj[MAX_FACE_TETS];
        memset(nadj, 0, nf * sizeof(int));

        for (int i = 0; i < nf; i++) {
            for (int j = i + 1; j < nf; j++) {
                int common = 0;
                bool hc2 = false, ha2 = false;
                for (int u = 0; u < 4; u++)
                    for (int v = 0; v < 4; v++)
                        if (star.tets[fidx[i]][u] == star.tets[fidx[j]][v]) {
                            common++;
                            if (star.tets[fidx[i]][u] == central) hc2 = true;
                            if (star.tets[fidx[i]][u] == a) ha2 = true;
                        }
                if (common == 3 && hc2 && ha2) {
                    if (nadj[i] < 2) adj[i][nadj[i]++] = j;
                    if (nadj[j] < 2) adj[j][nadj[j]++] = i;
                }
            }
        }

        bool valid = true;
        for (int i = 0; i < nf; i++)
            if (nadj[i] != 2) { valid = false; break; }
        if (!valid) continue;

        // Trace cycle
        int cycle[MAX_FACE_TETS];
        cycle[0] = 0;
        cycle[1] = adj[0][0] < adj[0][1] ? adj[0][0] : adj[0][1];
        int prev = 0, cur = cycle[1];
        int cn = 2;

        valid = true;
        for (int step = 0; step < nf - 2; step++) {
            int next = (adj[cur][0] != prev) ? adj[cur][0] : adj[cur][1];
            if (next < 0) { valid = false; break; }
            prev = cur; cur = next;
            cycle[cn++] = cur;
        }
        if (!valid) continue;
        if (adj[cur][0] != 0 && adj[cur][1] != 0) continue;

        FaceFixed& face = result.faces[result.n];
        face.n = nf;
        for (int i = 0; i < nf; i++)
            face.verts[i] = fidx[cycle[i]];
        result.n++;
    }
}

// ===================================================================
// orient_faces (zero-allocation version)
// ===================================================================

static void orient_faces_fast(FacesFixed& ff) {
    int n = ff.n;
    if (n <= 1) return;

    // Edge map: for each undirected edge, store up to 2 face entries
    struct EdgeEntry { int fi, u, v; };
    struct EdgeBucket { EdgeEntry entries[2]; int count; };

    // Use a simple hash table for edges
    static constexpr int EDGE_HASH = 4096;
    EdgeBucket edge_table[EDGE_HASH];
    memset(edge_table, 0, sizeof(edge_table));

    for (int i = 0; i < n; i++) {
        int fn = ff.faces[i].n;
        for (int j = 0; j < fn; j++) {
            int u = ff.faces[i].verts[j];
            int v = ff.faces[i].verts[(j+1) % fn];
            int lo = u < v ? u : v, hi = u < v ? v : u;
            int h = (lo * 131 + hi) & (EDGE_HASH - 1);
            // Linear probe to find or create bucket
            for (int p = 0; p < EDGE_HASH; p++) {
                int idx = (h + p) & (EDGE_HASH - 1);
                EdgeBucket& b = edge_table[idx];
                if (b.count == 0) {
                    b.entries[0] = {i, u, v};
                    b.count = 1;
                    break;
                }
                int elo = b.entries[0].u < b.entries[0].v
                          ? b.entries[0].u : b.entries[0].v;
                int ehi = b.entries[0].u < b.entries[0].v
                          ? b.entries[0].v : b.entries[0].u;
                if (elo == lo && ehi == hi) {
                    if (b.count < 2)
                        b.entries[b.count++] = {i, u, v};
                    break;
                }
            }
        }
    }

    // BFS for consistent orientation
    bool flipped[MAX_FACES] = {};
    bool visited[MAX_FACES] = {};
    int queue[MAX_FACES];
    int qhead = 0, qtail = 0;
    visited[0] = true;
    queue[qtail++] = 0;

    while (qhead < qtail) {
        int fi = queue[qhead++];
        int fn = ff.faces[fi].n;
        for (int j = 0; j < fn; j++) {
            int u = ff.faces[fi].verts[j];
            int v = ff.faces[fi].verts[(j+1) % fn];
            int lo = u < v ? u : v, hi = u < v ? v : u;
            int h = (lo * 131 + hi) & (EDGE_HASH - 1);
            for (int p = 0; p < EDGE_HASH; p++) {
                int idx = (h + p) & (EDGE_HASH - 1);
                EdgeBucket& b = edge_table[idx];
                if (b.count == 0) break;
                int elo = b.entries[0].u < b.entries[0].v
                          ? b.entries[0].u : b.entries[0].v;
                int ehi = b.entries[0].u < b.entries[0].v
                          ? b.entries[0].v : b.entries[0].u;
                if (elo == lo && ehi == hi) {
                    if (b.count == 2) {
                        int fj = (b.entries[0].fi == fi)
                                 ? b.entries[1].fi : b.entries[0].fi;
                        if (!visited[fj]) {
                            visited[fj] = true;
                            bool same_dir =
                                (b.entries[0].u == b.entries[1].u);
                            flipped[fj] = same_dir
                                          ? !flipped[fi] : flipped[fi];
                            queue[qtail++] = fj;
                        }
                    }
                    break;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (flipped[i]) {
            int fn = ff.faces[i].n;
            for (int j = 0; j < fn / 2; j++) {
                int tmp = ff.faces[i].verts[j];
                ff.faces[i].verts[j] = ff.faces[i].verts[fn - 1 - j];
                ff.faces[i].verts[fn - 1 - j] = tmp;
            }
        }
    }
}

// ===================================================================
// Weinberg vector (zero-allocation version)
// ===================================================================

// Thread-local generation counters to avoid memset in hot path
struct WeinbergState {
    int sigma[MAX_DARTS];
    int dart_fs[MAX_DARTS];
    int sigma_gen[MAX_DARTS];
    int visited_gen[MAX_DARTS];
    int label_val[MAX_V];
    int label_gen[MAX_V];
    int cur_sigma_gen = 1;
    int cur_visit_gen = 1;
    int cur_label_gen = 1;
    int dart_list[MAX_DARTS];
    int n_darts = 0;

    WeinbergState() {
        memset(sigma_gen, 0, sizeof(sigma_gen));
        memset(visited_gen, 0, sizeof(visited_gen));
        memset(label_gen, 0, sizeof(label_gen));
    }

    void reset_sigma() { ++cur_sigma_gen; n_darts = 0; }
    void reset_visited() { ++cur_visit_gen; }
    void reset_labels() { ++cur_label_gen; }

    void set_sigma(int d, int val, int fs) {
        sigma[d] = val;
        dart_fs[d] = fs;
        sigma_gen[d] = cur_sigma_gen;
        dart_list[n_darts++] = d;
    }
    int get_sigma(int d) const {
        return (sigma_gen[d] == cur_sigma_gen) ? sigma[d] : -1;
    }

    void set_visited(int d) { visited_gen[d] = cur_visit_gen; }
    bool is_visited(int d) const { return visited_gen[d] == cur_visit_gen; }

    int assign_label(int v, int& next_label) {
        if (label_gen[v] != cur_label_gen) {
            label_gen[v] = cur_label_gen;
            label_val[v] = ++next_label;
        }
        return label_val[v];
    }
};

static thread_local WeinbergState ws;

static int build_sigma_fast(const FacesFixed& ff) {
    ws.reset_sigma();
    for (int fi = 0; fi < ff.n; fi++) {
        int fn = ff.faces[fi].n;
        for (int i = 0; i < fn; i++) {
            int u = ff.faces[fi].verts[i];
            int v = ff.faces[fi].verts[(i+1) % fn];
            int w = ff.faces[fi].verts[(i+2) % fn];
            int did = dart_id(u, v);
            if (ws.get_sigma(did) != -1) return -1;
            ws.set_sigma(did, dart_id(v, w), fn);
        }
    }
    return ws.n_darts;
}

static bool traverse_fast(int d0, int n_darts,
                          const int* best, int best_len,
                          int* code, int& code_len) {
    code_len = 0;
    ws.reset_visited();
    ws.reset_labels();

    int next_label = 0;
    bool tied = (best_len > 0);
    int pos = 0;

    auto emit = [&](int val) -> bool {
        code[code_len++] = val;
        if (tied && pos < best_len) {
            if (val > best[pos]) return false;
            if (val < best[pos]) tied = false;
        }
        pos++;
        return true;
    };

    ws.set_visited(d0);
    if (!emit(ws.assign_label(dart_u(d0), next_label))) return false;

    int d = d0;
    for (;;) {
        int nv = dart_v(d);
        int lbl = ws.assign_label(nv, next_label);
        if (!emit(lbl)) return false;

        if (lbl == next_label) {
            int nd = ws.get_sigma(d);
            if (nd < 0) return false;
            d = nd;
            ws.set_visited(d);
        } else {
            int start = dart_id(nv, dart_u(d));
            int scan = start;
            bool found = false;
            for (;;) {
                if (!ws.is_visited(scan)) {
                    d = scan;
                    ws.set_visited(d);
                    found = true;
                    break;
                }
                int rev = dart_id(dart_v(scan), dart_u(scan));
                int nxt = ws.get_sigma(rev);
                if (nxt < 0) return false;
                if (nxt == start) break;
                scan = nxt;
            }
            if (!found) break;
        }
    }
    return true;
}

static int weinberg_vector_fast(const FacesFixed& ff,
                                int* best, int& best_len) {
    best_len = 0;
    int code[MAX_DARTS + 1];

    FacesFixed oriented;

    for (int orient = 0; orient < 2; orient++) {
        oriented.n = ff.n;
        for (int fi = 0; fi < ff.n; fi++) {
            oriented.faces[fi].n = ff.faces[fi].n;
            if (orient == 0) {
                memcpy(oriented.faces[fi].verts, ff.faces[fi].verts,
                       ff.faces[fi].n * sizeof(int));
            } else {
                int fn = ff.faces[fi].n;
                for (int j = 0; j < fn; j++)
                    oriented.faces[fi].verts[j] =
                        ff.faces[fi].verts[fn - 1 - j];
            }
        }

        int nd = build_sigma_fast(oriented);
        if (nd < 0) return -1;

        // Sort dart list by face size for better pruning
        std::sort(ws.dart_list, ws.dart_list + nd,
                  [](int a, int b) { return ws.dart_fs[a] < ws.dart_fs[b]; });

        for (int di = 0; di < nd; di++) {
            int code_len;
            if (traverse_fast(ws.dart_list[di], nd,
                              best, best_len, code, code_len)) {
                if (best_len == 0 ||
                    std::lexicographical_compare(code, code + code_len,
                                                best, best + best_len)) {
                    memcpy(best, code, code_len * sizeof(int));
                    best_len = code_len;
                }
            }
        }
    }
    return 0;
}

static std::vector<int> p_vector_fast(const FacesFixed& ff) {
    std::vector<int> pv;
    pv.reserve(ff.n);
    for (int i = 0; i < ff.n; i++) pv.push_back(ff.faces[i].n);
    std::sort(pv.begin(), pv.end());
    return pv;
}

// ===================================================================
// classify_combo: star → faces → Weinberg vector
// ===================================================================

struct CellType {
    std::vector<int> wv;
    std::vector<int> pv;
    int n_faces;
};

static CellType classify_combo(const int* combo, const EnumData& data,
                               bool compute_weinberg = true) {
    // Build star into fixed-size array
    StarFixed star;
    star.n = 0;
    for (int i = 0; i < (int)data.generic_tets.size(); i++)
        star.tets[star.n++] = data.generic_tets[i];
    for (int vi = 0; vi < data.n_vertices; vi++) {
        const auto& tets = data.vertex_options[vi][combo[vi]];
        for (const auto& t : tets)
            star.tets[star.n++] = t;
    }
    std::sort(star.tets, star.tets + star.n);
    star.n = (int)(std::unique(star.tets, star.tets + star.n) - star.tets);

    FacesFixed ff;
    star_to_faces_fast(star, data.central_idx, ff);
    if (ff.n == 0) return {};

    auto pv = p_vector_fast(ff);

    if (!compute_weinberg)
        return {{}, std::move(pv), ff.n};

    orient_faces_fast(ff);
    int wv_buf[MAX_DARTS + 1];
    int wv_len;
    if (weinberg_vector_fast(ff, wv_buf, wv_len) < 0) return {};

    std::vector<int> wv(wv_buf, wv_buf + wv_len);
    return {std::move(wv), std::move(pv), ff.n};
}

// ===================================================================
// Orderly generation: branch-and-bound DFS
// ===================================================================

struct Subtree {
    std::vector<int> prefix;
    int n_active;
    ActiveElem active[MAX_SYM];
};

static void gen_subtrees_dfs(
        int depth, int split_depth,
        int* combo,
        ActiveElem active[], int n_active,
        const EnumData& data,
        std::vector<Subtree>& subtrees) {
    if (depth == split_depth) {
        Subtree st;
        st.prefix.assign(combo, combo + split_depth);
        st.n_active = n_active;
        memcpy(st.active, active, n_active * sizeof(ActiveElem));
        subtrees.push_back(std::move(st));
        return;
    }
    int nv = data.n_vertices;
    int mnr = data.max_n_res;
    const int* vpif = data.vperm_inv_flat.data();
    const int* rpf = data.res_perms_flat.data();

    ActiveElem new_active[MAX_SYM];
    for (int r = 0; r < data.n_res[depth]; r++) {
        combo[depth] = r;
        int new_n = 0;
        bool pruned = false;

        for (int a = 0; a < n_active; a++) {
            int gi = active[a].gi;
            int resume = active[a].resume;
            int status = 0;
            int new_resume = depth + 1;
            const int* vpi = vpif + gi * nv;
            const int* rp = rpf + gi * nv * mnr;

            for (int j = resume; j <= depth; j++) {
                int src = vpi[j];
                if (src > depth) {
                    new_resume = j;
                    break;
                }
                int img = rp[src * mnr + combo[src]];
                int cj = combo[j];
                if (img < cj) { status = -1; break; }
                if (img > cj) { status = 1; break; }
            }
            if (status == -1) { pruned = true; break; }
            if (status != 1) {
                new_active[new_n++] = {gi, new_resume};
            }
        }
        if (!pruned)
            gen_subtrees_dfs(depth + 1, split_depth, combo,
                             new_active, new_n, data, subtrees);
    }
}

static std::vector<Subtree> generate_subtrees(
        const EnumData& data, int split_depth) {
    int d = data.n_vertices;
    split_depth = std::min(split_depth, d);
    std::vector<Subtree> subtrees;
    int combo[MAX_DEPTH] = {};

    ActiveElem init[MAX_SYM];
    for (int gi = 0; gi < data.sym_order; gi++)
        init[gi] = {gi, 0};

    gen_subtrees_dfs(0, split_depth, combo, init, data.sym_order,
                     data, subtrees);
    return subtrees;
}

// Process one subtree: orderly DFS from prefix, classify each combo
struct SubtreeResult {
    int64_t n_canonical = 0;
    // p-vector string -> (n_types, total_orbit_count)
    std::map<std::vector<int>, std::pair<int64_t, int64_t>> pvec_hist;
    // Weinberg vectors (optional, for small enumerations)
    std::map<std::vector<int>, int64_t> wv_counts;
};

static void process_dfs(
        int depth, int d,
        int* combo,
        ActiveElem levels[][MAX_SYM],
        int* n_active_at,
        const int* vperm_inv_flat,
        const int* res_perms_flat,
        int nv, int mnr, int sym_order,
        const EnumData& data,
        bool compute_weinberg, bool count_only,
        SubtreeResult& result,
        std::atomic<int64_t>* global_classified) {
    if (depth == d) {
        result.n_canonical++;
        int orbit_size = sym_order / n_active_at[depth];
        if (count_only) return;

        auto ct = classify_combo(combo, data, compute_weinberg);
        if (global_classified) ++(*global_classified);
        if (ct.pv.empty()) return;

        auto& ph = result.pvec_hist[ct.pv];
        ph.first++;
        ph.second += orbit_size;

        if (compute_weinberg) {
            result.wv_counts[ct.wv] += orbit_size;
        }
        return;
    }

    const ActiveElem* cur = levels[depth];
    int cur_n = n_active_at[depth];
    ActiveElem* nxt = levels[depth + 1];

    for (int r = 0; r < data.n_res[depth]; r++) {
        combo[depth] = r;
        int new_n = 0;
        bool pruned = false;

        for (int a = 0; a < cur_n; a++) {
            int gi = cur[a].gi;
            int resume = cur[a].resume;
            int status = 0;
            int new_resume = depth + 1;
            const int* vpi = vperm_inv_flat + gi * nv;
            const int* rp = res_perms_flat + gi * nv * mnr;

            for (int j = resume; j <= depth; j++) {
                int src = vpi[j];
                if (src > depth) {
                    new_resume = j;
                    break;
                }
                int img = rp[src * mnr + combo[src]];
                int cj = combo[j];
                if (img < cj) { status = -1; break; }
                if (img > cj) { status = 1; break; }
            }
            if (status == -1) { pruned = true; break; }
            if (status != 1) {
                nxt[new_n++] = {gi, new_resume};
            }
        }
        if (!pruned) {
            n_active_at[depth + 1] = new_n;
            process_dfs(depth + 1, d, combo, levels, n_active_at,
                        vperm_inv_flat, res_perms_flat, nv, mnr, sym_order,
                        data, compute_weinberg, count_only, result,
                        global_classified);
        }
    }
}

static SubtreeResult process_subtree(
        const Subtree& st, const EnumData& data,
        bool compute_weinberg, bool count_only,
        std::atomic<int64_t>* global_classified = nullptr) {
    int d = data.n_vertices;
    int combo[MAX_DEPTH] = {};
    int start = (int)st.prefix.size();
    for (int i = 0; i < start; i++)
        combo[i] = st.prefix[i];

    ActiveElem levels[MAX_DEPTH + 1][MAX_SYM];
    int n_active_at[MAX_DEPTH + 1] = {};

    memcpy(levels[start], st.active, st.n_active * sizeof(ActiveElem));
    n_active_at[start] = st.n_active;

    SubtreeResult result;
    process_dfs(start, d, combo, levels, n_active_at,
                data.vperm_inv_flat.data(), data.res_perms_flat.data(),
                data.n_vertices, data.max_n_res, data.sym_order,
                data, compute_weinberg, count_only, result,
                global_classified);
    return result;
}

// ===================================================================
// Main
// ===================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.bin [-j N] [--count-only] [--no-weinberg]\n",
                argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    int n_workers = 1;
    bool count_only = false;
    bool compute_weinberg = true;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-j") == 0 && i+1 < argc)
            n_workers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--count-only") == 0)
            count_only = true;
        else if (strcmp(argv[i], "--no-weinberg") == 0)
            compute_weinberg = false;
    }

    // Load data
    EnumData data;
    if (!load_data(input_path, data)) return 1;

    printf("Loaded: %d vertices, %d symmetry order, central=%d\n",
           data.n_vertices, data.sym_order, data.central_idx);
    printf("Resolutions per vertex:");
    int64_t total = 1;
    for (int vi = 0; vi < data.n_vertices; vi++) {
        printf(" %d", data.n_res[vi]);
        total *= data.n_res[vi];
    }
    printf("\nTotal combinations: %lld\n", (long long)total);
    printf("Workers: %d\n", n_workers);
    if (count_only) printf("Mode: count-only\n");
    else if (!compute_weinberg) printf("Mode: p-vector histogram\n");
    else printf("Mode: full Weinberg enumeration\n");

    // Choose split depth: ~4× workers subtrees
    int split_depth = 0;
    {
        int64_t est = 1;
        while (split_depth < data.n_vertices && est < 64 * n_workers) {
            est *= data.n_res[split_depth];
            split_depth++;
        }
    }

    auto t0 = std::chrono::steady_clock::now();

    printf("Generating subtrees (split depth %d)...\n", split_depth);
    auto subtrees = generate_subtrees(data, split_depth);
    printf("  %d subtrees generated\n", (int)subtrees.size());

    // Process subtrees in parallel
    std::vector<SubtreeResult> results(subtrees.size());
    std::atomic<int64_t> progress{0};
    std::atomic<int64_t> classified{0};
    std::atomic<bool> done_flag{false};
    int64_t total_subtrees = (int64_t)subtrees.size();

    #ifdef _OPENMP
    omp_set_num_threads(n_workers);
    #endif

    auto t1 = std::chrono::steady_clock::now();

    // Background progress reporter
    std::thread reporter([&]() {
        int64_t last_cls = 0;
        auto last_time = t1;
        while (!done_flag.load(std::memory_order_relaxed)) {
            for (int i = 0; i < 50 && !done_flag.load(std::memory_order_relaxed); i++)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (done_flag.load(std::memory_order_relaxed)) break;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t1).count();
            int64_t done = progress.load(std::memory_order_relaxed);

            if (count_only) {
                int64_t canonical_so_far = 0;
                for (int64_t j = 0; j < std::min(done, total_subtrees); j++)
                    canonical_so_far += results[j].n_canonical;
                fprintf(stderr, "\r  %.1f%% (%lld/%lld subtrees, "
                        "~%lld canonical, %.1fs)",
                        100.0 * done / total_subtrees,
                        (long long)done, (long long)total_subtrees,
                        (long long)canonical_so_far, elapsed);
            } else {
                int64_t cls = classified.load(std::memory_order_relaxed);
                double dt = std::chrono::duration<double>(now - last_time).count();
                double rate = dt > 0 ? (cls - last_cls) / dt : 0;
                fprintf(stderr, "\r  %lld/%lld subtrees, "
                        "%lld classified, %.0f/s, %.1fs elapsed",
                        (long long)done, (long long)total_subtrees,
                        (long long)cls, rate, elapsed);
                last_cls = cls;
                last_time = now;
            }
            fflush(stderr);
        }
    });

    #pragma omp parallel for schedule(dynamic)
    for (int64_t si = 0; si < total_subtrees; si++) {
        results[si] = process_subtree(subtrees[si], data,
                                       compute_weinberg, count_only,
                                       count_only ? nullptr : &classified);
        ++progress;
    }

    done_flag.store(true);
    reporter.join();
    fprintf(stderr, "\n");

    // Merge results
    int64_t n_canonical = 0;
    std::map<std::vector<int>, std::pair<int64_t, int64_t>> pvec_hist;
    std::map<std::vector<int>, int64_t> wv_counts;

    for (auto& r : results) {
        n_canonical += r.n_canonical;
        for (auto& [pv, stats] : r.pvec_hist) {
            pvec_hist[pv].first += stats.first;
            pvec_hist[pv].second += stats.second;
        }
        if (compute_weinberg) {
            for (auto& [wv, cnt] : r.wv_counts)
                wv_counts[wv] += cnt;
        }
    }

    auto t2 = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t2 - t0).count();

    printf("\n=== Results ===\n");
    printf("Canonical combinations: %lld\n", (long long)n_canonical);

    if (!count_only) {
        int64_t n_types = 0;
        if (compute_weinberg) {
            n_types = (int64_t)wv_counts.size();
        } else {
            for (auto& [pv, s] : pvec_hist)
                n_types += s.first;
        }
        printf("Distinct cell types: %lld\n", (long long)n_types);

        printf("\np-vector histogram:\n");
        printf("%-50s  %10s  %15s\n", "p-vector", "# types", "total count");
        for (auto& [pv, stats] : pvec_hist) {
            std::string pvs = "(";
            for (size_t i = 0; i < pv.size(); i++) {
                if (i) pvs += ", ";
                pvs += std::to_string(pv[i]);
            }
            pvs += ")";
            printf("%-50s  %10lld  %15lld\n", pvs.c_str(),
                   (long long)stats.first, (long long)stats.second);
        }
    }

    printf("\nTotal time: %.1f seconds\n", total_time);

    // Write Weinberg vectors to binary file if computed
    if (compute_weinberg && !wv_counts.empty()) {
        std::string out_path = std::string(input_path);
        auto dot = out_path.rfind('.');
        if (dot != std::string::npos)
            out_path = out_path.substr(0, dot);
        out_path += "_results.bin";

        FILE* out = fopen(out_path.c_str(), "wb");
        if (out) {
            int32_t n_wv = (int32_t)wv_counts.size();
            fwrite("VRES", 1, 4, out);
            fwrite(&n_wv, 4, 1, out);
            for (auto& [wv, cnt] : wv_counts) {
                int32_t wv_len = (int32_t)wv.size();
                int32_t count = (int32_t)cnt;
                fwrite(&wv_len, 4, 1, out);
                for (int v : wv) {
                    int32_t iv = v;
                    fwrite(&iv, 4, 1, out);
                }
                fwrite(&count, 4, 1, out);
            }
            fclose(out);
            printf("Weinberg vectors written to: %s\n", out_path.c_str());
        }
    }

    return 0;
}
