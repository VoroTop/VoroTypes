#!/bin/bash
# Regression test: Python and C++ enumerators must produce bit-identical
# .filter files for any single-site enumeration both can complete.
#
# C++ exclusive features (--count-only, --no-weinberg, --hash-count) are
# not tested here because they have no Python equivalent by design.
#
# Usage: ./test_regression.sh

set -u
cd "$(dirname "$0")"

if [ ! -x ./voronoi_core ]; then
    echo "Building C++ enumerator..."
    g++ -O3 -fopenmp -std=c++17 -o voronoi_core \
        voronoi_enumerate/core/voronoi_core.cpp || exit 1
fi

TMP=$(mktemp -d)
PASS=0; FAIL=0

run_case() {
    local name="$1" pyflag="$2" expflag="$3" atom="$4"
    local label="$name $pyflag (atom=$atom)"

    python3 enumerate_cells.py "$name" $pyflag --atoms "$atom" \
        > "$TMP/py.log" 2>&1 || { echo "  ERROR py: $label"; FAIL=$((FAIL+1)); return; }
    mv -f "${name}.filter" "$TMP/py.filter"

    python3 export_enum.py "$name" $expflag --atoms "$atom" \
        -o "$TMP/in.bin" > "$TMP/exp.log" 2>&1 || \
        { echo "  ERROR export: $label"; FAIL=$((FAIL+1)); return; }

    ./voronoi_core "$TMP/in.bin" -o "$TMP/cpp.filter" -n "$name" \
        > "$TMP/cpp.log" 2>&1 || \
        { echo "  ERROR cpp: $label"; FAIL=$((FAIL+1)); return; }

    if diff -q "$TMP/py.filter" "$TMP/cpp.filter" > /dev/null; then
        local n=$(grep -c '^[0-9]' "$TMP/py.filter")
        printf "  PASS: %-40s %5d types\n" "$label" "$n"
        PASS=$((PASS+1))
    else
        echo "  FAIL: $label"
        diff "$TMP/py.filter" "$TMP/cpp.filter" | head -5
        FAIL=$((FAIL+1))
    fi
}

echo "=== Regression: Python vs C++ filter equivalence ==="
run_case fcc "--primary" "--primary" 0
run_case fcc ""          ""          0
run_case hcp "--primary" "--primary" 0
run_case hcp ""          ""          0

rm -rf "$TMP"
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
