#!/usr/bin/env bash
set -euo pipefail

WHEEL="$1"
DEST_DIR="$2"
DELOCATE_ARCHS="$3"

# Step 1: delocate, excluding libomp
delocate-wheel --require-archs "$DELOCATE_ARCHS" -w "$DEST_DIR" -v --exclude libomp.dylib "$WHEEL"

# Step 2: rewrite libomp path to @rpath
cd "$DEST_DIR"
for whl in *.whl; do
    unzip -o -d _tmpwhl "$whl"

    for so in $(find _tmpwhl -name '*.so'); do
        old=$(otool -L "$so" | grep -o '[^ ]*libomp[^ ]*' | head -1 || true)
        if [ -n "$old" ]; then
            echo "Rewriting $old -> @rpath/libomp.dylib in $so"
            install_name_tool -change "$old" "@rpath/libomp.dylib" "$so"
        fi
    done

    rm "$whl"
    (cd _tmpwhl && zip -r "../$whl" .)
    rm -rf _tmpwhl
done
