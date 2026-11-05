#!/usr/bin/env bash
# Builds the Barracuda WebAssembly module into web/wasm-dist/ (needs Emscripten: `emcc` on PATH).
# Usage: web/build.sh [extra emcc flags]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=web/wasm-dist
mkdir -p "$OUT"

SOURCES=(
  engine/src/bitboard.cpp engine/src/zobrist.cpp engine/src/position.cpp engine/src/movegen.cpp
  engine/src/notation.cpp engine/src/evaluate.cpp engine/src/tt.cpp engine/src/search.cpp
  nnue/src/float_network.cpp nnue/src/network.cpp nnue/src/inspector.cpp
  web/wasm/brc_api.cpp
)

EXPORTS='["_brc_init","_brc_load_net","_brc_last_message","_brc_set_eval","_brc_simd","_brc_legal_moves","_brc_fen","_brc_status","_brc_in_check","_brc_net_view","_brc_has_net","_brc_set_style","_brc_last_styled","_brc_last_objective","_brc_set_opponent","_brc_last_trap","_brc_last_defending","_brc_predict","_brc_san","_brc_static_eval","_brc_new_game","_brc_search","_malloc","_free"]'

# -msimd128 + -msse4.1 enables the SSE4.1 NNUE kernels (mapped to WASM SIMD by Emscripten).
# Pass SIMD=0 to build the scalar kernels for very old browsers.
SIMD_FLAGS=(-msimd128 -msse4.1)
if [ "${SIMD:-1}" = "0" ]; then SIMD_FLAGS=(); fi

STACK_FLAG="-sSTACK_SIZE=2097152"
# Emscripten < 3.1.27 used TOTAL_STACK instead
if emcc --version | head -1 | grep -Eq ' 3\.1\.([0-9]|1[0-9]|2[0-6])([^0-9]|$)'; then STACK_FLAG="-sTOTAL_STACK=2097152"; fi

emcc "${SOURCES[@]}" \
  -Iengine/include -Innue/include \
  -std=c++20 -O3 -flto -fno-exceptions -DNDEBUG \
  "${SIMD_FLAGS[@]}" \
  -sMODULARIZE=1 -sEXPORT_NAME=createBarracuda \
  -sENVIRONMENT=worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 "$STACK_FLAG" \
  -sFORCE_FILESYSTEM=1 \
  -sEXPORTED_FUNCTIONS="$EXPORTS" \
  -sEXPORTED_RUNTIME_METHODS='["cwrap","ccall","UTF8ToString","FS"]' \
  "$@" \
  -o "$OUT/barracuda.js"

ls -la "$OUT"
