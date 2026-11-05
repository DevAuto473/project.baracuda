// Barracuda - Zobrist hashing keys.
#pragma once

#include "barracuda/engine/types.h"

namespace barracuda::zobrist {

extern Key PIECE_SQUARE[PIECE_NB][SQUARE_NB];
extern Key CASTLING[16];
extern Key EN_PASSANT_FILE[8];
extern Key SIDE_TO_MOVE;

void init(); // thread-safe, idempotent

} // namespace barracuda::zobrist
