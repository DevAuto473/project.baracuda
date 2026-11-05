// Barracuda - hand-crafted fallback evaluation (tapered material + PST + pawn structure).
// Used before an NNUE network has been trained, and as the bootstrap teacher for self-play.
#pragma once

#include "barracuda/engine/position.h"

namespace barracuda::eval {

Value classical(const Position& pos); // side-to-move relative, centipawns

} // namespace barracuda::eval
