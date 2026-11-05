// Barracuda NNUE - network topology and quantisation constants.
//
//   768 binary inputs (2 colours x 6 pieces x 64 squares), per perspective
//        |  Feature Transformer (int16 weights, int16 accumulator), shared by both perspectives
//   2 x 256  ->  ClippedReLU [0,127]  -> concatenated [side-to-move | other side]  (512 x uint8)
//        |  Dense L1 (int8 weights, int32 sums, >>6)
//   32  -> ClippedReLU
//        |  Dense L2 (int8 weights)
//   32  -> ClippedReLU
//        |  Output (int8 weights)
//   1   -> centipawns
#pragma once

#include <cstdint>

#include "barracuda/engine/types.h"

namespace barracuda::nnue {

constexpr int kInputs = 768;
constexpr int kHidden = 256; // per perspective
constexpr int kL1 = 32;
constexpr int kL2 = 32;
constexpr int kMaxActive = 32; // max pieces on the board

// Quantisation scales.
constexpr int kQA = 127;       // FT output scale; ClippedReLU range is [0, kQA]
constexpr int kQB = 64;        // dense weight scale (int8)
constexpr int kQBShift = 6;    // log2(kQB)
constexpr int kOutputScale = 400; // network output 1.0 == 400cp (sigmoid(cp / 400) == win probability)

// Float-space clamps that keep quantised values in range.
constexpr float kDenseWeightLimit = 127.0f / kQB; // ~1.98 (int8)
constexpr float kFtWeightLimit = 4.0f;            // 127 * 4 * 33 < 32767 (int16 accumulator headroom)

// Feature index of `pc` on `sq`, seen from `perspective`.
// Each side sees its own pieces as "friendly" and the board flipped so that it always plays "up".
constexpr int feature_index(Color perspective, Piece pc, Square sq) {
    const int rel_sq = perspective == WHITE ? int(sq) : int(sq) ^ 56;
    const int rel_color = color_of(pc) == perspective ? 0 : 1;
    return rel_color * 384 + int(type_of(pc)) * 64 + rel_sq;
}

} // namespace barracuda::nnue
