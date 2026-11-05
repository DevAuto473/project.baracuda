// Barracuda NNUE - full-precision parameters (training master copy + on-disk format).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "barracuda/nnue/architecture.h"

namespace barracuda::nnue {

struct FloatNetwork {
    std::vector<float> ft_w;  // [kInputs][kHidden]
    std::vector<float> ft_b;  // [kHidden]
    std::vector<float> l1_w;  // [kL1][2*kHidden]
    std::vector<float> l1_b;  // [kL1]
    std::vector<float> l2_w;  // [kL2][kL1]
    std::vector<float> l2_b;  // [kL2]
    std::vector<float> out_w; // [kL2]
    float out_b = 0.0f;

    std::uint32_t epochs_trained = 0;
    std::uint64_t samples_trained = 0;

    FloatNetwork();

    void init_random(std::uint64_t seed);
    void clamp_weights();

    // File format: "BRCDNNUE" | u32 version | u32 dims[4] | u32 epochs | u64 samples | float32 tensors (LE)
    bool save(const std::string& path, std::string* error = nullptr) const;
    bool load(const std::string& path, std::string* error = nullptr);

    // Float forward pass from sparse features. Mirrors the quantised network exactly
    // (modulo rounding). Returns the raw output (1.0 == kOutputScale centipawns, stm-relative).
    struct Trace {
        float acc[2][kHidden];   // [stm, nstm] pre-activation
        float x[2 * kHidden];    // clipped FT output
        float z1[kL1], h1[kL1];
        float z2[kL2], h2[kL2];
        float out;
    };
    float forward(const int* stm_features, int n_stm, const int* nstm_features, int n_nstm, Trace* trace = nullptr) const;
};

} // namespace barracuda::nnue
