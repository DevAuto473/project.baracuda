// Barracuda NNUE - activation/weight snapshot for the network visualiser.
// Produced on a worker thread, consumed (read-only) by the render thread.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "barracuda/nnue/network.h"

namespace barracuda::nnue {

struct NetworkView {
    struct Layer {
        std::string name;
        int total_nodes = 0;              // real width of the layer
        std::vector<float> activation;    // displayed nodes, normalised to [0, 1]
        std::vector<std::string> labels;  // optional per-node label
    };
    static constexpr int kLayers = 5; // inputs(active), FT(sampled), L1, L2, output

    bool valid = false;
    std::uint64_t serial = 0;
    int eval_cp = 0;       // stm-relative
    int eval_white_cp = 0; // white-relative
    bool trained = false;
    std::uint32_t epochs = 0;
    std::string fen;
    std::array<Layer, kLayers> layers;
    // edges[k][src * dst_count + dst] = signed weight (float scale) between layers k and k+1
    std::array<std::vector<float>, kLayers - 1> edges;
};

constexpr int kViewFtNodes = 32; // 16 sampled neurons per perspective

void build_network_view(const Network& net, const Position& pos, NetworkView& out);

} // namespace barracuda::nnue
