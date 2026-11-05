#include "barracuda/nnue/inspector.h"

#include <algorithm>

namespace barracuda::nnue {

namespace {
std::string square_name(Square s) {
    std::string r;
    r += char('a' + file_of(s));
    r += char('1' + rank_of(s));
    return r;
}

// Sampled FT neuron for display slot j (0..kViewFtNodes-1). First half: side to move.
int ft_slot_neuron(int j) {
    constexpr int per_half = kViewFtNodes / 2;
    return (j % per_half) * (kHidden / per_half);
}
} // namespace

void build_network_view(const Network& net, const Position& pos, NetworkView& out) {
    Accumulator acc;
    net.refresh(pos, acc);
    Network::Trace tr;
    const Color stm = pos.side_to_move();
    out.eval_cp = net.evaluate(acc, stm, &tr);
    out.eval_white_cp = stm == WHITE ? out.eval_cp : -out.eval_cp;
    out.trained = net.is_trained();
    out.epochs = net.epochs_trained();
    out.fen = pos.fen();
    out.valid = true;
    ++out.serial;

    // ---- layer 0: active input features (pieces on board) -------------------
    auto& in = out.layers[0];
    in.name = "Input";
    in.total_nodes = kInputs;
    in.activation.clear();
    in.labels.clear();
    std::vector<std::pair<Piece, Square>> pieces;
    Bitboard occ = pos.pieces();
    while (occ) {
        const Square s = pop_lsb(occ);
        pieces.emplace_back(pos.piece_on(s), s);
    }
    // stable order: by piece then square, kings first for readability
    std::sort(pieces.begin(), pieces.end(), [](const auto& a, const auto& b) {
        const int ta = 5 - int(type_of(a.first)), tb = 5 - int(type_of(b.first));
        if (color_of(a.first) != color_of(b.first)) return color_of(a.first) < color_of(b.first);
        if (ta != tb) return ta < tb;
        return a.second < b.second;
    });
    for (const auto& [pc, sq] : pieces) {
        in.activation.push_back(1.0f);
        std::string l;
        l += color_of(pc) == WHITE ? 'w' : 'b';
        l += "PNBRQK"[type_of(pc)];
        l += ' ';
        l += square_name(sq);
        in.labels.push_back(l);
    }

    // ---- layer 1: sampled feature-transformer neurons -----------------------
    auto& ft = out.layers[1];
    ft.name = "Feature Transformer";
    ft.total_nodes = 2 * kHidden;
    ft.activation.assign(kViewFtNodes, 0.0f);
    ft.labels.assign(kViewFtNodes, {});
    for (int j = 0; j < kViewFtNodes; ++j) {
        const int half = j / (kViewFtNodes / 2);
        const int n = ft_slot_neuron(j);
        ft.activation[j] = float(tr.ft[half * kHidden + n]) / kQA;
        ft.labels[j] = std::string(half == 0 ? "stm#" : "opp#") + std::to_string(n);
    }

    // ---- layers 2..4 --------------------------------------------------------
    auto& l1 = out.layers[2];
    l1.name = "Hidden L1";
    l1.total_nodes = kL1;
    l1.activation.resize(kL1);
    l1.labels.assign(kL1, {});
    for (int i = 0; i < kL1; ++i) l1.activation[i] = float(tr.l1[i]) / kQA;

    auto& l2 = out.layers[3];
    l2.name = "Hidden L2";
    l2.total_nodes = kL2;
    l2.activation.resize(kL2);
    l2.labels.assign(kL2, {});
    for (int i = 0; i < kL2; ++i) l2.activation[i] = float(tr.l2[i]) / kQA;

    auto& o = out.layers[4];
    o.name = "Output";
    o.total_nodes = 1;
    // squash centipawns into [0,1] for display brightness
    o.activation = {std::clamp(0.5f + float(out.eval_cp) / 2000.0f, 0.0f, 1.0f)};
    o.labels = {std::to_string(out.eval_cp) + " cp"};

    // ---- edges --------------------------------------------------------------
    const int n_in = int(pieces.size());
    auto& e0 = out.edges[0];
    e0.assign(std::size_t(n_in) * kViewFtNodes, 0.0f);
    for (int i = 0; i < n_in; ++i) {
        const auto [pc, sq] = pieces[std::size_t(i)];
        for (int j = 0; j < kViewFtNodes; ++j) {
            const int half = j / (kViewFtNodes / 2);
            const Color persp = half == 0 ? stm : ~stm;
            e0[std::size_t(i) * kViewFtNodes + j] = net.ft_weight(feature_index(persp, pc, sq), ft_slot_neuron(j));
        }
    }
    auto& e1 = out.edges[1];
    e1.assign(std::size_t(kViewFtNodes) * kL1, 0.0f);
    for (int j = 0; j < kViewFtNodes; ++j) {
        const int half = j / (kViewFtNodes / 2);
        const int src = half * kHidden + ft_slot_neuron(j);
        for (int k = 0; k < kL1; ++k) e1[std::size_t(j) * kL1 + k] = net.l1_weight(k, src);
    }
    auto& e2 = out.edges[2];
    e2.assign(std::size_t(kL1) * kL2, 0.0f);
    for (int i = 0; i < kL1; ++i)
        for (int k = 0; k < kL2; ++k) e2[std::size_t(i) * kL2 + k] = net.l2_weight(k, i);
    auto& e3 = out.edges[3];
    e3.assign(kL2, 0.0f);
    for (int i = 0; i < kL2; ++i) e3[std::size_t(i)] = net.out_weight(i);
}

} // namespace barracuda::nnue
