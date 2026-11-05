// Barracuda WebAssembly API.
//
// A tiny, stateless-per-call C interface used by web/js/engine-worker.js. Every function takes a
// start FEN plus a space-separated list of UCI moves played from it, so repetition and the
// 50-move rule see the whole game. Strings returned point into static buffers that stay valid
// until the next call.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"
#include "barracuda/engine/search.h"
#include "barracuda/nnue/inspector.h"
#include "barracuda/nnue/network.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define BRC_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
// Posts a search-progress message from the worker to the page.
EM_JS(void, brc_js_info, (int depth, int score, int nodes_k, int nps_k, int time_ms, const char* pv, const char* view), {
    postMessage({type : 'info', depth : depth, score : score, nodesK : nodes_k, npsK : nps_k, timeMs : time_ms, pv : UTF8ToString(pv), view : UTF8ToString(view)});
});
#else
#define BRC_EXPORT extern "C"
static void brc_js_info(int, int, int, int, int, const char*, const char*) {}
#endif

using namespace barracuda;

namespace {

std::unique_ptr<Search> g_search;
std::shared_ptr<const nnue::Network> g_net;
std::string g_out;
std::string g_view_json;
bool g_last_styled = false;
std::string g_last_objective;
std::string g_last_trap;
bool g_last_defending = false;
std::unique_ptr<Search> g_predictor; // separate small search: predicting never disturbs the main hash table
StyleOptions g_style;
nnue::NetworkView g_view;

// Compact JSON of what the network "sees" for `pos`: activations per layer and the edge weights
// between layers (the same data the desktop dashboard draws). Empty when no network is loaded.
const std::string& net_view_json(const Position& pos) {
    g_view_json.clear();
    if (!g_net) return g_view_json;
    nnue::build_network_view(*g_net, pos, g_view);
    std::string& s = g_view_json;
    char b[32];
    auto arr = [&](const std::vector<float>& a) {
        s += '[';
        for (std::size_t i = 0; i < a.size(); ++i) {
            std::snprintf(b, sizeof(b), "%.3f", double(a[i]));
            if (i) s += ',';
            s += b;
        }
        s += ']';
    };
    const auto& in = g_view.layers[0];
    s += "{\"n\":" + std::to_string(in.activation.size()) + ",\"labels\":[";
    for (std::size_t i = 0; i < in.labels.size(); ++i) s += (i ? ",\"" : "\"") + in.labels[i] + "\"";
    s += "],\"a1\":";
    arr(g_view.layers[1].activation);
    s += ",\"a2\":";
    arr(g_view.layers[2].activation);
    s += ",\"a3\":";
    arr(g_view.layers[3].activation);
    s += ",\"a4\":";
    arr(g_view.layers[4].activation);
    for (int k = 0; k < 4; ++k) {
        s += ",\"e" + std::to_string(k) + "\":";
        arr(g_view.edges[std::size_t(k)]);
    }
    s += ",\"cp\":" + std::to_string(g_view.eval_cp) + ",\"white\":" + std::to_string(g_view.eval_white_cp) + "}";
    return s;
}

Search& search() {
    if (!g_search) {
        Position::init();
        g_search = std::make_unique<Search>(32);
    }
    return *g_search;
}

// Builds the position; returns false and sets g_out to "!<reason>" on bad input / illegal move.
bool build(const char* fen, const char* moves, Position& pos) {
    Position::init();
    std::string err;
    if (!pos.set_fen(fen && *fen ? fen : Position::START_FEN, &err)) {
        g_out = "!" + err;
        return false;
    }
    std::istringstream ss(moves ? moves : "");
    std::string tok;
    while (ss >> tok) {
        const Move m = parse_uci_move(pos, tok);
        if (!m) {
            g_out = "!illegal move " + tok;
            return false;
        }
        pos.do_move(m);
    }
    return true;
}

} // namespace

BRC_EXPORT int brc_init() {
    search();
    return 1;
}

// Loads a .bnn network (path inside the wasm virtual file system). Returns 1 on success.
BRC_EXPORT int brc_load_net(const char* path) {
    std::string err;
    auto net = nnue::load_network(path, &err);
    if (!net) {
        g_out = err;
        return 0;
    }
    g_net = net;
    search().set_network(g_net);
    return 1;
}

BRC_EXPORT const char* brc_last_message() { return g_out.c_str(); }

// 0 = classical, 1 = NNUE (requires a loaded network)
BRC_EXPORT void brc_set_eval(int mode) {
    search().set_eval_mode(mode == 1 && g_net ? EvalMode::NNUE : EvalMode::CLASSICAL);
}

BRC_EXPORT const char* brc_simd() { return nnue::simd_backend(); }

// Strange style on/off (see StyleOptions in search.h).
BRC_EXPORT void brc_set_style(int enabled) {
    g_style.enabled = enabled != 0;
    search().set_style(g_style);
}
// Opponent model (0..100: how readily this player falls for natural-looking mistakes) and contempt
// (centipawns a draw costs Barracuda). See StyleOptions.
BRC_EXPORT void brc_set_opponent(int trap, int contempt) {
    g_style.trap = std::clamp(trap, 0, 100);
    g_style.contempt = std::clamp(contempt, 0, 200);
    g_style.hunt = true;
    g_style.aggression = 100; // hungry for the enemy king
    g_style.adaptive = true;   // ... and digs in when the game turns against it
    search().set_style(g_style);
}
// After brc_search: the natural-looking reply the move sets a trap for ("" when it is no trap).
BRC_EXPORT const char* brc_last_trap() { return g_last_trap.c_str(); }
// After brc_search: 1 when the move was chosen in defence mode, 0 in attack mode.
BRC_EXPORT int brc_last_defending() { return g_last_defending ? 1 : 0; }
// The move a human would most likely play after `moves` (UCI, "" when there is none).
BRC_EXPORT const char* brc_predict(const char* fen, const char* moves) {
    Position pos;
    if (!build(fen, moves, pos)) return "";
    if (!g_predictor) {
        g_predictor = std::make_unique<Search>(1);
        g_predictor->set_eval_mode(EvalMode::CLASSICAL);
    }
    // (the predictor never uses the style: it models the opponent, not Barracuda)
    const Move m = g_predictor->predict_human(pos, 2);
    g_out = m ? m.to_uci() : std::string();
    return g_out.c_str();
}
// After brc_search: 1 if the style replaced the plain best move, and that plain best move (UCI).
BRC_EXPORT int brc_last_styled() { return g_last_styled ? 1 : 0; }
BRC_EXPORT const char* brc_last_objective() { return g_last_objective.c_str(); }

BRC_EXPORT const char* brc_legal_moves(const char* fen, const char* moves) {
    Position pos;
    if (!build(fen, moves, pos)) return g_out.c_str();
    g_out.clear();
    for (Move m : legal_moves(pos)) g_out += m.to_uci() + " ";
    if (!g_out.empty()) g_out.pop_back();
    return g_out.c_str();
}

BRC_EXPORT const char* brc_fen(const char* fen, const char* moves) {
    Position pos;
    if (!build(fen, moves, pos)) return g_out.c_str();
    g_out = pos.fen();
    return g_out.c_str();
}

// "ongoing" | "1-0 checkmate" | "0-1 checkmate" | "1/2-1/2 stalemate" | "1/2-1/2 <reason>"
BRC_EXPORT const char* brc_status(const char* fen, const char* moves) {
    Position pos;
    if (!build(fen, moves, pos)) return g_out.c_str();
    std::string reason;
    switch (game_result(pos, &reason)) {
    case GameResult::ONGOING: g_out = "ongoing"; break;
    case GameResult::WHITE_WINS: g_out = "1-0 " + reason; break;
    case GameResult::BLACK_WINS: g_out = "0-1 " + reason; break;
    case GameResult::DRAW: g_out = "1/2-1/2 " + reason; break;
    }
    return g_out.c_str();
}

// Network activations/weights for the position after `moves` ("" when no network is loaded).
BRC_EXPORT const char* brc_net_view(const char* fen, const char* moves) {
    Position pos;
    if (!build(fen, moves, pos)) return "";
    return net_view_json(pos).c_str();
}

BRC_EXPORT int brc_has_net() { return g_net ? 1 : 0; }

BRC_EXPORT int brc_in_check(const char* fen, const char* moves) {
    Position pos;
    return build(fen, moves, pos) && pos.in_check() ? 1 : 0;
}

// SAN of `uci` played from the position after `moves`.
BRC_EXPORT const char* brc_san(const char* fen, const char* moves, const char* uci) {
    Position pos;
    if (!build(fen, moves, pos)) return g_out.c_str();
    const Move m = parse_uci_move(pos, uci ? uci : "");
    if (!m) {
        g_out = "!illegal move";
        return g_out.c_str();
    }
    g_out = move_to_san(pos, m);
    return g_out.c_str();
}

// Static evaluation (centipawns, side to move) with the active evaluator.
BRC_EXPORT int brc_static_eval(const char* fen, const char* moves) {
    Position pos;
    return build(fen, moves, pos) ? search().static_eval(pos) : 0;
}

BRC_EXPORT void brc_new_game() { search().new_game(); }

// Blocking search. Returns "<bestmove> <score_cp_white> <depth> <nodes> <pv...>"; the score is from
// White's point of view and mate scores are mapped to +/-30000. Progress is posted to the page.
BRC_EXPORT const char* brc_search(const char* fen, const char* moves, int movetime_ms, int max_depth) {
    Position pos;
    if (!build(fen, moves, pos)) return g_out.c_str();
    SearchLimits lim;
    lim.movetime_ms = std::max(10, movetime_ms);
    lim.depth = std::clamp(max_depth, 1, MAX_PLY - 2);
    lim.move_overhead_ms = 0;
    const Color stm = pos.side_to_move();

    const SearchResult r = search().go(pos, lim, [&](const SearchInfo& si) {
        std::string pv;
        for (Move m : si.pv) pv += m.to_uci() + " ";
        const int white_score = stm == WHITE ? si.score : -si.score;
        // the network evaluates the leaf of the line being analysed: show what it sees there
        Position leaf = pos;
        for (Move m : si.pv) {
            if (!legal_moves(leaf).contains(m)) break;
            leaf.do_move(m);
        }
        brc_js_info(si.depth, white_score, int(si.nodes / 1000), int(si.nps / 1000), int(si.time_ms), pv.c_str(),
                    net_view_json(leaf).c_str());
    });

    g_last_styled = r.styled;
    g_last_objective = r.objective_best ? r.objective_best.to_uci() : std::string();
    g_last_trap = r.trap_reply ? r.trap_reply.to_uci() : std::string();
    g_last_defending = r.defending;
    int score = stm == WHITE ? r.score : -r.score;
    if (is_mate_score(r.score)) score = score > 0 ? 30000 : -30000;
    std::string pv;
    for (Move m : r.pv) pv += " " + m.to_uci();
    g_out = (r.best ? r.best.to_uci() : std::string("none")) + " " + std::to_string(score) + " " + std::to_string(r.depth) +
            " " + std::to_string(r.nodes) + pv;
    return g_out.c_str();
}
