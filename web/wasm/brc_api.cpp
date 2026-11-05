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
#include "barracuda/nnue/network.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define BRC_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
// Posts a search-progress message from the worker to the page.
EM_JS(void, brc_js_info, (int depth, int score, int nodes_k, int nps_k, int time_ms, const char* pv), {
    postMessage({type : 'info', depth : depth, score : score, nodesK : nodes_k, npsK : nps_k, timeMs : time_ms, pv : UTF8ToString(pv)});
});
#else
#define BRC_EXPORT extern "C"
static void brc_js_info(int, int, int, int, int, const char*) {}
#endif

using namespace barracuda;

namespace {

std::unique_ptr<Search> g_search;
std::shared_ptr<const nnue::Network> g_net;
std::string g_out;

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
        brc_js_info(si.depth, white_score, int(si.nodes / 1000), int(si.nps / 1000), int(si.time_ms), pv.c_str());
    });

    int score = stm == WHITE ? r.score : -r.score;
    if (is_mate_score(r.score)) score = score > 0 ? 30000 : -30000;
    std::string pv;
    for (Move m : r.pv) pv += " " + m.to_uci();
    g_out = (r.best ? r.best.to_uci() : std::string("none")) + " " + std::to_string(score) + " " + std::to_string(r.depth) +
            " " + std::to_string(r.nodes) + pv;
    return g_out.c_str();
}
