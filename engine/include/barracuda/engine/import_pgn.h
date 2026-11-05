// Barracuda - turn human games in PGN (for example the free Lichess database) into training data.
//
// The reader streams the file (or stdin), so a multi-gigabyte database can be cut off after the
// first N games. To keep the training set small and fast to train on, positions are filtered:
// rating window, no opening book moves, no checks, no captures, one position per N plies, each
// position only once. Positions get a score from the PGN's own engine evaluation when it has one,
// and from a short Barracuda search otherwise.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <string>

#include "barracuda/nnue/network.h"

namespace barracuda {

struct ImportPgnOptions {
    int max_games = 50000;     // stop after this many accepted games
    int min_elo = 0;           // both players must be inside [min_elo, max_elo] (0 = no limit)
    int max_elo = 0;
    int skip_plies = 10;       // opening moves are skipped
    int every = 3;             // keep one eligible position out of `every`
    int min_plies = 20;        // games shorter than this are dropped (aborted games, bots testing)
    int depth = 5;             // search depth for positions without an engine evaluation in the PGN
    int threads = 1;
    bool use_pgn_eval = true;  // take [%eval ...] from the PGN when present (free, and deeper than `depth`)
    bool require_result = true;
};

struct ImportPgnStats {
    int games_seen = 0, games_used = 0;
    int rejected_elo = 0, rejected_short = 0, rejected_other = 0, illegal = 0;
    std::uint64_t positions = 0, from_pgn_eval = 0, from_search = 0;
};

using ImportPgnLog = std::function<void(const std::string&)>;

// Writes training lines ("fen | score_cp_white | result") to `out_path`. Returns false on I/O errors.
bool import_pgn(std::istream& in, const std::string& out_path, const ImportPgnOptions& opt,
                std::shared_ptr<const nnue::Network> net, const std::atomic<bool>& stop, ImportPgnStats& stats,
                std::string* error, const ImportPgnLog& log = {});

} // namespace barracuda
