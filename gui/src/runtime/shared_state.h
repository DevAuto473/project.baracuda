// Barracuda runtime - data exchanged between worker threads and the 60 FPS render thread.
//
// Threading contract:
//   * Workers write snapshots into TripleBuffers they own (single producer each).
//   * The render thread is the single consumer of every TripleBuffer and never blocks on it.
//   * Log lines from any thread go through a lock-free MPMC ring, drained once per frame.
//   * The quantised network is shared through NetworkRegistry (shared_ptr swap under a tiny
//     mutex that only the workers touch - never the render path).
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "barracuda/engine/types.h"
#include "barracuda/nnue/network.h"
#include "runtime/concurrency.h"

namespace barracuda::runtime {

// ----------------------------------------------------------------------------- logging
enum class LogLevel : std::uint8_t { Info, Search, Train, Bench, Warn, Error, Success };

struct LogLine {
    LogLevel level = LogLevel::Info;
    double time_s = 0.0;
    char text[256] = {};
};

class Logger {
public:
    Logger() : start_(std::chrono::steady_clock::now()) {}

    void log(LogLevel lvl, std::string_view msg) {
        LogLine l;
        l.level = lvl;
        l.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        const std::size_t n = std::min(msg.size(), sizeof(l.text) - 1);
        msg.copy(l.text, n);
        l.text[n] = '\0';
        if (!queue_.try_push(l)) dropped_.fetch_add(1, std::memory_order_relaxed);
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    void logf(LogLevel lvl, const char* fmt, ...) {
        char buf[256];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        log(lvl, buf);
    }

    bool pop(LogLine& out) { return queue_.try_pop(out); }
    std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    MpmcQueue<LogLine> queue_{8192};
    std::atomic<std::uint64_t> dropped_{0};
    std::chrono::steady_clock::time_point start_;
};

// ----------------------------------------------------------------------------- network
class NetworkRegistry {
public:
    void publish(std::shared_ptr<const nnue::Network> net) {
        std::lock_guard<std::mutex> lk(mutex_);
        net_ = std::move(net);
        version_.fetch_add(1, std::memory_order_release);
    }
    std::shared_ptr<const nnue::Network> get() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return net_;
    }
    std::uint64_t version() const { return version_.load(std::memory_order_acquire); }

private:
    mutable std::mutex mutex_;
    std::shared_ptr<const nnue::Network> net_;
    std::atomic<std::uint64_t> version_{0};
};

// ----------------------------------------------------------------------------- snapshots
struct BoardSnapshot {
    BoardSnapshot() { board.fill(NO_PIECE); } // note: a zeroed Piece would be W_PAWN
    std::array<Piece, 64> board;
    Color stm = WHITE;
    Move last_move;
    bool in_check = false;
    std::string fen;
    void set(const Position& pos) {
        for (int s = 0; s < 64; ++s) board[std::size_t(s)] = pos.piece_on(Square(s));
        stm = pos.side_to_move();
        last_move = pos.history_size() > 1 ? pos.last_move() : Move();
        in_check = pos.in_check();
        fen = pos.fen();
    }
};

// Published by the search worker (play / analysis mode).
struct SearchSnapshot {
    bool thinking = false;
    std::uint64_t request_id = 0;
    int depth = 0, seldepth = 0;
    Value score_white = 0;
    std::uint64_t nodes = 0, nps = 0;
    std::int64_t time_ms = 0;
    int hashfull = 0;
    bool nnue = false;
    std::string pv_san;
    std::string pv_uci;
};

struct MoveResult {
    std::uint64_t request_id = 0;
    Move best;
    Value score_white = 0;
    int depth = 0;
};

// Published by the training worker (observation mode).
struct ObservationSnapshot {
    bool active = false;
    BoardSnapshot board;
    Value score_white = 0;
    int game = 0;
    int ply = 0;
    std::uint64_t serial = 0;
};

struct SuiteRow {
    std::string id, expected, found;
    bool solved = false;
    int depth = 0;
    Value score = 0;
};

enum class TrainingTask : std::uint8_t { Idle, SelfPlay, LoadingData, Training, Benchmark, Match, Publish };

struct TrainingSnapshot {
    TrainingTask task = TrainingTask::Idle;
    float progress = 0.0f;
    std::string status = "idle";

    // self-play
    int sp_games_done = 0, sp_games_total = 0;
    int sp_white = 0, sp_black = 0, sp_draw = 0;
    std::uint64_t sp_positions = 0;
    double sp_moves_per_sec = 0.0;

    // training
    int epoch = 0, epochs = 0;
    std::uint64_t batch = 0, batches = 0;
    double batch_loss = 0.0, train_loss = 0.0, val_loss = 0.0, samples_per_sec = 0.0, lr = 0.0;
    std::uint64_t dataset_size = 0;
    std::vector<float> loss_curve;  // smoothed batch loss (bounded)
    std::vector<float> epoch_train; // per-epoch train loss
    std::vector<float> epoch_val;   // per-epoch validation loss

    // benchmark / Elo
    std::vector<float> elo_history;
    float last_elo = 0.0f;
    int bench_solved = 0, bench_total = 0, bench_done = 0;
    int bench_movetime = 0;
    std::vector<SuiteRow> bench_rows;

    // Elo match (network vs classical evaluator)
    bool match_valid = false;
    int match_done = 0, match_total = 0, match_w = 0, match_d = 0, match_l = 0;
    double match_elo = 0.0;    // advantage over the classical evaluator
    double match_err = 0.0;    // 95% half-width
    double match_anchor = 0.0; // tactical-suite estimate of the classical evaluator
    double match_abs = 0.0;    // anchor + match_elo
    std::vector<float> match_history;

    // network
    std::uint32_t net_epochs = 0;
    std::uint64_t net_samples = 0;
    std::string net_path;
};

} // namespace barracuda::runtime
