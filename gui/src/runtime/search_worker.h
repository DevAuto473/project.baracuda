// Barracuda runtime - search worker thread (Human vs. Barracuda, analysis, NNUE inspection).
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <variant>

#include "barracuda/engine/search.h"
#include "barracuda/nnue/inspector.h"
#include "runtime/shared_state.h"

namespace barracuda::runtime {

class SearchWorker {
public:
    SearchWorker(Logger& log, NetworkRegistry& nets, std::atomic<int>& eval_mode, std::atomic<bool>& strange_style);
    ~SearchWorker();

    // All requests are non-blocking (called from the render thread).
    void request_move(const Position& pos, const SearchLimits& limits, std::uint64_t request_id);
    void request_inspect(const Position& pos);
    void request_new_game();
    void cancel(); // drops queued searches and aborts the running one

    // Outputs (render thread is the only consumer).
    TripleBuffer<SearchSnapshot> info;
    TripleBuffer<MoveResult> results;
    TripleBuffer<nnue::NetworkView> network_view;

private:
    struct Go {
        Position pos;
        SearchLimits limits;
        std::uint64_t id;
    };
    struct Inspect {
        Position pos;
    };
    struct NewGame {};
    struct Quit {};
    using Command = std::variant<Go, Inspect, NewGame, Quit>;

    void run();
    void push(Command c);
    void do_go(const Go& g);
    void do_inspect(const Position& pos);

    Logger& log_;
    NetworkRegistry& nets_;
    std::atomic<int>& eval_mode_;
    std::atomic<bool>& strange_style_;
    Search search_{64};

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Command> queue_;
    std::thread thread_;
    nnue::NetworkView view_scratch_;
};

} // namespace barracuda::runtime
