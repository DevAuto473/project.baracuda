#include "runtime/search_worker.h"

#include "barracuda/engine/notation.h"

namespace barracuda::runtime {

SearchWorker::SearchWorker(Logger& log, NetworkRegistry& nets, std::atomic<int>& eval_mode)
    : log_(log), nets_(nets), eval_mode_(eval_mode) {
    thread_ = std::thread([this] { run(); });
}

SearchWorker::~SearchWorker() {
    cancel();
    push(Quit{});
    if (thread_.joinable()) thread_.join();
}

void SearchWorker::push(Command c) {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(c));
    }
    cv_.notify_one();
}

void SearchWorker::request_move(const Position& pos, const SearchLimits& limits, std::uint64_t id) {
    push(Go{pos, limits, id});
}

void SearchWorker::request_inspect(const Position& pos) {
    {
        // only the most recent inspection matters
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto it = queue_.begin(); it != queue_.end();)
            it = std::holds_alternative<Inspect>(*it) ? queue_.erase(it) : it + 1;
        queue_.push_back(Inspect{pos});
    }
    cv_.notify_one();
}

void SearchWorker::request_new_game() { push(NewGame{}); }

void SearchWorker::cancel() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto it = queue_.begin(); it != queue_.end();)
            it = std::holds_alternative<Go>(*it) ? queue_.erase(it) : it + 1;
    }
    search_.stop();
}

void SearchWorker::run() {
    while (true) {
        Command cmd;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [this] { return !queue_.empty(); });
            cmd = std::move(queue_.front());
            queue_.pop_front();
        }
        if (std::holds_alternative<Quit>(cmd)) return;
        if (auto* g = std::get_if<Go>(&cmd)) do_go(*g);
        else if (auto* i = std::get_if<Inspect>(&cmd)) do_inspect(i->pos);
        else if (std::holds_alternative<NewGame>(cmd)) search_.new_game();
    }
}

void SearchWorker::do_inspect(const Position& pos) {
    const auto net = nets_.get();
    if (!net) return;
    nnue::build_network_view(*net, pos, view_scratch_);
    network_view.publish(view_scratch_);
}

void SearchWorker::do_go(const Go& g) {
    search_.set_network(nets_.get());
    search_.set_eval_mode(EvalMode(eval_mode_.load()));
    const Color stm = g.pos.side_to_move();

    SearchSnapshot snap;
    snap.thinking = true;
    snap.request_id = g.id;
    snap.nnue = search_.nnue_active();
    info.publish(snap);

    log_.logf(LogLevel::Search, "search started (%s eval)", snap.nnue ? "NNUE" : "classical");

    int last_seldepth = 0;
    const SearchResult r = search_.go(g.pos, g.limits, [&](const SearchInfo& si) {
        last_seldepth = si.seldepth;
        SearchSnapshot s;
        s.thinking = true;
        s.request_id = g.id;
        s.depth = si.depth;
        s.seldepth = si.seldepth;
        s.score_white = stm == WHITE ? si.score : -si.score;
        s.nodes = si.nodes;
        s.nps = si.nps;
        s.time_ms = si.time_ms;
        s.hashfull = si.hashfull;
        s.nnue = si.nnue;
        s.pv_san = pv_to_san(g.pos, si.pv);
        for (Move m : si.pv) s.pv_uci += m.to_uci() + " ";
        info.publish(s);
        if (si.depth >= 4)
            log_.logf(LogLevel::Search, "depth %2d  score %-7s  nodes %-9llu  nps %-8llu  pv %s", si.depth,
                      score_to_string(s.score_white).c_str(), (unsigned long long)si.nodes,
                      (unsigned long long)si.nps, s.pv_san.c_str());
    });

    SearchSnapshot done;
    done.thinking = false;
    done.request_id = g.id;
    done.depth = r.depth;
    done.seldepth = last_seldepth;
    done.score_white = stm == WHITE ? r.score : -r.score;
    done.nodes = r.nodes;
    done.time_ms = r.time_ms;
    done.nps = r.time_ms > 0 ? r.nodes * 1000 / std::uint64_t(r.time_ms) : 0;
    done.nnue = search_.nnue_active();
    done.pv_san = pv_to_san(g.pos, r.pv);
    info.publish(done);

    MoveResult mr;
    mr.request_id = g.id;
    mr.best = r.best;
    mr.score_white = done.score_white;
    mr.depth = r.depth;
    results.publish(mr);

    // refresh the visualiser with the position after the engine's move
    if (r.best) {
        Position after = g.pos;
        after.do_move(r.best);
        do_inspect(after);
    }
}

} // namespace barracuda::runtime
