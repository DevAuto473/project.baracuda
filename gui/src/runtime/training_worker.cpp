#include "runtime/training_worker.h"

#include <algorithm>
#include <filesystem>

#include "barracuda/engine/match.h"
#include "barracuda/engine/notation.h"
#include "barracuda/engine/selfplay.h"
#include "barracuda/engine/suite.h"
#include "barracuda/nnue/dataset.h"
#include "barracuda/nnue/trainer.h"

namespace barracuda::runtime {

namespace {
constexpr std::size_t kLossCurveMax = 600;
constexpr const char* kViewPositions[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
};

void push_bounded(std::vector<float>& v, float x) {
    v.push_back(x);
    if (v.size() >= kLossCurveMax) { // decimate by 2 to keep the whole history visible
        std::vector<float> half;
        half.reserve(v.size() / 2 + 1);
        for (std::size_t i = 0; i + 1 < v.size(); i += 2) half.push_back(0.5f * (v[i] + v[i + 1]));
        v.swap(half);
    }
}
} // namespace

TrainingWorker::TrainingWorker(Logger& log, NetworkRegistry& nets, std::atomic<int>& eval_mode)
    : log_(log), nets_(nets), eval_mode_(eval_mode) {
    std::string err;
    std::error_code ec;
    if (std::filesystem::exists("barracuda.bnn", ec) && master_.load("barracuda.bnn", &err)) {
        state_.net_path = "barracuda.bnn";
        log_.logf(LogLevel::Success, "loaded network barracuda.bnn (%u epochs, %llu samples)", master_.epochs_trained,
                  (unsigned long long)master_.samples_trained);
    } else {
        master_.init_random(0xBA22AC0DA);
        log_.log(LogLevel::Info, "no barracuda.bnn found: starting from an untrained network (classical eval active)");
    }
    publish_network();
    {
        const auto net = nets_.get();
        Position start;
        publish_view(*net, start);
    }
    publish_stats(true);
    thread_ = std::thread([this] { run(); });
}

TrainingWorker::~TrainingWorker() {
    stop();
    push(Quit{});
    if (thread_.joinable()) thread_.join();
}

void TrainingWorker::push(Command c) {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(c));
    }
    cv_.notify_one();
}

void TrainingWorker::start_selfplay(const SelfPlayParams& p) { push(p); }
void TrainingWorker::start_training(const TrainParams& p) { push(p); }
void TrainingWorker::start_benchmark(const BenchParams& p) { push(p); }
void TrainingWorker::start_match(const MatchParams& p) { push(p); }
void TrainingWorker::save_network(const std::string& path) { push(Save{path}); }
void TrainingWorker::load_network(const std::string& path) { push(Load{path}); }
void TrainingWorker::reset_network() { push(Reset{}); }

void TrainingWorker::stop() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        for (auto it = queue_.begin(); it != queue_.end();)
            it = std::holds_alternative<Quit>(*it) ? it + 1 : queue_.erase(it);
    }
    stop_.store(true);
    search_.stop();
}

void TrainingWorker::run() {
    while (true) {
        Command cmd;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [this] { return !queue_.empty(); });
            cmd = std::move(queue_.front());
            queue_.pop_front();
        }
        if (std::holds_alternative<Quit>(cmd)) return;
        stop_.store(false);
        busy_.store(true);

        if (auto* sp = std::get_if<SelfPlayParams>(&cmd)) {
            do_selfplay(*sp);
        } else if (auto* tp = std::get_if<TrainParams>(&cmd)) {
            do_train(*tp);
        } else if (auto* bp = std::get_if<BenchParams>(&cmd)) {
            do_benchmark(*bp, false);
        } else if (auto* mp = std::get_if<MatchParams>(&cmd)) {
            do_match(*mp);
        } else if (auto* s = std::get_if<Save>(&cmd)) {
            std::string err;
            if (master_.save(s->path, &err)) {
                state_.net_path = s->path;
                log_.logf(LogLevel::Success, "network saved to %s", s->path.c_str());
            } else {
                log_.logf(LogLevel::Error, "save failed: %s", err.c_str());
            }
        } else if (auto* l = std::get_if<Load>(&cmd)) {
            std::string err;
            nnue::FloatNetwork tmp;
            if (tmp.load(l->path, &err)) {
                master_ = std::move(tmp);
                state_.net_path = l->path;
                publish_network();
                log_.logf(LogLevel::Success, "network loaded from %s (%u epochs)", l->path.c_str(), master_.epochs_trained);
            } else {
                log_.logf(LogLevel::Error, "load failed: %s", err.c_str());
            }
        } else if (std::holds_alternative<Reset>(cmd)) {
            master_.init_random(std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()));
            publish_network();
            log_.log(LogLevel::Warn, "network re-initialised with random weights");
        }

        state_.task = TrainingTask::Idle;
        state_.status = stop_.load() ? "stopped" : "idle";
        state_.progress = 0.0f;
        busy_.store(false);
        publish_stats(true);
        if (auto net = nets_.get()) publish_view(*net, Position());
    }
}

void TrainingWorker::publish_stats(bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - last_stats_ < std::chrono::milliseconds(50)) return;
    last_stats_ = now;
    state_.net_epochs = master_.epochs_trained;
    state_.net_samples = master_.samples_trained;
    stats.publish(state_);
}

void TrainingWorker::publish_network() { nets_.publish(nnue::Network::from_float(master_)); }

void TrainingWorker::publish_view(const nnue::Network& net, const Position& pos) {
    nnue::build_network_view(net, pos, view_scratch_);
    network_view.publish(view_scratch_);
    last_view_ = std::chrono::steady_clock::now();
}

// ----------------------------------------------------------------------------- self-play
void TrainingWorker::do_selfplay(const SelfPlayParams& p) {
    state_.task = TrainingTask::SelfPlay;
    state_.sp_games_done = 0;
    state_.sp_games_total = p.games;
    state_.sp_white = state_.sp_black = state_.sp_draw = 0;
    state_.sp_positions = 0;
    state_.status = "self-play";
    publish_stats(true);

    search_.set_network(nets_.get());
    search_.set_eval_mode(EvalMode(eval_mode_.load()));
    log_.logf(LogLevel::Train, "self-play: %d games at depth %d -> %s (%s eval)", p.games, p.depth, p.output.c_str(),
              search_.nnue_active() ? "NNUE" : "classical");

    SelfPlayConfig cfg;
    cfg.games = p.games;
    cfg.depth = p.depth;
    cfg.random_plies = p.random_plies;
    cfg.output_path = p.output;

    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t moves = 0;
    std::uint64_t serial = 0;
    ObservationSnapshot last; // keeps the final position on screen once self-play ends
    SelfPlayCallbacks cb;
    cb.on_move = [&](const Position& pos, Move, Value score_white, int game, int ply) {
        ++moves;
        ObservationSnapshot& o = observation.write_buffer();
        o.active = true;
        o.board.set(pos);
        o.score_white = score_white;
        o.game = game + 1;
        o.ply = ply;
        o.serial = ++serial;
        last = o;
        observation.publish();

        const auto now = std::chrono::steady_clock::now();
        if (now - last_view_ > std::chrono::milliseconds(100)) {
            if (auto net = nets_.get()) publish_view(*net, pos);
        }
        const double secs = std::chrono::duration<double>(now - t0).count();
        state_.sp_moves_per_sec = secs > 0 ? double(moves) / secs : 0.0;
        publish_stats();

        // optional slow-down for human observation
        for (int waited = 0; waited < observe_delay_ms.load() && !stop_.load(); waited += 10)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };
    cb.on_game_end = [&](const SelfPlayStats& st, const std::string& res) {
        state_.sp_games_done = st.games_done;
        state_.sp_white = st.white_wins;
        state_.sp_black = st.black_wins;
        state_.sp_draw = st.draws;
        state_.sp_positions = st.positions;
        state_.progress = float(st.games_done) / float(std::max(1, p.games));
        log_.logf(LogLevel::Train, "game %d/%d: %s", st.games_done, p.games, res.c_str());
        publish_stats(true);
    };

    SelfPlayStats st;
    std::string err;
    if (!run_selfplay(search_, cfg, stop_, cb, &st, &err)) {
        log_.logf(LogLevel::Error, "self-play failed: %s", err.c_str());
    } else {
        log_.logf(LogLevel::Success, "self-play finished: %d games, %llu positions (+%d =%d -%d)", st.games_done,
                  (unsigned long long)st.positions, st.white_wins, st.draws, st.black_wins);
    }
    last.active = false;
    last.serial = ++serial;
    observation.publish(last);
}

// ----------------------------------------------------------------------------- training
void TrainingWorker::do_train(const TrainParams& p) {
    state_.task = TrainingTask::LoadingData;
    state_.status = "loading " + p.dataset;
    state_.progress = 0.0f;
    state_.epoch = 0;
    state_.epochs = p.epochs;
    state_.loss_curve.clear();
    state_.epoch_train.clear();
    state_.epoch_val.clear();
    publish_stats(true);

    nnue::Dataset data;
    nnue::Dataset::LoadStats ls;
    std::string err;
    log_.logf(LogLevel::Train, "loading dataset %s ...", p.dataset.c_str());
    const bool ok = data.load_text(p.dataset, p.max_samples, &ls, &err, &stop_, [&](std::uint64_t lines, std::uint64_t acc) {
        state_.status = "parsing: " + std::to_string(lines) + " lines, " + std::to_string(acc) + " samples";
        state_.dataset_size = acc;
        publish_stats();
    });
    if (!ok || stop_.load()) {
        if (!ok) log_.logf(LogLevel::Error, "%s", err.c_str());
        return;
    }
    state_.dataset_size = data.size();
    log_.logf(LogLevel::Train, "dataset: %llu samples (%llu rejected lines)", (unsigned long long)data.size(),
              (unsigned long long)ls.rejected);
    if (data.size() < 1000) {
        log_.log(LogLevel::Error, "dataset too small (need >= 1000 positions) - run self-play first");
        return;
    }

    if (p.reset_network) {
        master_.init_random(std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()));
        log_.log(LogLevel::Warn, "training from scratch (random initialisation)");
    }

    nnue::TrainerConfig cfg;
    cfg.epochs = p.epochs;
    cfg.batch_size = p.batch;
    cfg.learning_rate = p.lr;
    cfg.lambda = p.lambda;
    cfg.threads = p.threads;
    nnue::Trainer trainer(master_, cfg);

    state_.task = TrainingTask::Training;
    state_.status = "training";
    float ema = -1.0f;
    std::size_t view_idx = 0;
    log_.logf(LogLevel::Train, "training: %d epochs, batch %d, lr %.1e, lambda %.2f", p.epochs, p.batch, double(p.lr),
              double(p.lambda));

    trainer.run(
        data, stop_,
        [&](const nnue::TrainingProgress& pr) {
            ema = ema < 0 ? float(pr.batch_loss) : 0.95f * ema + 0.05f * float(pr.batch_loss);
            push_bounded(state_.loss_curve, ema);
            state_.epoch = pr.epoch;
            state_.batch = pr.batch;
            state_.batches = pr.batches_per_epoch;
            state_.batch_loss = pr.batch_loss;
            state_.train_loss = pr.train_loss;
            state_.samples_per_sec = pr.samples_per_sec;
            state_.lr = pr.learning_rate;
            state_.progress = (float(pr.epoch - 1) + float(pr.batch) / float(std::max<std::uint64_t>(1, pr.batches_per_epoch))) /
                              float(std::max(1, pr.epochs));
            publish_stats();
            // live weights in the visualiser (~1 Hz)
            if (std::chrono::steady_clock::now() - last_view_ > std::chrono::seconds(1)) {
                const auto live = nnue::Network::from_float(master_);
                publish_view(*live, Position(kViewPositions[view_idx++ % std::size(kViewPositions)]));
            }
        },
        [&](const nnue::TrainingProgress& pr) {
            state_.val_loss = pr.val_loss;
            state_.epoch_train.push_back(float(pr.train_loss));
            state_.epoch_val.push_back(float(pr.val_loss));
            std::string e;
            if (!master_.save(p.output, &e)) log_.logf(LogLevel::Error, "checkpoint failed: %s", e.c_str());
            else state_.net_path = p.output;
            publish_network();
            log_.logf(LogLevel::Train, "epoch %d/%d  train %.6f  val %.6f  %.0f pos/s  -> %s", pr.epoch, pr.epochs,
                      pr.train_loss, pr.val_loss, pr.samples_per_sec, p.output.c_str());
            publish_stats(true);
            if (p.bench_each_epoch && !stop_.load()) {
                BenchParams bp;
                bp.movetime_ms = p.bench_movetime;
                do_benchmark(bp, true, int(EvalMode::NNUE));
                state_.task = TrainingTask::Training;
                state_.status = "training";
            }
        });

    if (stop_.load()) log_.log(LogLevel::Warn, "training stopped by user");
    else log_.logf(LogLevel::Success, "training complete: %s (%u total epochs)", p.output.c_str(), master_.epochs_trained);
}

// ----------------------------------------------------------------------------- Elo match
// The current network (forced NNUE) plays colour-swapped games against the classical evaluator.
// The classical evaluator is anchored with the tactical-suite estimate, so the headline number is
// anchor + match Elo. The match part is a real measurement; the anchor is the heuristic part.
void TrainingWorker::do_match(const MatchParams& p) {
    const auto net = nets_.get();
    state_.task = TrainingTask::Match;
    state_.status = "calibrating classical anchor";
    state_.progress = 0.0f;
    state_.match_valid = false;
    state_.match_done = 0;
    state_.match_total = p.games;
    state_.match_w = state_.match_d = state_.match_l = 0;
    state_.match_elo = state_.match_err = state_.match_abs = 0.0;
    state_.match_history.clear();
    publish_stats(true);

    if (!net || !net->is_trained())
        log_.log(LogLevel::Warn, "the network is untrained - expect a very low match result (train it first)");

    search_.set_network(nullptr);
    search_.set_eval_mode(EvalMode::CLASSICAL);
    const SuiteResult cal = run_suite(search_, builtin_wac_suite(), 150, stop_);
    if (stop_.load()) return;
    state_.match_anchor = cal.elo_estimate;
    log_.logf(LogLevel::Bench, "anchor: classical evaluator solves %d/%d -> ~%.0f Elo (tactical estimate)", cal.solved,
              cal.total, cal.elo_estimate);

    Search a(16), b(16);
    a.set_network(net);
    a.set_eval_mode(EvalMode::NNUE);
    b.set_network(nullptr);
    b.set_eval_mode(EvalMode::CLASSICAL);

    MatchConfig cfg;
    cfg.games = p.games;
    cfg.movetime_ms = p.movetime_ms;
    state_.status = "match: Barracuda NNUE vs classical";
    log_.logf(LogLevel::Bench, "match: %d games at %d ms/move (NNUE vs classical)", p.games, p.movetime_ms);
    publish_stats(true);

    std::uint64_t serial = 0;
    ObservationSnapshot last;
    MatchCallbacks cb;
    cb.on_move = [&](const Position& pos, Move, Value score_white, int game, int ply) {
        ObservationSnapshot& o = observation.write_buffer();
        o.active = true;
        o.board.set(pos);
        o.score_white = score_white;
        o.game = game + 1;
        o.ply = ply;
        o.serial = ++serial;
        last = o;
        observation.publish();
        for (int waited = 0; waited < observe_delay_ms.load() && !stop_.load(); waited += 10)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };
    cb.on_game_end = [&](const MatchStats& st, const std::string& desc) {
        state_.match_done = st.games_done;
        state_.match_w = st.wins;
        state_.match_d = st.draws;
        state_.match_l = st.losses;
        state_.match_elo = st.elo.elo;
        state_.match_err = st.elo.error;
        state_.match_abs = state_.match_anchor + st.elo.elo;
        state_.match_valid = true;
        state_.progress = float(st.games_done) / float(std::max(1, p.games));
        state_.match_history.push_back(float(state_.match_abs));
        log_.logf(LogLevel::Bench, "game %d/%d: %s   (+%d =%d -%d)  Elo %.0f +/- %.0f", st.games_done, p.games,
                  desc.c_str(), st.wins, st.draws, st.losses, state_.match_abs, st.elo.error);
        publish_stats(true);
    };
    MatchStats st;
    run_match(a, b, cfg, stop_, cb, &st);
    last.active = false;
    last.serial = ++serial;
    observation.publish(last);

    if (st.games_done > 0)
        log_.logf(LogLevel::Success, "match finished: +%d =%d -%d  ->  Barracuda ~%.0f Elo (+/- %.0f vs classical %.0f)",
                  st.wins, st.draws, st.losses, state_.match_abs, st.elo.error, state_.match_anchor);
    else
        log_.log(LogLevel::Warn, "match stopped before the first game finished");
}

// ----------------------------------------------------------------------------- benchmark
float TrainingWorker::do_benchmark(const BenchParams& p, bool quiet, int force_mode) {
    std::vector<SuitePosition> suite;
    if (!p.epd_file.empty()) {
        std::string err;
        int skipped = 0;
        if (!load_epd_file(p.epd_file, suite, &err, &skipped)) {
            log_.logf(LogLevel::Error, "benchmark: %s", err.c_str());
            return 0.0f;
        }
        if (skipped) log_.logf(LogLevel::Warn, "benchmark: skipped %d invalid EPD lines", skipped);
    } else {
        suite = builtin_wac_suite();
    }

    const TrainingTask prev_task = state_.task;
    state_.task = TrainingTask::Benchmark;
    state_.status = "benchmark";
    state_.bench_rows.clear();
    state_.bench_total = int(suite.size());
    state_.bench_done = 0;
    state_.bench_solved = 0;
    state_.bench_movetime = p.movetime_ms;
    publish_stats(true);

    search_.set_network(nets_.get());
    // per-epoch benchmarks measure the network being trained, whatever the play setting is
    search_.set_eval_mode(force_mode >= 0 ? EvalMode(force_mode) : EvalMode(eval_mode_.load()));
    if (!quiet)
        log_.logf(LogLevel::Bench, "benchmark: %d positions @ %d ms (%s eval)", int(suite.size()), p.movetime_ms,
                  search_.nnue_active() ? "NNUE" : "classical");

    const SuiteResult r = run_suite(search_, suite, p.movetime_ms, stop_, [&](const SuiteEntryResult& e, int i, int n) {
        state_.bench_rows.push_back({e.id, e.expected, e.found, e.solved, e.depth, e.score});
        state_.bench_done = i + 1;
        if (e.solved) state_.bench_solved++;
        if (prev_task != TrainingTask::Training) state_.progress = float(i + 1) / float(n);
        publish_stats();
        if (!quiet)
            log_.logf(LogLevel::Bench, "%-9s expected %-10s found %-8s %s", e.id.c_str(), e.expected.c_str(),
                      e.found.c_str(), e.solved ? "OK" : "MISS");
    });
    if (stop_.load() && int(r.entries.size()) < int(suite.size())) {
        log_.log(LogLevel::Warn, "benchmark aborted");
        return 0.0f;
    }
    state_.last_elo = float(r.elo_estimate);
    state_.elo_history.push_back(state_.last_elo);
    log_.logf(LogLevel::Bench, "suite solved %d/%d -> estimated Elo %.0f (%s)", r.solved, r.total, r.elo_estimate,
              search_.nnue_active() ? "NNUE" : "classical");
    publish_stats(true);
    return state_.last_elo;
}

} // namespace barracuda::runtime
