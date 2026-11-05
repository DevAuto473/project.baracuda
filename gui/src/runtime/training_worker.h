// Barracuda runtime - training worker thread: self-play generation, dataset parsing,
// NNUE backpropagation, tactical benchmarks. Owns the float master network.
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <variant>

#include "barracuda/engine/search.h"
#include "barracuda/nnue/float_network.h"
#include "barracuda/nnue/inspector.h"
#include "runtime/shared_state.h"

namespace barracuda::runtime {

struct SelfPlayParams {
    int games = 50;
    int depth = 6;
    int random_plies = 8;
    std::string output = "selfplay.txt";
};

struct TrainParams {
    std::string dataset = "selfplay.txt";
    std::string output = "barracuda.bnn";
    int epochs = 10;
    int batch = 8192;
    float lr = 1e-3f;
    float lambda = 0.75f;
    int threads = 0;
    std::uint64_t max_samples = 0;
    bool reset_network = false;
    bool bench_each_epoch = true;
    int bench_movetime = 150;
};

struct MatchParams {
    int games = 40;
    int movetime_ms = 100;
};

struct BenchParams {
    std::string epd_file; // empty = built-in WAC subset
    int movetime_ms = 300;
};

class TrainingWorker {
public:
    TrainingWorker(Logger& log, NetworkRegistry& nets, std::atomic<int>& eval_mode);
    ~TrainingWorker();

    void start_selfplay(const SelfPlayParams& p);
    void start_training(const TrainParams& p);
    void start_benchmark(const BenchParams& p);
    void start_match(const MatchParams& p);
    void save_network(const std::string& path);
    void load_network(const std::string& path);
    void reset_network();
    void stop();       // abort the running task (queued tasks are dropped)
    bool busy() const { return busy_.load(std::memory_order_relaxed); }

    std::atomic<int> observe_delay_ms{0}; // slows self-play so humans can watch

    TripleBuffer<TrainingSnapshot> stats;
    TripleBuffer<ObservationSnapshot> observation;
    TripleBuffer<nnue::NetworkView> network_view;

private:
    struct Save { std::string path; };
    struct Load { std::string path; };
    struct Reset {};
    struct Quit {};
    using Command = std::variant<SelfPlayParams, TrainParams, BenchParams, MatchParams, Save, Load, Reset, Quit>;

    void run();
    void push(Command c);
    void publish_stats(bool force = false);
    void publish_network(); // quantise master -> registry
    void publish_view(const nnue::Network& net, const Position& pos);

    void do_selfplay(const SelfPlayParams& p);
    void do_train(const TrainParams& p);
    float do_benchmark(const BenchParams& p, bool quiet, int force_mode = -1);
    void do_match(const MatchParams& p);

    Logger& log_;
    NetworkRegistry& nets_;
    std::atomic<int>& eval_mode_;

    nnue::FloatNetwork master_;
    Search search_{32};
    TrainingSnapshot state_;
    nnue::NetworkView view_scratch_;
    std::chrono::steady_clock::time_point last_stats_{};
    std::chrono::steady_clock::time_point last_view_{};

    std::atomic<bool> stop_{false};
    std::atomic<bool> busy_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Command> queue_;
    std::thread thread_;
};

} // namespace barracuda::runtime
