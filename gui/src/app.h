// Barracuda GUI - application root: owns workers, shared state and the four dashboard panels.
#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <string>

#include "barracuda/nnue/inspector.h"
#include "runtime/search_worker.h"
#include "runtime/shared_state.h"
#include "runtime/training_worker.h"
#include "theme.h"

namespace barracuda::gui {

class BoardPanel;
class NetworkPanel;
class MetricsPanel;
class ConsolePanel;

enum class ViewMode { Play, Observe };

class App {
public:
    explicit App(const Fonts& fonts);
    ~App();

    void frame(); // called once per rendered frame on the render thread

    // ---- shared infrastructure ------------------------------------------------
    runtime::Logger log;
    runtime::NetworkRegistry nets;
    std::atomic<int> eval_mode{0}; // EvalMode
    std::unique_ptr<runtime::SearchWorker> search;
    std::unique_ptr<runtime::TrainingWorker> training;
    Fonts fonts;

    // ---- render-thread view of worker snapshots (refreshed at the start of each frame) --
    const runtime::SearchSnapshot& search_info() const { return search->info.read_buffer(); }
    const runtime::TrainingSnapshot& training_stats() const { return training->stats.read_buffer(); }
    const runtime::ObservationSnapshot& observation() const { return training->observation.read_buffer(); }
    const nnue::NetworkView& network_view() const;
    bool observation_changed() const { return observation_changed_; }

    ViewMode mode = ViewMode::Play;
    void execute_command(const std::string& line);

    BoardPanel& board() { return *board_; }

private:
    void poll_workers();
    void draw_top_bar(float height);

    std::unique_ptr<BoardPanel> board_;
    std::unique_ptr<NetworkPanel> network_;
    std::unique_ptr<MetricsPanel> metrics_;
    std::unique_ptr<ConsolePanel> console_;

    bool observation_changed_ = false;
    int view_source_ = 0; // 0 = search worker, 1 = training worker (newest wins)
    double fps_smoothed_ = 60.0;
};

} // namespace barracuda::gui
