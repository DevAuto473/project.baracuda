// Barracuda GUI - metrics (Elo tracker, loss curves, search stats) and training controls.
#pragma once

#include <imgui.h>

#include <vector>

#include "runtime/training_worker.h"

namespace barracuda::gui {

class App;

class MetricsPanel {
public:
    explicit MetricsPanel(App& app);
    void draw();

private:
    void draw_overview();
    void draw_training();
    void draw_benchmark();
    void draw_match();

    App& app_;
    runtime::SelfPlayParams sp_;
    runtime::TrainParams tp_;
    runtime::BenchParams bp_;
    runtime::MatchParams mp_;
    runtime::PublishParams pub_;
    char sp_out_[256];
    char tp_data_[256];
    char tp_out_[256];
    char bp_file_[256];
    char net_path_[256];
    std::vector<float> nps_history_;
    std::vector<float> depth_history_;
    std::uint64_t last_nodes_ = 0;
};

} // namespace barracuda::gui
