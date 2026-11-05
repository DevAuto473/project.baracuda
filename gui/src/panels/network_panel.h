// Barracuda GUI - NNUE architecture visualiser (node-link graph with live activations).
#pragma once

#include <imgui.h>

#include <array>
#include <vector>

#include "barracuda/nnue/inspector.h"

namespace barracuda::gui {

class App;

class NetworkPanel {
public:
    explicit NetworkPanel(App& app) : app_(app) {}
    void draw();

private:
    void sync(const nnue::NetworkView& v);

    App& app_;
    // smoothed activations (render-local) so changes animate instead of flickering
    std::array<std::vector<float>, nnue::NetworkView::kLayers> shown_;
    int edge_mode_ = 0;          // 0 = signal (|w| * activation), 1 = weight magnitude
    float edge_threshold_ = 0.12f;
    bool show_labels_ = true;
    float controls_w_ = 380.0f; // measured each frame for right alignment
};

} // namespace barracuda::gui
