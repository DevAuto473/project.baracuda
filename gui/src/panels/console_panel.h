// Barracuda GUI - live console (epochs, loss, depth, NPS, PV) with a small command line.
#pragma once

#include <imgui.h>

#include <deque>

#include "runtime/shared_state.h"

namespace barracuda::gui {

class App;

class ConsolePanel {
public:
    explicit ConsolePanel(App& app) : app_(app) { input_[0] = '\0'; filter_[0] = '\0'; }
    void drain();  // pull new lines from the lock-free log queue (every frame)
    void draw();

private:
    App& app_;
    std::deque<runtime::LogLine> lines_;
    char input_[256];
    char filter_[64];
    bool autoscroll_ = true;
    bool scroll_to_bottom_ = false;
    bool show_search_ = true;
    std::deque<std::string> history_;
};

} // namespace barracuda::gui
