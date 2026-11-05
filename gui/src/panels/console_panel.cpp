#include "panels/console_panel.h"

#include <algorithm>
#include <cstring>

#include "app.h"
#include "theme.h"

namespace barracuda::gui {

namespace {
constexpr std::size_t kMaxLines = 5000;

ImU32 level_color(runtime::LogLevel l) {
    switch (l) {
    case runtime::LogLevel::Search: return palette::kAccent;
    case runtime::LogLevel::Train: return palette::kAccent2;
    case runtime::LogLevel::Bench: return palette::rgba(86, 196, 255);
    case runtime::LogLevel::Warn: return palette::kWarning;
    case runtime::LogLevel::Error: return palette::kNegative;
    case runtime::LogLevel::Success: return palette::kPositive;
    default: return palette::kText;
    }
}
const char* level_tag(runtime::LogLevel l) {
    switch (l) {
    case runtime::LogLevel::Search: return "search";
    case runtime::LogLevel::Train: return "train ";
    case runtime::LogLevel::Bench: return "bench ";
    case runtime::LogLevel::Warn: return "warn  ";
    case runtime::LogLevel::Error: return "error ";
    case runtime::LogLevel::Success: return "ok    ";
    default: return "info  ";
    }
}
} // namespace

void ConsolePanel::drain() {
    runtime::LogLine l;
    int budget = 2000; // bound per-frame work
    while (budget-- > 0 && app_.log.pop(l)) {
        lines_.push_back(l);
        if (lines_.size() > kMaxLines) lines_.pop_front();
        if (autoscroll_) scroll_to_bottom_ = true;
    }
}

void ConsolePanel::draw() {
    ImGui::PushFont(app_.fonts.title);
    ImGui::TextUnformatted("Console");
    ImGui::PopFont();
    ImGui::SameLine(0, 12);
    // toggles wrap onto a second row when the panel is narrow
    const ImGuiStyle& st = ImGui::GetStyle();
    const float toggles = 2.0f * (ImGui::GetFrameHeight() + st.ItemInnerSpacing.x) + ImGui::CalcTextSize("searchautoscroll").x +
                          ImGui::CalcTextSize("clear").x + st.FramePadding.x * 2 + st.ItemSpacing.x * 3;
    const float avail = ImGui::GetContentRegionAvail().x; // measured on the title row
    const float filter_w = std::clamp(avail - toggles - st.ItemSpacing.x, 80.0f, 200.0f);
    ImGui::SetNextItemWidth(filter_w);
    ImGui::InputTextWithHint("##filter", "filter", filter_, sizeof(filter_));
    if (avail - filter_w - st.ItemSpacing.x >= toggles) ImGui::SameLine();
    ImGui::Checkbox("search", &show_search_);
    ImGui::SameLine();
    ImGui::Checkbox("autoscroll", &autoscroll_);
    ImGui::SameLine();
    if (ImGui::SmallButton("clear")) lines_.clear();

    const float input_h = ImGui::GetFrameHeightWithSpacing() + 4.0f;
    // integrated-terminal look: pure black, mono font
    ImGui::PushStyleColor(ImGuiCol_ChildBg, palette::to_vec4(palette::kBackground));
    ImGui::BeginChild("##log", ImVec2(0, -input_h), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::PushFont(app_.fonts.mono);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 1));
    for (const auto& l : lines_) {
        if (!show_search_ && l.level == runtime::LogLevel::Search) continue;
        if (filter_[0] && !std::strstr(l.text, filter_)) continue;
        ImGui::TextColored(palette::to_vec4(palette::kTextDim), "%8.2f %s", l.time_s, level_tag(l.level));
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, palette::to_vec4(level_color(l.level)));
        ImGui::TextUnformatted(l.text);
        ImGui::PopStyleColor();
    }
    if (app_.log.dropped()) ImGui::TextColored(palette::to_vec4(palette::kWarning), "(%llu lines dropped under load)",
                                               (unsigned long long)app_.log.dropped());
    ImGui::PopStyleVar();
    ImGui::PopFont();
    if (scroll_to_bottom_) {
        ImGui::SetScrollHereY(1.0f);
        scroll_to_bottom_ = false;
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::SetNextItemWidth(-1);
    ImGui::PushFont(app_.fonts.mono);
    if (ImGui::InputTextWithHint("##cmd", "command (type 'help')", input_, sizeof(input_), ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (input_[0]) {
            app_.execute_command(input_);
            history_.push_back(input_);
            if (history_.size() > 50) history_.pop_front();
        }
        input_[0] = '\0';
        ImGui::SetKeyboardFocusHere(-1);
    }
    ImGui::PopFont();
}

} // namespace barracuda::gui
