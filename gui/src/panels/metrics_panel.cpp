#include "panels/metrics_panel.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "app.h"
#include "barracuda/engine/search.h"
#include "panels/board_panel.h"
#include "theme.h"
#include "widgets.h"

namespace barracuda::gui {

namespace {
void copy_str(char* dst, std::size_t n, const std::string& s) {
    std::snprintf(dst, n, "%s", s.c_str());
}
const char* task_name(runtime::TrainingTask t) {
    switch (t) {
    case runtime::TrainingTask::SelfPlay: return "Self-play";
    case runtime::TrainingTask::LoadingData: return "Parsing dataset";
    case runtime::TrainingTask::Training: return "Training";
    case runtime::TrainingTask::Benchmark: return "Benchmark";
    default: return "Idle";
    }
}
void section(const char* title) {
    ImGui::Spacing();
    ImGui::SeparatorText(title);
}
} // namespace

MetricsPanel::MetricsPanel(App& app) : app_(app) {
    copy_str(sp_out_, sizeof(sp_out_), sp_.output);
    copy_str(tp_data_, sizeof(tp_data_), tp_.dataset);
    copy_str(tp_out_, sizeof(tp_out_), tp_.output);
    bp_file_[0] = '\0';
    copy_str(net_path_, sizeof(net_path_), "barracuda.bnn");
}

void MetricsPanel::draw() {
    // sample search speed once per new snapshot for the sparkline
    const auto& si = app_.search_info();
    if (si.nodes != last_nodes_ && si.depth > 0) {
        last_nodes_ = si.nodes;
        nps_history_.push_back(float(si.nps) / 1e6f);
        depth_history_.push_back(float(si.depth));
        if (nps_history_.size() > 240) nps_history_.erase(nps_history_.begin());
        if (depth_history_.size() > 240) depth_history_.erase(depth_history_.begin());
    }

    if (ImGui::BeginTabBar("##metrics_tabs")) {
        if (ImGui::BeginTabItem("Metrics")) {
            draw_overview();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Training")) {
            draw_training();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Benchmark")) {
            draw_benchmark();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void MetricsPanel::draw_overview() {
    const auto& t = app_.training_stats();
    const auto& si = app_.search_info();
    ImGui::BeginChild("##overview", ImVec2(0, 0), ImGuiChildFlags_None);
    const float w = ImGui::GetContentRegionAvail().x;
    const float sp = ImGui::GetStyle().ItemSpacing.x;

    // ---- Elo ----
    ImGui::BeginGroup();
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "Estimated Elo (tactical suite)");
    ImGui::PushFont(app_.fonts.large);
    if (t.elo_history.empty()) ImGui::TextColored(palette::to_vec4(palette::kTextDim), "----");
    else ImGui::TextColored(palette::to_vec4(palette::kAccent), "%.0f", double(t.last_elo));
    ImGui::PopFont();
    if (t.task == runtime::TrainingTask::Benchmark)
        ImGui::TextColored(palette::to_vec4(palette::kWarning), "running suite: %d / %d (%d solved)", t.bench_done,
                           t.bench_total, t.bench_solved);
    else if (t.bench_total > 0)
        ImGui::TextColored(palette::to_vec4(palette::kTextDim), "%d / %d solved @ %d ms", t.bench_solved, t.bench_total,
                           t.bench_movetime);
    else
        ImGui::TextColored(palette::to_vec4(palette::kTextDim), "run a benchmark to start tracking");
    ImGui::EndGroup();
    ImGui::SameLine(0, 16);
    line_chart("elo", {{&t.elo_history, palette::kAccent, "Elo"}}, ImVec2(w - ImGui::GetItemRectSize().x - 16, 92.0f),
               "%.0f", false, "no benchmark yet");

    // ---- search ----
    section("Search");
    const float tw = (w - 3 * sp) / 4.0f;
    stat_tile("Depth / Sel", std::to_string(si.depth) + " / " + std::to_string(si.seldepth), tw);
    ImGui::SameLine();
    stat_tile("NPS", format_count(si.nps), tw, palette::kAccent);
    ImGui::SameLine();
    stat_tile("Nodes", format_count(si.nodes), tw);
    ImGui::SameLine();
    stat_tile("Score (White)", score_to_string(si.score_white), tw, si.score_white >= 0 ? palette::kPositive : palette::kNegative);
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "PV");
    ImGui::SameLine();
    ImGui::PushFont(app_.fonts.mono);
    ImGui::TextWrapped("%s", si.pv_san.empty() ? "-" : si.pv_san.c_str());
    ImGui::PopFont();
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "evaluator: %s   hash %d%%   %s", si.nnue ? "NNUE" : "classical",
                       si.hashfull / 10, si.thinking ? "searching" : "idle");
    line_chart("nps", {{&nps_history_, palette::kAccent2, "Mnps"}}, ImVec2(w, 70.0f), "%.2f", true,
               "search speed (Mnps) appears once Barracuda thinks");

    // ---- training ----
    section("Training");
    stat_tile("Task", task_name(t.task), tw, t.task == runtime::TrainingTask::Idle ? palette::kTextDim : palette::kWarning);
    ImGui::SameLine();
    stat_tile("Epoch", std::to_string(t.epoch) + " / " + std::to_string(t.epochs), tw);
    ImGui::SameLine();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.5f", t.train_loss);
    stat_tile("Train loss", buf, tw, palette::kAccent);
    ImGui::SameLine();
    std::snprintf(buf, sizeof(buf), "%.5f", t.val_loss);
    stat_tile("Val loss", buf, tw, palette::kAccent2);
    if (t.task != runtime::TrainingTask::Idle) {
        ImGui::ProgressBar(t.progress, ImVec2(w, 0), t.status.c_str());
    }
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "%s pos/s   dataset %s   net: %u epochs, %s samples seen",
                       format_count(std::uint64_t(t.samples_per_sec)).c_str(), format_count(t.dataset_size).c_str(),
                       t.net_epochs, format_count(t.net_samples).c_str());
    line_chart("loss", {{&t.loss_curve, palette::kAccent, "batch loss (EMA)"}}, ImVec2(w, 110.0f), "%.4f", true,
               "loss curve appears while training");
    line_chart("epochs", {{&t.epoch_train, palette::kAccent, "train"}, {&t.epoch_val, palette::kAccent2, "val"}},
               ImVec2(w, 90.0f), "%.4f", true, "per-epoch losses appear after the first epoch");
    ImGui::EndChild();
}

void MetricsPanel::draw_training() {
    auto* tw = app_.training.get();
    const auto& t = app_.training_stats();
    const bool busy = tw->busy();
    ImGui::BeginChild("##training", ImVec2(0, 0), ImGuiChildFlags_None);
    const float w = ImGui::GetContentRegionAvail().x;

    if (busy) {
        ImGui::TextColored(palette::to_vec4(palette::kWarning), "%s: %s", task_name(t.task), t.status.c_str());
        ImGui::ProgressBar(t.progress, ImVec2(w - 90, 0));
        ImGui::SameLine();
        if (ImGui::Button("Stop", ImVec2(80, 0))) tw->stop();
    } else {
        ImGui::TextColored(palette::to_vec4(palette::kPositive), "Worker idle - all tasks run off the render thread.");
    }

    // evaluator
    section("Evaluator");
    int mode = app_.eval_mode.load();
    ImGui::SetNextItemWidth(220);
    if (ImGui::Combo("Search evaluation", &mode, "Auto (NNUE after 1M samples)\0Classical\0NNUE (always)\0"))
        app_.eval_mode.store(mode);
    ImGui::TextColored(palette::to_vec4(palette::kTextDim),
                       "Auto keeps the hand-crafted eval until the net has seen %s training samples (now %s).",
                       format_count(kAutoNnueMinSamples).c_str(), format_count(t.net_samples).c_str());

    // self-play
    section("1. Generate data (self-play)");
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("games", &sp_.games, 10, 100);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("depth", &sp_.depth, 2, 12);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("random plies", &sp_.random_plies, 0, 16);
    ImGui::SetNextItemWidth(w * 0.5f);
    ImGui::InputText("output file##sp", sp_out_, sizeof(sp_out_));
    ImGui::SameLine();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Start self-play")) {
        sp_.games = std::max(1, sp_.games);
        sp_.output = sp_out_;
        tw->start_selfplay(sp_);
        app_.mode = ViewMode::Observe;
    }
    ImGui::EndDisabled();
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "Last run: %d/%d games, %s positions  (+%d =%d -%d)", t.sp_games_done,
                       t.sp_games_total, format_count(t.sp_positions).c_str(), t.sp_white, t.sp_draw, t.sp_black);

    // training
    section("2. Train the NNUE");
    ImGui::SetNextItemWidth(w * 0.5f);
    ImGui::InputText("dataset", tp_data_, sizeof(tp_data_));
    ImGui::SetNextItemWidth(w * 0.5f);
    ImGui::InputText("network out", tp_out_, sizeof(tp_out_));
    ImGui::SetNextItemWidth(110);
    ImGui::InputInt("epochs", &tp_.epochs);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::InputInt("batch", &tp_.batch, 1024, 4096);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::InputFloat("lr", &tp_.lr, 0.0f, 0.0f, "%.1e");
    ImGui::SetNextItemWidth(180);
    ImGui::SliderFloat("lambda (eval vs result)", &tp_.lambda, 0.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderInt("threads", &tp_.threads, 0, 32, tp_.threads == 0 ? "auto" : "%d");
    ImGui::Checkbox("start from scratch", &tp_.reset_network);
    ImGui::SameLine();
    ImGui::Checkbox("benchmark each epoch", &tp_.bench_each_epoch);
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Start training")) {
        tp_.epochs = std::clamp(tp_.epochs, 1, 1000);
        tp_.batch = std::clamp(tp_.batch, 64, 1 << 20);
        tp_.lr = std::clamp(tp_.lr, 1e-6f, 1.0f);
        tp_.dataset = tp_data_;
        tp_.output = tp_out_;
        tw->start_training(tp_);
    }
    ImGui::SameLine();
    if (ImGui::Button("Self-play + train")) {
        sp_.output = sp_out_;
        tp_.dataset = sp_out_;
        copy_str(tp_data_, sizeof(tp_data_), sp_.output);
        tp_.output = tp_out_;
        tw->start_selfplay(sp_);
        tw->start_training(tp_);
        app_.mode = ViewMode::Observe;
    }
    ImGui::EndDisabled();

    // network file
    section("Network file");
    ImGui::SetNextItemWidth(w * 0.5f);
    ImGui::InputText("##netpath", net_path_, sizeof(net_path_));
    ImGui::SameLine();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Save")) tw->save_network(net_path_);
    ImGui::SameLine();
    if (ImGui::Button("Load")) tw->load_network(net_path_);
    ImGui::SameLine();
    if (ImGui::Button("Reset weights")) tw->reset_network();
    ImGui::EndDisabled();
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "active: %s (%u epochs)", t.net_path.empty() ? "<in memory>" : t.net_path.c_str(),
                       t.net_epochs);
    ImGui::EndChild();
}

void MetricsPanel::draw_benchmark() {
    auto* tw = app_.training.get();
    const auto& t = app_.training_stats();
    const bool busy = tw->busy();
    ImGui::BeginChild("##bench", ImVec2(0, 0), ImGuiChildFlags_None);
    const float w = ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(w * 0.45f);
    ImGui::InputTextWithHint("##epd", "EPD file (empty = built-in WAC subset)", bp_file_, sizeof(bp_file_));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    ImGui::SliderInt("ms/pos", &bp_.movetime_ms, 50, 5000);
    ImGui::SameLine();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Run suite")) {
        bp_.epd_file = bp_file_;
        tw->start_benchmark(bp_);
    }
    ImGui::EndDisabled();
    if (busy && t.task == runtime::TrainingTask::Benchmark) {
        ImGui::SameLine();
        if (ImGui::Button("Stop")) tw->stop();
    }

    ImGui::Text("Solved %d / %d", t.bench_solved, std::max(t.bench_done, 0));
    ImGui::SameLine(0, 20);
    if (!t.elo_history.empty()) ImGui::TextColored(palette::to_vec4(palette::kAccent), "Estimated Elo %.0f", double(t.last_elo));
    ImGui::SameLine(0, 20);
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "(logit fit of the solve rate - a trend indicator, not a rating list)");

    if (ImGui::BeginTable("##suite", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("ID");
        ImGui::TableSetupColumn("Expected");
        ImGui::TableSetupColumn("Found");
        ImGui::TableSetupColumn("Depth");
        ImGui::TableSetupColumn("Result");
        ImGui::TableHeadersRow();
        for (const auto& r : t.bench_rows) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.id.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.expected.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.found.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", r.depth);
            ImGui::TableNextColumn();
            ImGui::TextColored(palette::to_vec4(r.solved ? palette::kPositive : palette::kNegative), r.solved ? "solved" : "missed");
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
}

} // namespace barracuda::gui
