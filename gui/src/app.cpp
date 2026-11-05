#include "app.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <thread>

#include "barracuda/engine/movegen.h"
#include "panels/board_panel.h"
#include "panels/console_panel.h"
#include "panels/metrics_panel.h"
#include "panels/network_panel.h"
#include "widgets.h"

namespace barracuda::gui {

App::App(const Fonts& f) : fonts(f) {
    Position::init();
    log.logf(runtime::LogLevel::Info, "Barracuda dashboard starting (NNUE SIMD backend: %s, %u hardware threads)",
             nnue::simd_backend(), std::thread::hardware_concurrency());
    training = std::make_unique<runtime::TrainingWorker>(log, nets, eval_mode); // publishes the initial network
    search = std::make_unique<runtime::SearchWorker>(log, nets, eval_mode, strange_style);
    board_ = std::make_unique<BoardPanel>(*this);
    network_ = std::make_unique<NetworkPanel>(*this);
    metrics_ = std::make_unique<MetricsPanel>(*this);
    console_ = std::make_unique<ConsolePanel>(*this);
    log.log(runtime::LogLevel::Info, "type 'help' in the console for commands; drag or click pieces to play");
}

App::~App() {
    // Stop workers before panels go away (workers never reference panels, but be tidy).
    if (search) search->cancel();
    if (training) training->stop();
    search.reset();
    training.reset();
}

const nnue::NetworkView& App::network_view() const {
    return view_source_ == 1 ? training->network_view.read_buffer() : search->network_view.read_buffer();
}

void App::poll_workers() {
    search->info.update();
    training->stats.update();
    observation_changed_ = training->observation.update();
    const bool s = search->network_view.update();
    const bool t = training->network_view.update();
    if (s && t) view_source_ = mode == ViewMode::Observe ? 1 : 0;
    else if (t) view_source_ = 1;
    else if (s) view_source_ = 0;
    if (!network_view().valid) view_source_ = training->network_view.read_buffer().valid ? 1 : 0;
}

void App::execute_command(const std::string& line) {
    std::istringstream ss(line);
    std::string cmd;
    ss >> cmd;
    std::string rest;
    std::getline(ss, rest);
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
    log.logf(runtime::LogLevel::Info, "> %s", line.c_str());

    if (cmd == "help") {
        log.log(runtime::LogLevel::Info, "commands: help | clear | newgame | undo | flip | fen <FEN> | moves |"
                                         " play | observe | stop | bench [ms] | selfplay <games> [depth] |"
                                         " match [games] [ms] | style on|off | publish [games] | train [dataset] | save [file] | load [file] | eval <auto|classical|nnue> | watch <ms>");
    } else if (cmd == "newgame") {
        mode = ViewMode::Play;
        board_->new_game();
    } else if (cmd == "undo") {
        board_->undo();
    } else if (cmd == "flip") {
        board_->flip();
    } else if (cmd == "fen") {
        mode = ViewMode::Play;
        board_->set_fen(rest);
    } else if (cmd == "moves") {
        MoveList l;
        generate_moves(board_->game(), l);
        std::string s;
        for (Move m : l) s += m.to_uci() + " ";
        log.logf(runtime::LogLevel::Info, "%d legal: %s", l.size, s.c_str());
    } else if (cmd == "play") {
        mode = ViewMode::Play;
    } else if (cmd == "observe") {
        board_->stop_engine();
        mode = ViewMode::Observe;
    } else if (cmd == "stop") {
        board_->stop_engine();
        training->stop();
    } else if (cmd == "bench") {
        runtime::BenchParams p;
        if (!rest.empty()) p.movetime_ms = std::max(10, std::atoi(rest.c_str()));
        training->start_benchmark(p);
    } else if (cmd == "match") {
        runtime::MatchParams p;
        std::istringstream a(rest);
        a >> p.games >> p.movetime_ms;
        p.games = std::clamp(p.games, 2, 1000);
        p.movetime_ms = std::clamp(p.movetime_ms, 10, 5000);
        training->start_match(p);
        mode = ViewMode::Observe;
    } else if (cmd == "style") {
        strange_style.store(rest != "off");
        log.logf(runtime::LogLevel::Info, "strange style: %s", strange_style.load() ? "on" : "off");
    } else if (cmd == "publish") {
        runtime::PublishParams p;
        if (!rest.empty()) p.games = std::clamp(std::atoi(rest.c_str()), 0, 1000);
        training->start_publish(p);
    } else if (cmd == "selfplay") {
        runtime::SelfPlayParams p;
        std::istringstream a(rest);
        a >> p.games >> p.depth;
        p.games = std::max(1, p.games);
        p.depth = std::clamp(p.depth, 1, 20);
        training->start_selfplay(p);
        mode = ViewMode::Observe;
    } else if (cmd == "train") {
        runtime::TrainParams p;
        if (!rest.empty()) p.dataset = rest;
        training->start_training(p);
    } else if (cmd == "save") {
        training->save_network(rest.empty() ? "barracuda.bnn" : rest);
    } else if (cmd == "load") {
        training->load_network(rest.empty() ? "barracuda.bnn" : rest);
    } else if (cmd == "eval") {
        if (rest == "classical") eval_mode = int(EvalMode::CLASSICAL);
        else if (rest == "nnue") eval_mode = int(EvalMode::NNUE);
        else eval_mode = int(EvalMode::AUTO);
        log.logf(runtime::LogLevel::Info, "evaluation mode: %s", rest.empty() ? "auto" : rest.c_str());
    } else if (cmd == "watch") {
        training->observe_delay_ms.store(std::clamp(std::atoi(rest.c_str()), 0, 5000));
        log.logf(runtime::LogLevel::Info, "observation delay: %d ms/move", training->observe_delay_ms.load());
    } else if (cmd == "clear") {
        // handled by the console's own "clear" button; emulate by logging a separator
        log.log(runtime::LogLevel::Info, "----------------------------------------");
    } else {
        log.logf(runtime::LogLevel::Warn, "unknown command '%s' (try 'help')", cmd.c_str());
    }
}

void App::draw_top_bar(float height) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    dl->AddText(fonts.title, fonts.title->FontSize, ImVec2(p.x + 2, p.y + (height - fonts.title->FontSize) * 0.5f),
                palette::kAccent, "BARRACUDA");
    const float tx = p.x + fonts.title->CalcTextSizeA(fonts.title->FontSize, 1e9f, 0, "BARRACUDA").x + 14;
    dl->AddText(ImVec2(tx, p.y + (height - ImGui::GetFontSize()) * 0.5f), palette::kTextDim,
                "C++20 chess engine  |  NNUE training dashboard");

    const float dt = ImGui::GetIO().DeltaTime;
    if (dt > 0) fps_smoothed_ = fps_smoothed_ * 0.95 + (1.0 / dt) * 0.05;
    const auto net = nets.get();
    const auto& si = search_info();
    const auto& ts = training_stats();
    char elo[48];
    if (ts.match_valid) std::snprintf(elo, sizeof(elo), "Elo %.0f +/- %.0f", ts.match_abs, ts.match_err);
    else if (!ts.elo_history.empty()) std::snprintf(elo, sizeof(elo), "Elo ~%.0f", double(ts.last_elo));
    else std::snprintf(elo, sizeof(elo), "Elo --");
    char right[256];
    std::snprintf(right, sizeof(right), "search: %s   trainer: %s   net: %s   %.0f FPS",
                  si.thinking ? "thinking" : "idle",
                  ts.task == runtime::TrainingTask::Idle ? "idle" : ts.status.c_str(),
                  net && net->is_trained() ? "trained" : "untrained", fps_smoothed_);
    const ImVec2 rs = ImGui::CalcTextSize(right);
    dl->AddText(ImVec2(p.x + w - rs.x - 4, p.y + (height - ImGui::GetFontSize()) * 0.5f), palette::kTextDim, right);
    const ImVec2 es = ImGui::CalcTextSize(elo);
    dl->AddText(fonts.title, fonts.title->FontSize,
                ImVec2(p.x + w - rs.x - es.x * 1.25f - 28, p.y + (height - fonts.title->FontSize) * 0.5f),
                palette::kAccent, elo);
    dl->AddLine(ImVec2(p.x, p.y + height - 1), ImVec2(p.x + w, p.y + height - 1), palette::kBorder);
    ImGui::Dummy(ImVec2(w, height));
}

void App::frame() {
    poll_workers();
    console_->drain();
    board_->update();

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 6));
    ImGui::Begin("##dashboard", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImGui::PopStyleVar();

    draw_top_bar(ImGui::GetFontSize() * 2.0f);

    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float gap = ImGui::GetStyle().ItemSpacing.x;
    const float left_w = std::clamp(avail.y * 0.74f, avail.x * 0.34f, avail.x * 0.48f);

    ImGui::BeginChild("##left", ImVec2(left_w, avail.y), ImGuiChildFlags_Borders);
    board_->draw();
    ImGui::EndChild();

    ImGui::SameLine(0, gap);
    ImGui::BeginChild("##right", ImVec2(0, avail.y), ImGuiChildFlags_None);
    {
        const ImVec2 r = ImGui::GetContentRegionAvail();
        const float top_h = std::floor(r.y * 0.47f);
        ImGui::BeginChild("##network", ImVec2(0, top_h), ImGuiChildFlags_Borders);
        network_->draw();
        ImGui::EndChild();

        const float metrics_w = std::floor(r.x * 0.54f);
        ImGui::BeginChild("##metrics", ImVec2(metrics_w, 0), ImGuiChildFlags_Borders);
        metrics_->draw();
        ImGui::EndChild();
        ImGui::SameLine(0, gap);
        ImGui::BeginChild("##console", ImVec2(0, 0), ImGuiChildFlags_Borders);
        console_->draw();
        ImGui::EndChild();
    }
    ImGui::EndChild();
    ImGui::End();
}

} // namespace barracuda::gui
