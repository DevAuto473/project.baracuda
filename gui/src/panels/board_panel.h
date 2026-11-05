// Barracuda GUI - interactive chessboard (Play: Human vs. Barracuda, Observe: live self-play).
#pragma once

#include <imgui.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "barracuda/engine/notation.h"
#include "barracuda/engine/position.h"
#include "runtime/shared_state.h"

namespace barracuda::gui {

class App;

class BoardPanel {
public:
    explicit BoardPanel(App& app);

    void update(); // every frame, before drawing (engine replies, clocks)
    void draw();

    void new_game();
    bool set_fen(const std::string& fen);
    void flip() { flipped_ = !flipped_; }
    void undo();
    void stop_engine();
    const Position& game() const { return game_; }
    Value eval_white() const { return eval_white_; }

private:
    struct TimeControl {
        const char* name;
        std::int64_t base_ms;
        std::int64_t inc_ms;
        std::int64_t fixed_ms; // > 0: engine thinks a fixed time per move, clocks disabled
    };
    static const TimeControl kTimeControls[];

    void draw_header();
    void draw_board(ImVec2 origin, float size, const runtime::BoardSnapshot& b, bool interactive);
    void draw_play_controls(float width);
    void draw_observe_info(float width);
    void draw_move_list(float height);
    void handle_input(ImVec2 origin, float sq, bool interactive);
    void try_move(Square from, Square to);
    void apply_move(Move m, bool by_engine);
    void start_engine_if_needed();
    void check_game_over();
    void restart_clock();
    Square square_at(ImVec2 origin, float sq, ImVec2 p) const;
    ImVec2 square_min(ImVec2 origin, float sq, Square s) const;

    App& app_;
    // --- play mode game state (render-thread owned) ---
    Position game_;
    std::string start_fen_ = Position::START_FEN;
    std::vector<std::string> san_;
    runtime::BoardSnapshot play_snapshot_;
    Color human_ = WHITE;
    bool flipped_ = false;
    GameResult result_ = GameResult::ONGOING;
    std::string result_reason_;

    // engine
    std::uint64_t request_id_ = 0;
    bool engine_thinking_ = false;
    Value eval_white_ = 0;

    // clocks
    int tc_index_ = 2;
    std::int64_t clock_ms_[COLOR_NB] = {0, 0};
    std::chrono::steady_clock::time_point last_tick_{};
    bool clock_started_ = false;

    // interaction
    Square selected_ = SQ_NONE;
    bool dragging_ = false;
    ImVec2 drag_pos_{};
    Square promo_from_ = SQ_NONE, promo_to_ = SQ_NONE;
    bool open_promo_popup_ = false;

    // observation mode cache
    runtime::BoardSnapshot observe_board_;
    bool scroll_moves_ = false;
};

} // namespace barracuda::gui
