#include "panels/board_panel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "app.h"
#include "barracuda/engine/movegen.h"
#include "barracuda/engine/search.h"
#include "theme.h"
#include "widgets.h"

namespace barracuda::gui {

const BoardPanel::TimeControl BoardPanel::kTimeControls[] = {
    {"Bullet 1+0", 60'000, 0, 0},       {"Blitz 3+2", 180'000, 2'000, 0},  {"Blitz 5+3", 300'000, 3'000, 0},
    {"Rapid 10+5", 600'000, 5'000, 0},  {"Rapid 15+10", 900'000, 10'000, 0},
    {"Engine 0.5 s/move", 0, 0, 500},   {"Engine 2 s/move", 0, 0, 2000},   {"Engine 5 s/move", 0, 0, 5000},
};

namespace {
std::string format_clock(std::int64_t ms) {
    ms = std::max<std::int64_t>(0, ms);
    char buf[32];
    const std::int64_t m = ms / 60000, s = (ms / 1000) % 60;
    if (ms < 10'000) std::snprintf(buf, sizeof(buf), "%lld:%02lld.%lld", (long long)m, (long long)s, (long long)(ms / 100) % 10);
    else std::snprintf(buf, sizeof(buf), "%lld:%02lld", (long long)m, (long long)s);
    return buf;
}
} // namespace

BoardPanel::BoardPanel(App& app) : app_(app) { new_game(); }

// ----------------------------------------------------------------------------- game flow
void BoardPanel::new_game() {
    stop_engine();
    game_ = Position(start_fen_);
    san_.clear();
    result_ = GameResult::ONGOING;
    result_reason_.clear();
    selected_ = SQ_NONE;
    dragging_ = false;
    eval_white_ = 0;
    restart_clock();
    play_snapshot_.set(game_);
    app_.search->request_new_game();
    app_.search->request_inspect(game_);
    app_.log.logf(runtime::LogLevel::Info, "new game: you play %s (%s)", human_ == WHITE ? "White" : "Black",
                  kTimeControls[tc_index_].name);
}

bool BoardPanel::set_fen(const std::string& fen) {
    Position p;
    std::string err;
    if (!p.set_fen(fen, &err)) {
        app_.log.logf(runtime::LogLevel::Error, "%s", err.c_str());
        return false;
    }
    start_fen_ = p.fen();
    new_game();
    return true;
}

void BoardPanel::restart_clock() {
    const TimeControl& tc = kTimeControls[tc_index_];
    clock_ms_[WHITE] = clock_ms_[BLACK] = tc.base_ms;
    clock_started_ = false;
    last_tick_ = std::chrono::steady_clock::now();
}

void BoardPanel::stop_engine() {
    if (engine_thinking_) {
        app_.search->cancel();
        engine_thinking_ = false;
        ++request_id_; // any late reply becomes stale
    }
}

void BoardPanel::undo() {
    stop_engine();
    if (san_.empty()) return;
    auto pop = [&] {
        game_.undo_move();
        san_.pop_back();
    };
    pop();
    if (!san_.empty() && game_.side_to_move() != human_) pop();
    result_ = GameResult::ONGOING;
    result_reason_.clear();
    play_snapshot_.set(game_);
    app_.search->request_inspect(game_);
    app_.log.log(runtime::LogLevel::Info, "move taken back");
}

void BoardPanel::check_game_over() {
    std::string reason;
    result_ = game_result(game_, &reason);
    if (result_ != GameResult::ONGOING) {
        result_reason_ = reason;
        const char* res = result_ == GameResult::WHITE_WINS ? "1-0" : result_ == GameResult::BLACK_WINS ? "0-1" : "1/2-1/2";
        app_.log.logf(runtime::LogLevel::Success, "game over: %s (%s)", res, reason.c_str());
    }
}

void BoardPanel::apply_move(Move m, bool by_engine) {
    const Color mover = game_.side_to_move();
    san_.push_back(move_to_san(game_, m));
    game_.do_move(m);
    if (kTimeControls[tc_index_].fixed_ms == 0 && clock_started_) clock_ms_[mover] += kTimeControls[tc_index_].inc_ms;
    clock_started_ = true;
    last_tick_ = std::chrono::steady_clock::now();
    selected_ = SQ_NONE;
    play_snapshot_.set(game_);
    scroll_moves_ = true;
    if (!by_engine) app_.search->request_inspect(game_);
    app_.log.logf(by_engine ? runtime::LogLevel::Search : runtime::LogLevel::Info, "%s plays %s",
                  by_engine ? "Barracuda" : "You", san_.back().c_str());
    check_game_over();
}

void BoardPanel::start_engine_if_needed() {
    if (result_ != GameResult::ONGOING || engine_thinking_ || game_.side_to_move() == human_) return;
    const TimeControl& tc = kTimeControls[tc_index_];
    SearchLimits lim;
    if (tc.fixed_ms > 0) {
        lim.movetime_ms = tc.fixed_ms;
    } else {
        lim.time_ms[WHITE] = clock_ms_[WHITE];
        lim.time_ms[BLACK] = clock_ms_[BLACK];
        lim.inc_ms[WHITE] = lim.inc_ms[BLACK] = tc.inc_ms;
    }
    engine_thinking_ = true;
    app_.search->request_move(game_, lim, ++request_id_);
}

void BoardPanel::try_move(Square from, Square to) {
    MoveList legal;
    generate_moves(game_, legal);
    int promos = 0;
    Move found;
    for (Move m : legal) {
        if (m.from() != from || m.to() != to) continue;
        if (m.is_promotion()) ++promos;
        found = m;
    }
    if (!found) return;
    if (promos > 1) {
        promo_from_ = from;
        promo_to_ = to;
        open_promo_popup_ = true;
        return;
    }
    apply_move(found, false);
}

void BoardPanel::update() {
    // engine replies
    if (app_.search->results.update()) {
        const runtime::MoveResult& r = app_.search->results.read_buffer();
        if (engine_thinking_ && r.request_id == request_id_) {
            engine_thinking_ = false;
            eval_white_ = r.score_white;
            MoveList legal;
            generate_moves(game_, legal);
            if (r.best && legal.contains(r.best)) apply_move(r.best, true);
        }
    }
    const runtime::SearchSnapshot& si = app_.search_info();
    if (si.request_id == request_id_ && si.depth > 0) eval_white_ = si.score_white;

    // clocks
    const auto now = std::chrono::steady_clock::now();
    const std::int64_t dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tick_).count();
    last_tick_ = now;
    if (kTimeControls[tc_index_].fixed_ms == 0 && clock_started_ && result_ == GameResult::ONGOING &&
        app_.mode == ViewMode::Play) {
        const Color stm = game_.side_to_move();
        clock_ms_[stm] -= dt;
        if (clock_ms_[stm] <= 0) {
            clock_ms_[stm] = 0;
            stop_engine();
            result_ = stm == WHITE ? GameResult::BLACK_WINS : GameResult::WHITE_WINS;
            result_reason_ = std::string(stm == WHITE ? "White" : "Black") + " lost on time";
            app_.log.logf(runtime::LogLevel::Warn, "%s", result_reason_.c_str());
        }
    }

    if (app_.observation_changed()) observe_board_ = app_.observation().board;
    if (app_.mode == ViewMode::Play) start_engine_if_needed();
}

// ----------------------------------------------------------------------------- geometry
ImVec2 BoardPanel::square_min(ImVec2 origin, float sq, Square s) const {
    const int col = flipped_ ? 7 - file_of(s) : file_of(s);
    const int row = flipped_ ? rank_of(s) : 7 - rank_of(s);
    return ImVec2(origin.x + float(col) * sq, origin.y + float(row) * sq);
}

Square BoardPanel::square_at(ImVec2 origin, float sq, ImVec2 p) const {
    const int col = int(std::floor((p.x - origin.x) / sq));
    const int row = int(std::floor((p.y - origin.y) / sq));
    if (col < 0 || col > 7 || row < 0 || row > 7) return SQ_NONE;
    const int file = flipped_ ? 7 - col : col;
    const int rank = flipped_ ? row : 7 - row;
    return make_square(file, rank);
}

// ----------------------------------------------------------------------------- drawing
void BoardPanel::draw_board(ImVec2 o, float size, const runtime::BoardSnapshot& b, bool interactive) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float sq = size / 8.0f;

    // frame
    dl->AddRectFilled(ImVec2(o.x - 2, o.y - 2), ImVec2(o.x + size + 2, o.y + size + 2), palette::kBorder);

    MoveList legal;
    if (interactive && selected_ != SQ_NONE) generate_moves(game_, legal);

    for (int s = 0; s < 64; ++s) {
        const Square sqr = Square(s);
        const ImVec2 a = square_min(o, sq, sqr);
        const ImVec2 c(a.x + sq, a.y + sq);
        const bool light = (file_of(sqr) + rank_of(sqr)) % 2 == 1;
        ImDrawFlags corners = ImDrawFlags_RoundCornersNone;
        float round = 0.0f;
        const int col = int((a.x - o.x) / sq + 0.5f), row = int((a.y - o.y) / sq + 0.5f);
        if (col == 0 && row == 0) corners = ImDrawFlags_RoundCornersTopLeft;
        if (col == 7 && row == 0) corners = ImDrawFlags_RoundCornersTopRight;
        if (col == 0 && row == 7) corners = ImDrawFlags_RoundCornersBottomLeft;
        if (col == 7 && row == 7) corners = ImDrawFlags_RoundCornersBottomRight;
        dl->AddRectFilled(a, c, light ? palette::kLightSquare : palette::kDarkSquare, round, corners);

        if (b.last_move && (sqr == b.last_move.from() || sqr == b.last_move.to()))
            dl->AddRectFilled(a, c, palette::kLastMove, round, corners);
        if (interactive && sqr == selected_) dl->AddRectFilled(a, c, palette::kSelected, round, corners);
        if (b.in_check && b.board[std::size_t(s)] == make_piece(b.stm, KING)) {
            const ImVec2 mid((a.x + c.x) * 0.5f, (a.y + c.y) * 0.5f);
            for (int r = 5; r >= 1; --r)
                dl->AddCircleFilled(mid, sq * 0.11f * float(r), palette::with_alpha(palette::kCheck, 0.12f), 32);
        }

        // coordinates
        const ImU32 coord = light ? palette::kDarkSquare : palette::kLightSquare;
        ImFont* f = app_.fonts.small;
        if (col == 0) {
            char t[2] = {char('1' + rank_of(sqr)), 0};
            dl->AddText(f, f->FontSize, ImVec2(a.x + 3, a.y + 2), coord, t);
        }
        if (row == 7) {
            char t[2] = {char('a' + file_of(sqr)), 0};
            dl->AddText(f, f->FontSize, ImVec2(c.x - f->FontSize * 0.6f - 2, c.y - f->FontSize - 1), coord, t);
        }
    }

    // pieces
    for (int s = 0; s < 64; ++s) {
        const Piece p = b.board[std::size_t(s)];
        if (p == NO_PIECE) continue;
        const bool lifted = interactive && dragging_ && Square(s) == selected_;
        draw_piece(dl, p, square_min(o, sq, Square(s)), sq, lifted ? 0.35f : 1.0f);
    }

    // legal move hints
    if (interactive && selected_ != SQ_NONE) {
        for (Move m : legal) {
            if (m.from() != selected_) continue;
            const ImVec2 a = square_min(o, sq, m.to());
            const ImVec2 mid(a.x + sq * 0.5f, a.y + sq * 0.5f);
            if (b.board[m.to()] != NO_PIECE) dl->AddCircle(mid, sq * 0.44f, palette::rgba(0, 122, 204, 190), 40, sq * 0.07f);
            else dl->AddCircleFilled(mid, sq * 0.15f, palette::rgba(0, 122, 204, 170), 24);
        }
    }

    // dragged piece follows the cursor
    if (interactive && dragging_ && selected_ != SQ_NONE && b.board[selected_] != NO_PIECE) {
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        draw_piece(fg, b.board[selected_], ImVec2(drag_pos_.x - sq * 0.5f, drag_pos_.y - sq * 0.55f), sq * 1.08f);
    }
}

void BoardPanel::handle_input(ImVec2 o, float sq, bool interactive) {
    ImGui::SetCursorScreenPos(o);
    ImGui::InvisibleButton("##board", ImVec2(sq * 8, sq * 8));
    if (!interactive) return;
    const bool my_turn = result_ == GameResult::ONGOING && game_.side_to_move() == human_ && !engine_thinking_;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    drag_pos_ = mouse;

    if (ImGui::IsItemActivated()) {
        const Square s = square_at(o, sq, mouse);
        if (s != SQ_NONE && my_turn) {
            const Piece p = game_.piece_on(s);
            if (p != NO_PIECE && color_of(p) == human_) {
                if (selected_ == s) {
                    dragging_ = true; // re-grab the selected piece
                } else {
                    selected_ = s;
                    dragging_ = true;
                }
            } else if (selected_ != SQ_NONE) {
                try_move(selected_, s);
                if (game_.side_to_move() == human_ && !open_promo_popup_) selected_ = SQ_NONE;
            }
        } else if (s != SQ_NONE && !my_turn) {
            selected_ = SQ_NONE;
        }
    }
    if (dragging_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        dragging_ = false;
        const Square to = square_at(o, sq, mouse);
        if (to != SQ_NONE && selected_ != SQ_NONE && to != selected_ && my_turn) try_move(selected_, to);
    }
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
        selected_ = SQ_NONE;
        dragging_ = false;
    }
}

void BoardPanel::draw_header() {
    const bool play = app_.mode == ViewMode::Play;
    auto seg = [&](const char* label, bool active) {
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, palette::to_vec4(palette::kAccentBlue));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, palette::to_vec4(palette::kAccentBlue));
        }
        const bool clicked = ImGui::Button(label);
        if (active) ImGui::PopStyleColor(2);
        return clicked;
    };
    ImGui::PushFont(app_.fonts.title);
    ImGui::TextUnformatted("Board");
    ImGui::PopFont();
    ImGui::SameLine(0, 16);
    if (seg("Play vs Barracuda", play)) app_.mode = ViewMode::Play;
    ImGui::SameLine(0, 4);
    if (seg("Observe Training", !play)) {
        if (play) stop_engine();
        app_.mode = ViewMode::Observe;
    }
    ImGui::SameLine(0, 12);
    if (ImGui::Button(flipped_ ? "Flip (B)" : "Flip (W)")) flip();
}

void BoardPanel::draw_play_controls(float width) {
    const TimeControl& tc = kTimeControls[tc_index_];
    const Color engine = ~human_;
    const float tile_w = (width - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

    auto clock_tile = [&](Color c, const char* who) {
        const bool active = result_ == GameResult::ONGOING && game_.side_to_move() == c;
        std::string v = tc.fixed_ms ? std::string(c == engine ? "fixed time" : "untimed") : format_clock(clock_ms_[c]);
        if (active && c == engine && engine_thinking_) v += "   thinking...";
        const ImU32 col = active ? palette::kAccent : palette::kText;
        char label[64];
        std::snprintf(label, sizeof(label), "%s  -  %s", who, c == WHITE ? "White" : "Black");
        stat_tile(label, v, tile_w, col);
    };
    clock_tile(engine, "Barracuda");
    ImGui::SameLine();
    clock_tile(human_, "You");

    // status
    std::string status;
    if (result_ != GameResult::ONGOING) {
        status = std::string(result_ == GameResult::WHITE_WINS ? "1-0" : result_ == GameResult::BLACK_WINS ? "0-1" : "1/2-1/2") +
                 "  " + result_reason_;
        ImGui::TextColored(palette::to_vec4(palette::kWarning), "%s", status.c_str());
    } else if (game_.side_to_move() == human_) {
        ImGui::TextColored(palette::to_vec4(palette::kPositive), "Your move%s", game_.in_check() ? " - you are in check!" : "");
    } else {
        const auto& si = app_.search_info();
        if (si.request_id == request_id_ && si.depth > 0)
            ImGui::TextColored(palette::to_vec4(palette::kTextDim), "Barracuda thinking: depth %d, %s nps", si.depth,
                               format_count(si.nps).c_str());
        else
            ImGui::TextColored(palette::to_vec4(palette::kTextDim), "Barracuda thinking...");
    }

    // controls
    if (ImGui::Button("New game")) new_game();
    ImGui::SameLine();
    if (ImGui::Button("Undo")) undo();
    ImGui::SameLine();
    if (ImGui::Button("Swap sides")) {
        human_ = ~human_;
        flipped_ = human_ == BLACK;
        new_game();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::BeginCombo("##tc", tc.name)) {
        for (int i = 0; i < int(std::size(kTimeControls)); ++i) {
            if (ImGui::Selectable(kTimeControls[i].name, i == tc_index_)) {
                tc_index_ = i;
                new_game();
            }
        }
        ImGui::EndCombo();
    }
}

void BoardPanel::draw_observe_info(float width) {
    const runtime::ObservationSnapshot& o = app_.observation();
    const runtime::TrainingSnapshot& t = app_.training_stats();
    const float tile_w = (width - 2 * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
    if (o.active) {
        const bool matching = t.task == runtime::TrainingTask::Match;
        stat_tile(matching ? "Elo match game" : "Self-play game",
                  std::to_string(o.game) + " / " + std::to_string(matching ? t.match_total : t.sp_games_total), tile_w);
        ImGui::SameLine();
        stat_tile("Ply", std::to_string(o.ply), tile_w);
        ImGui::SameLine();
        stat_tile("Eval (White)", score_to_string(o.score_white), tile_w,
                  o.score_white >= 0 ? palette::kPositive : palette::kNegative);
    } else {
        ImGui::TextColored(palette::to_vec4(palette::kTextDim),
                           "Idle - start self-play in the Training tab to watch Barracuda generate data.");
    }
    int delay = app_.training->observe_delay_ms.load();
    ImGui::SetNextItemWidth(width * 0.6f);
    if (ImGui::SliderInt("Watch delay (ms/move)", &delay, 0, 1000)) app_.training->observe_delay_ms.store(delay);
}

void BoardPanel::draw_move_list(float height) {
    if (height < 40) return;
    ImGui::BeginChild("##moves", ImVec2(0, height), ImGuiChildFlags_None);
    ImGui::PushFont(app_.fonts.mono);
    const bool black_first = Position(start_fen_).side_to_move() == BLACK;
    const int first_no = Position(start_fen_).fullmove_number();
    std::string line;
    int idx = 0;
    if (black_first && !san_.empty()) {
        ImGui::Text("%3d. ...    %-8s", first_no, san_[0].c_str());
        idx = 1;
    }
    int no = first_no + (black_first ? 1 : 0);
    for (; idx < int(san_.size()); idx += 2, ++no) {
        const char* b = idx + 1 < int(san_.size()) ? san_[std::size_t(idx + 1)].c_str() : "";
        ImGui::Text("%3d. %-8s %-8s", no, san_[std::size_t(idx)].c_str(), b);
    }
    ImGui::PopFont();
    if (scroll_moves_) {
        ImGui::SetScrollHereY(1.0f);
        scroll_moves_ = false;
    }
    ImGui::EndChild();
}

void BoardPanel::draw() {
    draw_header();
    const bool play = app_.mode == ViewMode::Play;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float bar_w = 18.0f;
    const float gap = 10.0f;
    const float controls_h = ImGui::GetTextLineHeightWithSpacing() * (play ? 5.2f : 4.4f);
    const float board_size = std::floor(std::max(160.0f, std::min(avail.x - bar_w - gap - 6.0f, avail.y - controls_h - 70.0f)) / 8.0f) * 8.0f;

    const ImVec2 cur = ImGui::GetCursorScreenPos();
    const ImVec2 origin(cur.x + bar_w + gap + 3.0f, cur.y + 4.0f);

    const runtime::BoardSnapshot& b = play ? play_snapshot_ : observe_board_;
    const Value eval = play ? eval_white_ : app_.observation().score_white;
    eval_bar(ImGui::GetWindowDrawList(), ImVec2(cur.x, origin.y), ImVec2(cur.x + bar_w, origin.y + board_size), eval,
             flipped_, app_.fonts.small);

    draw_board(origin, board_size, b, play);
    handle_input(origin, board_size / 8.0f, play);

    ImGui::SetCursorScreenPos(ImVec2(cur.x, origin.y + board_size + 10.0f));
    const float width = bar_w + gap + board_size + 3.0f;
    ImGui::BeginGroup();
    if (play) draw_play_controls(width);
    else draw_observe_info(width);
    ImGui::EndGroup();
    if (play) draw_move_list(ImGui::GetContentRegionAvail().y - 4.0f);

    // promotion chooser
    if (open_promo_popup_) {
        ImGui::OpenPopup("Promote to");
        open_promo_popup_ = false;
    }
    if (ImGui::BeginPopup("Promote to")) {
        const float s = 56.0f;
        for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT}) {
            ImGui::PushID(int(pt));
            const ImVec2 p = ImGui::GetCursorScreenPos();
            if (ImGui::InvisibleButton("promo", ImVec2(s, s))) {
                MoveList legal;
                generate_moves(game_, legal);
                for (Move m : legal)
                    if (m.from() == promo_from_ && m.to() == promo_to_ && m.is_promotion() && m.promotion_type() == pt)
                        apply_move(m, false);
                ImGui::CloseCurrentPopup();
            }
            ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + s, p.y + s),
                                                      ImGui::IsItemHovered() ? palette::kSelected : palette::kPanelAlt);
            draw_piece(ImGui::GetWindowDrawList(), make_piece(human_, pt), p, s);
            ImGui::PopID();
            ImGui::SameLine();
        }
        ImGui::NewLine();
        ImGui::EndPopup();
    }
}

} // namespace barracuda::gui
