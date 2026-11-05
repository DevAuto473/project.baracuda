// Barracuda GUI - custom drawing helpers: vector chess pieces, line charts, eval bar, stat tiles.
#pragma once

#include <imgui.h>

#include <string>
#include <vector>

#include "barracuda/engine/types.h"

namespace barracuda::gui {

// Draws a resolution-independent vector piece inside the square [min, min+size].
void draw_piece(ImDrawList* dl, Piece p, ImVec2 min, float size, float alpha = 1.0f);

struct ChartSeries {
    const std::vector<float>* data = nullptr;
    ImU32 color = 0;
    const char* label = "";
};

// Line chart with grid, auto-scaling (optionally from zero) and a hover readout.
void line_chart(const char* id, const std::vector<ChartSeries>& series, ImVec2 size, const char* value_fmt = "%.4f",
                bool from_zero = false, const char* empty_text = "no data yet");

// Vertical evaluation bar (white at the bottom unless flipped).
void eval_bar(ImDrawList* dl, ImVec2 min, ImVec2 max, Value score_white, bool flipped, ImFont* font);

// Compact "label / value" tile.
void stat_tile(const char* label, const std::string& value, float width, ImU32 value_color = 0);

std::string format_count(std::uint64_t n); // 1.23M, 45.6k ...

} // namespace barracuda::gui
