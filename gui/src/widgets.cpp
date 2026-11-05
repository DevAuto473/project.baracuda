#include "widgets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "theme.h"

namespace barracuda::gui {

namespace {

ImVec2 map(ImVec2 p, ImVec2 min, float s) { return ImVec2(min.x + p.x * s, min.y + p.y * s); }

void fill_poly(ImDrawList* dl, const std::vector<ImVec2>& pts, ImVec2 min, float s, ImU32 fill, ImU32 edge, float th) {
    std::vector<ImVec2> m;
    m.reserve(pts.size());
    for (const ImVec2& p : pts) m.push_back(map(p, min, s));
    dl->AddConcavePolyFilled(m.data(), int(m.size()), fill);
    dl->AddPolyline(m.data(), int(m.size()), edge, ImDrawFlags_Closed, th);
}

void fill_circle(ImDrawList* dl, ImVec2 c, float r, ImVec2 min, float s, ImU32 fill, ImU32 edge, float th) {
    const ImVec2 mc = map(c, min, s);
    dl->AddCircleFilled(mc, r * s, fill, 24);
    dl->AddCircle(mc, r * s, edge, 24, th);
}

void fill_rect(ImDrawList* dl, ImVec2 a, ImVec2 b, float round, ImVec2 min, float s, ImU32 fill, ImU32 edge, float th) {
    dl->AddRectFilled(map(a, min, s), map(b, min, s), fill, round * s);
    dl->AddRect(map(a, min, s), map(b, min, s), edge, round * s, 0, th);
}

} // namespace

void draw_piece(ImDrawList* dl, Piece p, ImVec2 min, float s, float alpha) {
    if (p == NO_PIECE) return;
    const bool white = color_of(p) == WHITE;
    const ImU32 fill = palette::with_alpha(white ? palette::kWhitePiece : palette::kBlackPiece, alpha);
    const ImU32 edge = palette::with_alpha(white ? palette::kWhitePieceEdge : palette::kBlackPieceEdge, alpha);
    const float th = std::max(1.0f, s * 0.028f);

    // soft drop shadow
    dl->AddRectFilled(map(ImVec2(0.25f, 0.86f), min, s), map(ImVec2(0.75f, 0.90f), min, s),
                      palette::rgba(0, 0, 0, int(60 * alpha)), s * 0.03f);

    switch (type_of(p)) {
    case PAWN:
        fill_poly(dl, {{0.40f, 0.47f}, {0.60f, 0.47f}, {0.67f, 0.79f}, {0.33f, 0.79f}}, min, s, fill, edge, th);
        fill_rect(dl, {0.35f, 0.44f}, {0.65f, 0.51f}, 0.03f, min, s, fill, edge, th);
        fill_circle(dl, {0.5f, 0.34f}, 0.115f, min, s, fill, edge, th);
        fill_rect(dl, {0.25f, 0.77f}, {0.75f, 0.86f}, 0.03f, min, s, fill, edge, th);
        break;
    case KNIGHT: {
        fill_poly(dl,
                  {{0.30f, 0.79f}, {0.33f, 0.62f}, {0.43f, 0.50f}, {0.40f, 0.46f}, {0.30f, 0.52f}, {0.21f, 0.47f},
                   {0.22f, 0.39f}, {0.38f, 0.26f}, {0.42f, 0.15f}, {0.49f, 0.21f}, {0.58f, 0.19f}, {0.69f, 0.28f},
                   {0.76f, 0.46f}, {0.74f, 0.79f}},
                  min, s, fill, edge, th);
        dl->AddCircleFilled(map(ImVec2(0.46f, 0.31f), min, s), s * 0.025f, edge);
        // mane accent
        dl->AddLine(map(ImVec2(0.60f, 0.27f), min, s), map(ImVec2(0.68f, 0.52f), min, s), palette::with_alpha(edge, 0.5f * alpha), th * 0.8f);
        fill_rect(dl, {0.25f, 0.77f}, {0.75f, 0.86f}, 0.03f, min, s, fill, edge, th);
        break;
    }
    case BISHOP: {
        fill_poly(dl, {{0.40f, 0.62f}, {0.60f, 0.62f}, {0.66f, 0.79f}, {0.34f, 0.79f}}, min, s, fill, edge, th);
        std::vector<ImVec2> mitre;
        const int n = 30;
        for (int i = 0; i <= n; ++i) {
            const float a = 3.14159265f * float(i) / float(n); // 0..pi, right->left over the top
            const float x = 0.5f + 0.16f * std::cos(a);
            const float sn = std::max(0.0f, std::sin(a)); // sin(pi) can be slightly negative -> pow() NaN
            const float y = 0.60f - 0.36f * std::pow(sn, 1.6f) - 0.04f * sn;
            mitre.push_back(ImVec2(x, y));
        }
        fill_poly(dl, mitre, min, s, fill, edge, th);
        dl->AddLine(map(ImVec2(0.44f, 0.36f), min, s), map(ImVec2(0.56f, 0.48f), min, s), edge, th);
        fill_circle(dl, {0.5f, 0.165f}, 0.045f, min, s, fill, edge, th);
        fill_rect(dl, {0.36f, 0.58f}, {0.64f, 0.64f}, 0.02f, min, s, fill, edge, th);
        fill_rect(dl, {0.25f, 0.77f}, {0.75f, 0.86f}, 0.03f, min, s, fill, edge, th);
        break;
    }
    case ROOK:
        fill_poly(dl, {{0.34f, 0.38f}, {0.66f, 0.38f}, {0.69f, 0.77f}, {0.31f, 0.77f}}, min, s, fill, edge, th);
        fill_poly(dl,
                  {{0.27f, 0.20f}, {0.36f, 0.20f}, {0.36f, 0.27f}, {0.45f, 0.27f}, {0.45f, 0.20f}, {0.55f, 0.20f},
                   {0.55f, 0.27f}, {0.64f, 0.27f}, {0.64f, 0.20f}, {0.73f, 0.20f}, {0.73f, 0.38f}, {0.27f, 0.38f}},
                  min, s, fill, edge, th);
        fill_rect(dl, {0.25f, 0.76f}, {0.75f, 0.86f}, 0.03f, min, s, fill, edge, th);
        break;
    case QUEEN:
        fill_poly(dl,
                  {{0.30f, 0.74f}, {0.22f, 0.30f}, {0.37f, 0.52f}, {0.40f, 0.22f}, {0.50f, 0.49f}, {0.60f, 0.22f},
                   {0.63f, 0.52f}, {0.78f, 0.30f}, {0.70f, 0.74f}},
                  min, s, fill, edge, th);
        for (ImVec2 c : {ImVec2(0.22f, 0.27f), ImVec2(0.40f, 0.19f), ImVec2(0.60f, 0.19f), ImVec2(0.78f, 0.27f)})
            fill_circle(dl, c, 0.04f, min, s, fill, edge, th);
        fill_rect(dl, {0.30f, 0.70f}, {0.70f, 0.77f}, 0.02f, min, s, fill, edge, th);
        fill_rect(dl, {0.25f, 0.76f}, {0.75f, 0.86f}, 0.03f, min, s, fill, edge, th);
        break;
    case KING:
        fill_poly(dl,
                  {{0.31f, 0.75f}, {0.22f, 0.47f}, {0.30f, 0.38f}, {0.42f, 0.40f}, {0.50f, 0.47f}, {0.58f, 0.40f},
                   {0.70f, 0.38f}, {0.78f, 0.47f}, {0.69f, 0.75f}},
                  min, s, fill, edge, th);
        fill_rect(dl, {0.465f, 0.12f}, {0.535f, 0.42f}, 0.01f, min, s, fill, edge, th);
        fill_rect(dl, {0.39f, 0.19f}, {0.61f, 0.25f}, 0.01f, min, s, fill, edge, th);
        fill_rect(dl, {0.30f, 0.70f}, {0.70f, 0.77f}, 0.02f, min, s, fill, edge, th);
        fill_rect(dl, {0.25f, 0.76f}, {0.75f, 0.86f}, 0.03f, min, s, fill, edge, th);
        break;
    default:
        break;
    }
}

void line_chart(const char* id, const std::vector<ChartSeries>& series, ImVec2 size, const char* fmt, bool from_zero,
                const char* empty_text) {
    ImGui::PushID(id);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    if (size.x <= 0) size.x = ImGui::GetContentRegionAvail().x;
    if (size.y <= 0) size.y = 120;
    ImGui::InvisibleButton("chart", size);
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + size.x, p0.y + size.y);
    dl->AddRectFilled(p0, p1, palette::kPanelAlt);

    float lo = 1e30f, hi = -1e30f;
    std::size_t longest = 0;
    for (const auto& s : series) {
        if (!s.data) continue;
        longest = std::max(longest, s.data->size());
        for (float v : *s.data) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    if (longest < 2) {
        const ImVec2 ts = ImGui::CalcTextSize(empty_text);
        dl->AddText(ImVec2(p0.x + (size.x - ts.x) * 0.5f, p0.y + (size.y - ts.y) * 0.5f), palette::kTextDim, empty_text);
        ImGui::PopID();
        return;
    }
    if (from_zero) lo = std::min(lo, 0.0f);
    if (hi - lo < 1e-9f) {
        hi += 0.5f;
        lo -= 0.5f;
    }
    const float pad = (hi - lo) * 0.08f;
    hi += pad;
    lo -= from_zero && lo >= 0.0f ? 0.0f : pad;

    const float left = 8.0f, right = 8.0f, top = 8.0f, bottom = 8.0f;
    const ImVec2 a(p0.x + left, p0.y + top), b(p1.x - right, p1.y - bottom);
    // grid + labels
    const float label_fs = ImGui::GetFontSize() * 0.8f;
    const bool dense = (b.y - a.y) / 3.0f < label_fs * 1.6f; // short chart: label top/bottom only
    for (int g = 0; g <= 3; ++g) {
        const float y = a.y + (b.y - a.y) * float(g) / 3.0f;
        dl->AddLine(ImVec2(a.x, y), ImVec2(b.x, y), palette::kGrid);
        if (dense && g != 0 && g != 3) continue;
        char buf[48];
        std::snprintf(buf, sizeof(buf), fmt, double(hi - (hi - lo) * float(g) / 3.0f));
        dl->AddText(ImGui::GetFont(), label_fs, ImVec2(a.x + 2, y - (g == 3 ? label_fs + 1.0f : -1.0f)),
                    palette::kTextDim, buf);
    }

    const float denom = float(std::max<std::size_t>(1, longest - 1));
    for (const auto& s : series) {
        if (!s.data || s.data->size() < 1) continue;
        const auto& d = *s.data;
        std::vector<ImVec2> pts;
        pts.reserve(d.size());
        for (std::size_t i = 0; i < d.size(); ++i) {
            const float x = a.x + (b.x - a.x) * float(i) / denom;
            const float y = b.y - (b.y - a.y) * (d[i] - lo) / (hi - lo);
            pts.emplace_back(x, y);
        }
        if (pts.size() >= 2) {
            // area fill under the primary series
            if (&s == &series.front()) {
                for (std::size_t i = 1; i < pts.size(); ++i)
                    dl->AddQuadFilled(pts[i - 1], pts[i], ImVec2(pts[i].x, b.y), ImVec2(pts[i - 1].x, b.y),
                                      palette::with_alpha(s.color, 0.10f));
            }
            dl->AddPolyline(pts.data(), int(pts.size()), s.color, 0, 2.0f);
        }
        dl->AddCircleFilled(pts.back(), 3.0f, s.color);
    }

    // legend
    float lx = b.x;
    for (auto it = series.rbegin(); it != series.rend(); ++it) {
        if (!it->label || !*it->label) continue;
        const ImVec2 ts = ImGui::CalcTextSize(it->label);
        lx -= ts.x;
        dl->AddText(ImVec2(lx, a.y), it->color, it->label);
        lx -= 14.0f;
    }

    if (hovered) {
        const float mx = ImGui::GetIO().MousePos.x;
        const std::size_t idx = std::size_t(std::clamp((mx - a.x) / (b.x - a.x), 0.0f, 1.0f) * denom + 0.5f);
        dl->AddLine(ImVec2(a.x + (b.x - a.x) * float(idx) / denom, a.y), ImVec2(a.x + (b.x - a.x) * float(idx) / denom, b.y),
                    palette::with_alpha(palette::kText, 0.25f));
        ImGui::BeginTooltip();
        ImGui::Text("#%d", int(idx) + 1);
        for (const auto& s : series) {
            if (!s.data || idx >= s.data->size()) continue;
            char buf[64];
            std::snprintf(buf, sizeof(buf), fmt, double((*s.data)[idx]));
            ImGui::TextColored(palette::to_vec4(s.color), "%s %s", s.label, buf);
        }
        ImGui::EndTooltip();
    }
    ImGui::PopID();
}

void eval_bar(ImDrawList* dl, ImVec2 min, ImVec2 max, Value score_white, bool flipped, ImFont* font) {
    // map score to white share through a logistic curve (same scale as the NNUE target)
    float share;
    if (is_mate_score(score_white)) share = score_white > 0 ? 1.0f : 0.0f;
    else share = 1.0f / (1.0f + std::exp(-float(score_white) / 300.0f));
    share = std::clamp(share, 0.03f, 0.97f);
    const float h = max.y - min.y;
    dl->AddRectFilled(min, max, palette::rgba(26, 26, 26));
    const float split = flipped ? min.y + h * share : max.y - h * share;
    if (!flipped) dl->AddRectFilled(ImVec2(min.x, split), max, palette::kWhitePiece);
    else dl->AddRectFilled(min, ImVec2(max.x, split), palette::kWhitePiece);
    dl->AddLine(ImVec2(min.x, (min.y + max.y) * 0.5f), ImVec2(max.x, (min.y + max.y) * 0.5f), palette::with_alpha(palette::kAccent, 0.6f));

    char buf[16];
    if (is_mate_score(score_white)) std::snprintf(buf, sizeof(buf), "M%d", (VALUE_MATE - std::abs(score_white) + 1) / 2);
    else std::snprintf(buf, sizeof(buf), "%.1f", std::abs(double(score_white)) / 100.0);
    const float fs = font ? font->FontSize : ImGui::GetFontSize();
    const ImVec2 ts = (font ? font : ImGui::GetFont())->CalcTextSizeA(fs, 1e9f, 0.0f, buf);
    const bool white_ahead = score_white >= 0;
    const bool text_bottom = white_ahead != flipped;
    const ImVec2 tp(min.x + (max.x - min.x - ts.x) * 0.5f, text_bottom ? max.y - ts.y - 4 : min.y + 4);
    dl->AddText(font, fs, tp, white_ahead ? palette::kBlackPiece : palette::kWhitePiece, buf);
}

void stat_tile(const char* label, const std::string& value, float width, ImU32 value_color) {
    ImGui::BeginGroup();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight() * 2.6f;
    ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + width, p.y + h), palette::kPanelAlt);
    ImGui::SetCursorScreenPos(ImVec2(p.x + 8, p.y + 4));
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "%s", label);
    ImGui::SetCursorScreenPos(ImVec2(p.x + 8, p.y + 4 + ImGui::GetTextLineHeight() * 1.15f));
    ImGui::TextColored(palette::to_vec4(value_color ? value_color : palette::kText), "%s", value.c_str());
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(width, h));
    ImGui::EndGroup();
}

std::string format_count(std::uint64_t n) {
    char buf[32];
    if (n >= 1'000'000'000ULL) std::snprintf(buf, sizeof(buf), "%.2fG", double(n) / 1e9);
    else if (n >= 1'000'000ULL) std::snprintf(buf, sizeof(buf), "%.2fM", double(n) / 1e6);
    else if (n >= 10'000ULL) std::snprintf(buf, sizeof(buf), "%.1fk", double(n) / 1e3);
    else std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)n);
    return buf;
}

} // namespace barracuda::gui
