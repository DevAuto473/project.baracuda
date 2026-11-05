#include "panels/network_panel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "app.h"
#include "barracuda/engine/search.h"
#include "theme.h"

namespace barracuda::gui {

namespace {
constexpr ImU32 kLayerColors[nnue::NetworkView::kLayers] = {
    palette::rgba(200, 200, 200), // input
    palette::rgba(0, 229, 255),   // FT (cyan)
    palette::rgba(90, 150, 255),  // L1
    palette::rgba(232, 121, 249), // L2 (magenta)
    palette::rgba(63, 230, 140),  // output
};
} // namespace

void NetworkPanel::sync(const nnue::NetworkView& v) {
    const float dt = ImGui::GetIO().DeltaTime;
    const float k = std::clamp(dt * 10.0f, 0.0f, 1.0f);
    for (int l = 0; l < nnue::NetworkView::kLayers; ++l) {
        const auto& target = v.layers[std::size_t(l)].activation;
        auto& cur = shown_[std::size_t(l)];
        if (cur.size() != target.size()) cur = target; // topology changed (e.g. different piece count)
        for (std::size_t i = 0; i < cur.size(); ++i) cur[i] += (target[i] - cur[i]) * k;
    }
}

void NetworkPanel::draw() {
    const nnue::NetworkView& v = app_.network_view();

    // ---- header ------------------------------------------------------------
    ImGui::PushFont(app_.fonts.title);
    ImGui::TextUnformatted("NNUE Architecture");
    ImGui::PopFont();
    ImGui::SameLine(0, 12);
    ImGui::TextColored(palette::to_vec4(palette::kTextDim), "768 -> 2x256 -> 32 -> 32 -> 1   int16 FT / int8 dense / ClippedReLU [0,127]   SIMD %s",
                       nnue::simd_backend());

    if (!v.valid) {
        ImGui::TextColored(palette::to_vec4(palette::kTextDim), "waiting for the first network snapshot...");
        return;
    }
    sync(v);

    if (v.trained) ImGui::TextColored(palette::to_vec4(palette::kPositive), "trained: %u epochs", v.epochs);
    else ImGui::TextColored(palette::to_vec4(palette::kWarning), "untrained (random weights) - classical eval drives the search");
    ImGui::SameLine(0, 18);
    ImGui::TextColored(palette::to_vec4(palette::kText), "network eval: %s (White)",
                       score_to_string(v.eval_white_cp).c_str());

    const float ctrl_w = 150.0f;
    ImGui::SameLine(0, 18);
    {
        // right-align the visualiser controls (width measured on the previous frame); wrap if narrow
        const float avail = ImGui::GetContentRegionAvail().x;
        if (avail < controls_w_) ImGui::NewLine();
        else ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - controls_w_);
    }
    const float controls_x0 = ImGui::GetCursorScreenPos().x;
    ImGui::SetNextItemWidth(ctrl_w);
    ImGui::Combo("##edges", &edge_mode_, "Edges: signal |w*a|\0Edges: weight |w|\0");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ctrl_w * 0.8f);
    ImGui::SliderFloat("##thr", &edge_threshold_, 0.0f, 0.6f, "cutoff %.2f");
    ImGui::SameLine();
    ImGui::Checkbox("labels", &show_labels_);
    controls_w_ = ImGui::GetItemRectMax().x - controls_x0 + 2.0f;

    // ---- canvas ------------------------------------------------------------
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();
    size.y = std::max(size.y, 80.0f);
    ImGui::InvisibleButton("##netcanvas", size);
    const bool hovered_canvas = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(p0, ImVec2(p0.x + size.x, p0.y + size.y), true);
    dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y), palette::rgba(0, 0, 0));
    dl->AddRect(p0, ImVec2(p0.x + size.x, p0.y + size.y), palette::kBorder);

    const int L = nnue::NetworkView::kLayers;
    const float label_w = show_labels_ ? 54.0f : 10.0f;
    const float left = p0.x + label_w + 12.0f, right = p0.x + size.x - 100.0f;
    const float top = p0.y + 30.0f, bottom = p0.y + size.y - 12.0f;

    std::array<std::vector<ImVec2>, nnue::NetworkView::kLayers> pos;
    float radius = 6.0f;
    for (int l = 0; l < L; ++l) {
        const int n = int(shown_[std::size_t(l)].size());
        const float x = left + (right - left) * float(l) / float(L - 1);
        pos[std::size_t(l)].resize(std::size_t(n));
        for (int i = 0; i < n; ++i) {
            const float y = n == 1 ? (top + bottom) * 0.5f : top + (bottom - top) * (float(i) + 0.5f) / float(n);
            pos[std::size_t(l)][std::size_t(i)] = ImVec2(x, y);
        }
        if (n > 1) radius = std::min(radius, (bottom - top) / float(n) * 0.36f);
    }
    radius = std::max(radius, 2.0f);

    // hovered node
    int hover_l = -1, hover_i = -1;
    if (hovered_canvas) {
        const ImVec2 m = ImGui::GetIO().MousePos;
        float best = (radius + 4.0f) * (radius + 4.0f);
        for (int l = 0; l < L; ++l)
            for (std::size_t i = 0; i < pos[std::size_t(l)].size(); ++i) {
                const float dx = pos[std::size_t(l)][i].x - m.x, dy = pos[std::size_t(l)][i].y - m.y;
                const float r = l == L - 1 ? radius * 2.2f : radius;
                if (dx * dx + dy * dy < std::max(best, (r + 4) * (r + 4))) {
                    best = dx * dx + dy * dy;
                    hover_l = l;
                    hover_i = int(i);
                }
            }
    }

    // ---- edges ---------------------------------------------------------------
    for (int k = 0; k < L - 1; ++k) {
        const auto& w = v.edges[std::size_t(k)];
        const auto& src = shown_[std::size_t(k)];
        const std::size_t ns = pos[std::size_t(k)].size(), nd = pos[std::size_t(k + 1)].size();
        if (w.size() != ns * nd || src.size() != ns) continue;
        float maxm = 1e-6f;
        for (std::size_t s = 0; s < ns; ++s)
            for (std::size_t d = 0; d < nd; ++d) {
                const float mag = std::abs(w[s * nd + d]) * (edge_mode_ == 0 ? src[s] : 1.0f);
                maxm = std::max(maxm, mag);
            }
        for (std::size_t s = 0; s < ns; ++s) {
            for (std::size_t d = 0; d < nd; ++d) {
                const float wt = w[s * nd + d];
                float t = std::abs(wt) * (edge_mode_ == 0 ? src[s] : 1.0f) / maxm;
                const bool touches = (hover_l == k && hover_i == int(s)) || (hover_l == k + 1 && hover_i == int(d));
                if (hover_l >= 0 && !touches) t *= 0.25f;
                if (t < edge_threshold_ && !touches) continue;
                const float a = std::clamp(std::pow(t, 1.4f) * 0.9f + (touches ? 0.15f : 0.0f), 0.0f, 1.0f);
                const ImU32 base = wt >= 0 ? palette::kAccent : palette::kAccent2; // cyan + / magenta -
                const ImU32 col = palette::with_alpha(base, a);
                if (t > 0.45f) // soft glow under the strongest connections
                    dl->AddLine(pos[std::size_t(k)][s], pos[std::size_t(k + 1)][d], palette::with_alpha(base, a * 0.16f), 4.0f + 3.0f * t);
                dl->AddLine(pos[std::size_t(k)][s], pos[std::size_t(k + 1)][d], col, 0.6f + 1.8f * t);
            }
        }
    }

    // ---- nodes ---------------------------------------------------------------
    for (int l = 0; l < L; ++l) {
        const auto& layer = v.layers[std::size_t(l)];
        const auto& act = shown_[std::size_t(l)];
        const float r = l == L - 1 ? radius * 2.2f : radius;
        for (std::size_t i = 0; i < pos[std::size_t(l)].size(); ++i) {
            const float a = std::clamp(i < act.size() ? act[i] : 0.0f, 0.0f, 1.0f);
            const ImU32 fill = palette::lerp(palette::rgba(8, 8, 8), kLayerColors[l], a);
            const ImVec2 c = pos[std::size_t(l)][i];
            if (a > 0.6f) dl->AddCircleFilled(c, r * 1.9f, palette::with_alpha(kLayerColors[l], 0.10f * a), 20);
            dl->AddCircleFilled(c, r, fill, 20);
            dl->AddCircle(c, r, (hover_l == l && hover_i == int(i)) ? palette::kText : palette::with_alpha(kLayerColors[l], 0.55f), 20, 1.0f);
            if (l == 0 && show_labels_ && i < layer.labels.size() && radius >= 3.0f) {
                ImFont* f = app_.fonts.small;
                const float fs = std::min(f->FontSize, (bottom - top) / float(std::max<std::size_t>(1, act.size())) * 1.05f);
                if (fs >= 8.0f)
                    dl->AddText(f, fs, ImVec2(p0.x + 6.0f, c.y - fs * 0.5f), palette::kTextDim, layer.labels[i].c_str());
            }
        }
        // column title (name + width on two short lines so neighbouring columns never collide)
        char sub[64];
        if (l == 1) std::snprintf(sub, sizeof(sub), "%zu of %d shown", act.size(), layer.total_nodes);
        else if (l == 0) std::snprintf(sub, sizeof(sub), "%zu active of %d", act.size(), layer.total_nodes);
        else std::snprintf(sub, sizeof(sub), "%d neuron%s", layer.total_nodes, layer.total_nodes == 1 ? "" : "s");
        const char* name = l == 1 ? "Feature Transf." : layer.name.c_str();
        ImFont* sf = app_.fonts.small;
        const ImVec2 ts = sf->CalcTextSizeA(sf->FontSize, 1e9f, 0.0f, name);
        const ImVec2 ss = sf->CalcTextSizeA(sf->FontSize, 1e9f, 0.0f, sub);
        const float x = left + (right - left) * float(l) / float(L - 1);
        const float maxx = p0.x + size.x - 4.0f;
        dl->AddText(sf, sf->FontSize, ImVec2(std::clamp(x - ts.x * 0.5f, p0.x + 4.0f, maxx - ts.x), p0.y + 3.0f), kLayerColors[l], name);
        dl->AddText(sf, sf->FontSize, ImVec2(std::clamp(x - ss.x * 0.5f, p0.x + 4.0f, maxx - ss.x), p0.y + 3.0f + sf->FontSize),
                    palette::kTextDim, sub);
    }

    // output value
    if (!pos[std::size_t(L - 1)].empty()) {
        const ImVec2 c = pos[std::size_t(L - 1)][0];
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%+d cp", v.eval_cp);
        dl->AddText(app_.fonts.title, app_.fonts.title->FontSize, ImVec2(c.x + radius * 2.2f + 8.0f, c.y - 10.0f), palette::kText, buf);
        dl->AddText(app_.fonts.small, app_.fonts.small->FontSize, ImVec2(c.x + radius * 2.2f + 8.0f, c.y + 10.0f), palette::kTextDim, "side to move");
    }

    dl->PopClipRect();

    if (hover_l >= 0) {
        const auto& layer = v.layers[std::size_t(hover_l)];
        ImGui::BeginTooltip();
        ImGui::Text("%s - node %d", layer.name.c_str(), hover_i);
        if (std::size_t(hover_i) < layer.labels.size() && !layer.labels[std::size_t(hover_i)].empty())
            ImGui::Text("%s", layer.labels[std::size_t(hover_i)].c_str());
        if (std::size_t(hover_i) < layer.activation.size())
            ImGui::Text("activation %.3f", double(layer.activation[std::size_t(hover_i)]));
        ImGui::EndTooltip();
    }
}

} // namespace barracuda::gui
