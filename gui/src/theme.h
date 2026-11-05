// Barracuda GUI - dark minimalist theme, palette and font loading.
#pragma once

#include <imgui.h>

namespace barracuda::gui {

namespace palette {
inline constexpr ImU32 rgba(int r, int g, int b, int a = 255) {
    return ImU32(r) | (ImU32(g) << 8) | (ImU32(b) << 16) | (ImU32(a) << 24);
}
inline constexpr ImU32 kBackground = rgba(14, 16, 20);
inline constexpr ImU32 kPanel = rgba(20, 23, 29);
inline constexpr ImU32 kPanelAlt = rgba(26, 30, 37);
inline constexpr ImU32 kBorder = rgba(42, 47, 57);
inline constexpr ImU32 kGrid = rgba(38, 43, 52);
inline constexpr ImU32 kText = rgba(222, 226, 233);
inline constexpr ImU32 kTextDim = rgba(128, 136, 150);
inline constexpr ImU32 kAccent = rgba(56, 189, 248);    // cyan
inline constexpr ImU32 kAccent2 = rgba(167, 139, 250);  // violet
inline constexpr ImU32 kPositive = rgba(52, 211, 153);  // green
inline constexpr ImU32 kNegative = rgba(251, 113, 133); // rose
inline constexpr ImU32 kWarning = rgba(251, 191, 36);   // amber
inline constexpr ImU32 kLightSquare = rgba(173, 184, 196);
inline constexpr ImU32 kDarkSquare = rgba(92, 109, 128);
inline constexpr ImU32 kLastMove = rgba(250, 204, 21, 90);
inline constexpr ImU32 kSelected = rgba(56, 189, 248, 120);
inline constexpr ImU32 kCheck = rgba(239, 68, 68, 170);
inline constexpr ImU32 kWhitePiece = rgba(244, 241, 234);
inline constexpr ImU32 kWhitePieceEdge = rgba(28, 30, 34);
inline constexpr ImU32 kBlackPiece = rgba(30, 33, 39);
inline constexpr ImU32 kBlackPieceEdge = rgba(205, 210, 220);

inline ImVec4 to_vec4(ImU32 c) { return ImGui::ColorConvertU32ToFloat4(c); }
inline ImU32 with_alpha(ImU32 c, float a) {
    const int al = int(a * 255.0f + 0.5f);
    return (c & 0x00FFFFFFu) | (ImU32(al < 0 ? 0 : (al > 255 ? 255 : al)) << 24);
}
inline ImU32 lerp(ImU32 a, ImU32 b, float t) {
    const ImVec4 va = to_vec4(a), vb = to_vec4(b);
    return ImGui::ColorConvertFloat4ToU32(ImVec4(va.x + (vb.x - va.x) * t, va.y + (vb.y - va.y) * t,
                                                 va.z + (vb.z - va.z) * t, va.w + (vb.w - va.w) * t));
}
} // namespace palette

struct Fonts {
    ImFont* regular = nullptr;
    ImFont* small = nullptr;
    ImFont* mono = nullptr;
    ImFont* large = nullptr;
    ImFont* title = nullptr;
};

// Applies the theme and loads fonts (system TTF if available, otherwise ImGui's built-in font).
Fonts setup_theme(float dpi_scale);

} // namespace barracuda::gui
