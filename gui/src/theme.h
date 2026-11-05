// Barracuda GUI - dark minimalist theme, palette and font loading.
#pragma once

#include <imgui.h>

namespace barracuda::gui {

namespace palette {
inline constexpr ImU32 rgba(int r, int g, int b, int a = 255) {
    return ImU32(r) | (ImU32(g) << 8) | (ImU32(b) << 16) | (ImU32(a) << 24);
}
// VS Code "OLED / high-contrast pitch black" palette
inline constexpr ImU32 kBackground = rgba(0, 0, 0);          // #000000
inline constexpr ImU32 kPanel = rgba(5, 5, 5);               // #050505 child panels
inline constexpr ImU32 kPanelAlt = rgba(13, 13, 13);         // #0D0D0D tiles / charts / frames
inline constexpr ImU32 kBorder = rgba(150, 70, 255);         // vivid purple #9646FF
inline constexpr ImU32 kGrid = rgba(30, 30, 30);
inline constexpr ImU32 kText = rgba(224, 224, 224);          // #E0E0E0
inline constexpr ImU32 kTextDim = rgba(110, 118, 129);       // #6E7681
inline constexpr ImU32 kAccent = rgba(0, 229, 255);          // cyan
inline constexpr ImU32 kAccentBlue = rgba(0, 122, 204);      // #007ACC VS Code blue
inline constexpr ImU32 kAccent2 = rgba(232, 121, 249);       // magenta
inline constexpr ImU32 kPositive = rgba(63, 230, 140);
inline constexpr ImU32 kNegative = rgba(255, 85, 115);
inline constexpr ImU32 kWarning = rgba(255, 193, 7);
inline constexpr ImU32 kLightSquare = rgba(218, 218, 218);   // bone / light silver
inline constexpr ImU32 kDarkSquare = rgba(32, 32, 32);       // matte charcoal
inline constexpr ImU32 kLastMove = rgba(255, 193, 7, 80);    // subtle amber
inline constexpr ImU32 kSelected = rgba(0, 255, 255, 51);    // #00FFFF33
inline constexpr ImU32 kCheck = rgba(255, 40, 40, 190);
inline constexpr ImU32 kWhitePiece = rgba(250, 248, 242);
inline constexpr ImU32 kWhitePieceEdge = rgba(15, 15, 15);
inline constexpr ImU32 kBlackPiece = rgba(10, 10, 10);
inline constexpr ImU32 kBlackPieceEdge = rgba(215, 215, 215);

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
