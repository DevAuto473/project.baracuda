#include "theme.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace barracuda::gui {

namespace {

std::string first_existing(const std::vector<std::string>& candidates) {
    std::error_code ec;
    for (const auto& c : candidates)
        if (std::filesystem::exists(c, ec)) return c;
    return {};
}

std::string windows_font(const char* file) {
    const char* windir = std::getenv("WINDIR");
    return std::string(windir ? windir : "C:\\Windows") + "\\Fonts\\" + file;
}

ImFont* load_font(ImGuiIO& io, const std::string& path, float size) {
    if (!path.empty()) {
        ImFontConfig cfg;
        cfg.OversampleH = 2;
        cfg.OversampleV = 2;
        if (ImFont* f = io.Fonts->AddFontFromFileTTF(path.c_str(), size, &cfg)) return f;
    }
    ImFontConfig cfg;
    cfg.SizePixels = size;
    return io.Fonts->AddFontDefault(&cfg);
}

} // namespace

Fonts setup_theme(float dpi) {
    ImGuiIO& io = ImGui::GetIO();
    Fonts fonts;

    const std::string sans = first_existing({
        // Linux (Ubuntu / Debian)
        "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        // Fedora
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        // Windows
        windows_font("segoeui.ttf"),
        windows_font("arial.ttf"),
    });
    const std::string mono = first_existing({
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Regular.otf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        windows_font("CascadiaMono.ttf"),
        windows_font("consola.ttf"),
    });

    const float base = 15.0f * dpi;
    fonts.regular = load_font(io, sans, base);
    fonts.small = load_font(io, sans, 12.5f * dpi);
    fonts.mono = load_font(io, mono, 13.5f * dpi);
    fonts.large = load_font(io, sans, 30.0f * dpi);
    fonts.title = load_font(io, sans, 18.0f * dpi);
    io.FontDefault = fonts.regular;

    ImGuiStyle& s = ImGui::GetStyle();
    s = ImGuiStyle();
    s.WindowPadding = ImVec2(10, 10);
    s.FramePadding = ImVec2(8, 5);
    s.ItemSpacing = ImVec2(8, 6);
    s.ItemInnerSpacing = ImVec2(6, 4);
    s.CellPadding = ImVec2(6, 3);
    s.ScrollbarSize = 10;
    s.GrabMinSize = 8;
    s.WindowBorderSize = 0;
    s.ChildBorderSize = 1.5f;
    s.FrameBorderSize = 0;
    s.PopupBorderSize = 1.5f;
    s.WindowRounding = 0;
    s.ChildRounding = 0;
    s.FrameRounding = 0;
    s.PopupRounding = 0;
    s.ScrollbarRounding = 0;
    s.GrabRounding = 0;
    s.TabRounding = 0;
    s.SeparatorTextBorderSize = 1;

    using namespace palette;
    ImVec4* c = s.Colors;
    auto rgb = [](int v) { return to_vec4(rgba(v, v, v)); };
    c[ImGuiCol_Text] = to_vec4(kText);                       // #E0E0E0
    c[ImGuiCol_TextDisabled] = to_vec4(kTextDim);            // #6E7681
    c[ImGuiCol_WindowBg] = to_vec4(kBackground);             // #000000
    c[ImGuiCol_ChildBg] = to_vec4(kPanel);                   // #050505
    c[ImGuiCol_PopupBg] = to_vec4(rgba(8, 8, 8));            // #080808
    c[ImGuiCol_Border] = to_vec4(kBorder);                   // purple
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = rgb(13);                           // #0D0D0D
    c[ImGuiCol_FrameBgHovered] = rgb(26);                    // #1A1A1A
    c[ImGuiCol_FrameBgActive] = rgb(38);                     // #262626
    c[ImGuiCol_TitleBg] = to_vec4(kBackground);
    c[ImGuiCol_TitleBgActive] = to_vec4(kBackground);
    c[ImGuiCol_TitleBgCollapsed] = to_vec4(kBackground);
    c[ImGuiCol_MenuBarBg] = to_vec4(kBackground);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = rgb(48);
    c[ImGuiCol_ScrollbarGrabHovered] = rgb(72);
    c[ImGuiCol_ScrollbarGrabActive] = rgb(96);
    c[ImGuiCol_CheckMark] = to_vec4(kAccent);
    c[ImGuiCol_SliderGrab] = to_vec4(kAccentBlue);
    c[ImGuiCol_SliderGrabActive] = to_vec4(kAccent);
    c[ImGuiCol_Button] = rgb(20);
    c[ImGuiCol_ButtonHovered] = rgb(34);
    c[ImGuiCol_ButtonActive] = to_vec4(kAccentBlue);
    c[ImGuiCol_Header] = rgb(20);                            // #141414
    c[ImGuiCol_HeaderHovered] = rgb(30);                     // #1E1E1E
    c[ImGuiCol_HeaderActive] = to_vec4(kAccentBlue);
    c[ImGuiCol_Separator] = to_vec4(kBorder);
    c[ImGuiCol_SeparatorHovered] = to_vec4(kAccentBlue);
    c[ImGuiCol_SeparatorActive] = to_vec4(kAccent);
    c[ImGuiCol_Tab] = rgb(13);                               // inactive #0D0D0D
    c[ImGuiCol_TabHovered] = rgb(30);
    c[ImGuiCol_TabSelected] = to_vec4(kBackground);          // active #000000
    c[ImGuiCol_TabSelectedOverline] = to_vec4(kBorder);  // top border highlight
    c[ImGuiCol_TabDimmed] = rgb(13);
    c[ImGuiCol_TabDimmedSelected] = to_vec4(kBackground);
    c[ImGuiCol_TabDimmedSelectedOverline] = to_vec4(kBorder);
    c[ImGuiCol_PlotLines] = to_vec4(kAccent);
    c[ImGuiCol_PlotHistogram] = to_vec4(kAccent);
    c[ImGuiCol_TableHeaderBg] = rgb(13);
    c[ImGuiCol_TableBorderStrong] = to_vec4(kBorder);
    c[ImGuiCol_TableBorderLight] = rgb(22);
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = to_vec4(rgba(255, 255, 255, 6));
    c[ImGuiCol_TextSelectedBg] = to_vec4(rgba(0, 122, 204, 110));
    c[ImGuiCol_NavCursor] = to_vec4(kAccentBlue);
    c[ImGuiCol_ModalWindowDimBg] = to_vec4(rgba(0, 0, 0, 170));

    s.ScaleAllSizes(dpi);
    return fonts;
}

} // namespace barracuda::gui
