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
    s.ChildBorderSize = 1;
    s.FrameBorderSize = 0;
    s.PopupBorderSize = 1;
    s.WindowRounding = 0;
    s.ChildRounding = 8;
    s.FrameRounding = 5;
    s.PopupRounding = 6;
    s.ScrollbarRounding = 6;
    s.GrabRounding = 4;
    s.TabRounding = 5;
    s.SeparatorTextBorderSize = 1;

    using namespace palette;
    ImVec4* c = s.Colors;
    c[ImGuiCol_Text] = to_vec4(kText);
    c[ImGuiCol_TextDisabled] = to_vec4(kTextDim);
    c[ImGuiCol_WindowBg] = to_vec4(kBackground);
    c[ImGuiCol_ChildBg] = to_vec4(kPanel);
    c[ImGuiCol_PopupBg] = to_vec4(rgba(24, 27, 34, 250));
    c[ImGuiCol_Border] = to_vec4(kBorder);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = to_vec4(rgba(32, 36, 45));
    c[ImGuiCol_FrameBgHovered] = to_vec4(rgba(40, 45, 56));
    c[ImGuiCol_FrameBgActive] = to_vec4(rgba(48, 54, 67));
    c[ImGuiCol_TitleBg] = to_vec4(kPanel);
    c[ImGuiCol_TitleBgActive] = to_vec4(kPanelAlt);
    c[ImGuiCol_MenuBarBg] = to_vec4(kPanel);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = to_vec4(rgba(55, 62, 75));
    c[ImGuiCol_ScrollbarGrabHovered] = to_vec4(rgba(70, 78, 94));
    c[ImGuiCol_ScrollbarGrabActive] = to_vec4(rgba(85, 94, 112));
    c[ImGuiCol_CheckMark] = to_vec4(kAccent);
    c[ImGuiCol_SliderGrab] = to_vec4(kAccent);
    c[ImGuiCol_SliderGrabActive] = to_vec4(rgba(125, 211, 252));
    c[ImGuiCol_Button] = to_vec4(rgba(36, 41, 51));
    c[ImGuiCol_ButtonHovered] = to_vec4(rgba(48, 56, 70));
    c[ImGuiCol_ButtonActive] = to_vec4(rgba(14, 116, 144));
    c[ImGuiCol_Header] = to_vec4(rgba(36, 41, 51));
    c[ImGuiCol_HeaderHovered] = to_vec4(rgba(46, 53, 66));
    c[ImGuiCol_HeaderActive] = to_vec4(rgba(14, 116, 144));
    c[ImGuiCol_Separator] = to_vec4(kBorder);
    c[ImGuiCol_Tab] = to_vec4(kPanel);
    c[ImGuiCol_TabHovered] = to_vec4(rgba(46, 53, 66));
    c[ImGuiCol_TabSelected] = to_vec4(rgba(36, 41, 51));
    c[ImGuiCol_TabSelectedOverline] = to_vec4(kAccent);
    c[ImGuiCol_TabDimmed] = to_vec4(kPanel);
    c[ImGuiCol_TabDimmedSelected] = to_vec4(rgba(30, 34, 42));
    c[ImGuiCol_PlotLines] = to_vec4(kAccent);
    c[ImGuiCol_PlotHistogram] = to_vec4(kAccent);
    c[ImGuiCol_TableHeaderBg] = to_vec4(kPanelAlt);
    c[ImGuiCol_TableBorderStrong] = to_vec4(kBorder);
    c[ImGuiCol_TableBorderLight] = to_vec4(kGrid);
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = to_vec4(rgba(255, 255, 255, 6));
    c[ImGuiCol_TextSelectedBg] = to_vec4(rgba(56, 189, 248, 70));
    c[ImGuiCol_NavCursor] = to_vec4(kAccent);
    c[ImGuiCol_ModalWindowDimBg] = to_vec4(rgba(0, 0, 0, 140));

    s.ScaleAllSizes(dpi);
    return fonts;
}

} // namespace barracuda::gui
