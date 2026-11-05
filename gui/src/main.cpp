// Barracuda dashboard - render thread entry point (GLFW + OpenGL 3 + Dear ImGui).
//
// The render loop only draws: engine search, self-play, dataset parsing and NNUE training run
// on worker threads and hand data over through lock-free snapshots (see runtime/).
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "app.h"
#include "theme.h"

namespace {

void glfw_error(int code, const char* msg) { std::fprintf(stderr, "GLFW error %d: %s\n", code, msg); }

struct Options {
    int target_fps = 60;
    bool vsync = true;
    bool software_cursor = false; // draw the mouse cursor inside the window (auto-on under WSL)
    const char* screenshot = nullptr; // --screenshot <file.ppm> : render N frames, dump, exit (CI smoke test)
    int screenshot_frames = 90;
    std::vector<const char*> exec; // --exec "<console command>" (repeatable), run at startup
};

Options parse(int argc, char** argv) {
    Options o;
    bool explicit_cursor = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-vsync")) o.vsync = false;
        else if (!std::strcmp(argv[i], "--software-cursor")) o.software_cursor = true;
        else if (!std::strcmp(argv[i], "--system-cursor")) o.software_cursor = false, explicit_cursor = true;
        else if (!std::strcmp(argv[i], "--fps") && i + 1 < argc) o.target_fps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--screenshot") && i + 1 < argc) o.screenshot = argv[++i];
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) o.screenshot_frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--exec") && i + 1 < argc) o.exec.push_back(argv[++i]);
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf("barracuda [--software-cursor|--system-cursor] [--no-vsync] [--fps N] [--exec \"<console command>\"]... [--screenshot out.ppm --frames N]\n");
            std::exit(0);
        }
    }
    // WSLg often has no usable X cursor theme, which makes the OS cursor invisible over the window
    if (!explicit_cursor && (std::getenv("WSL_DISTRO_NAME") || std::getenv("WSL_INTEROP"))) o.software_cursor = true;
    if (o.target_fps < 10) o.target_fps = 10;
    return o;
}

void save_ppm(const char* path, int w, int h) {
    std::vector<unsigned char> px(std::size_t(w) * std::size_t(h) * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    if (FILE* f = std::fopen(path, "wb")) {
        std::fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (int y = h - 1; y >= 0; --y) std::fwrite(&px[std::size_t(y) * std::size_t(w) * 3], 1, std::size_t(w) * 3, f);
        std::fclose(f);
        std::printf("screenshot written to %s (%dx%d)\n", path, w, h);
    }
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);

    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fprintf(stderr, "failed to initialise GLFW (no display?)\n");
        return 1;
    }

    // OpenGL 3.3 core (works on Mesa, NVIDIA, AMD and Intel on Linux + Windows)
    const char* glsl_version = "#version 330 core";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1600, 940, "Barracuda - Chess Engine & NNUE Dashboard", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "failed to create an OpenGL 3.3 window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(opt.vsync ? 1 : 0);

    float xscale = 1.0f, yscale = 1.0f;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    const float dpi = std::max(1.0f, std::min(xscale, 3.0f));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr; // fixed layout, nothing to persist
    if (opt.software_cursor) io.MouseDrawCursor = true; // ImGui draws the cursor and hides the OS one

    const barracuda::gui::Fonts fonts = barracuda::gui::setup_theme(dpi);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    {
        barracuda::gui::App app(fonts);
        for (const char* cmd : opt.exec) app.execute_command(cmd);
        const auto frame_budget = std::chrono::microseconds(1'000'000 / opt.target_fps);
        int frame = 0;

        while (!glfwWindowShouldClose(window)) {
            const auto frame_start = std::chrono::steady_clock::now();
            glfwPollEvents();
            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
                glfwWaitEventsTimeout(0.1); // minimised: workers keep running, we stop drawing
                continue;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            app.frame();
            ImGui::Render();

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);
            const ImVec4 bg = barracuda::gui::palette::to_vec4(barracuda::gui::palette::kBackground);
            glClearColor(bg.x, bg.y, bg.z, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            if (opt.screenshot && ++frame >= opt.screenshot_frames) {
                save_ppm(opt.screenshot, w, h);
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            glfwSwapBuffers(window);

            // 60 FPS cap even when the driver ignores vsync (some Wayland/VM setups)
            const auto spent = std::chrono::steady_clock::now() - frame_start;
            if (spent < frame_budget) std::this_thread::sleep_for(frame_budget - spent);
        }
    } // App destructor joins the worker threads

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
