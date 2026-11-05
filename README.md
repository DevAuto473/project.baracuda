# Barracuda

**A cross-platform C++20 chess engine with a custom NNUE, an offline trainer, and a live ImGui dashboard.**

Barracuda runs fully offline with no LibTorch, ONNX, or Python. Bitboards, legal move generation, alpha-beta search, quantised NNUE inference with AVX2/SSE4.1/scalar kernels, and backpropagation are all written by hand in C++20. A 60 FPS Dear ImGui dashboard sits on top: you can play against the engine, watch self-play games, see the network's activations change, and track loss and an estimated Elo while it trains.

![Barracuda dashboard](docs/dashboard.png)

*Screenshot (software-rendered under Xvfb): the user plays White in Blitz 5+3 and the engine has answered 4...Nxe4. The network graph shows the activations of the network trained so far, and the console streams search depth, NPS and PV alongside the epoch losses. The Elo chart shows the tactical-suite estimate after each epoch.*

---

## Contents
1. [Multi-OS setup & build guide](#1-multi-os-setup--build-guide)
2. [Running Barracuda](#2-running-barracuda)
3. [Project file blueprint](#3-project-file-blueprint)
4. [Architecture](#4-architecture)
5. [Implementation milestones](#5-implementation-milestones)
6. [CMake options](#6-cmake-options)
7. [Data formats](#7-data-formats)
8. [Testing & verification](#8-testing--verification)
9. [Known limitations](#9-known-limitations)
10. [Play online: website on GitHub Pages](#10-play-online-website-on-github-pages)
11. [Strange style: teaching Barracuda a personality](#11-strange-style-teaching-barracuda-a-personality)

---

## 1. Multi-OS setup & build guide

The only things you install are a compiler, CMake ≥ 3.21, Git, and (on Linux) the windowing/GL **headers**. CMake **FetchContent** downloads and builds **GLFW 3.4** and **Dear ImGui 1.91.5** at configure time, so you don't need any system-wide GUI framework. To build without the GUI (engine, UCI and trainer only), pass `-DBARRACUDA_BUILD_GUI=OFF`; that build needs nothing beyond a compiler.

Minimum compilers: GCC 10, Clang 12, MSVC 19.30 (Visual Studio 2022).

### 1.1 Fedora (GCC/Clang, Wayland or X11)

```bash
sudo dnf install gcc-c++ clang cmake ninja-build git pkgconf-pkg-config \
    libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel libXext-devel \
    mesa-libGL-devel wayland-devel libxkbcommon-devel
```

```bash
git clone <your-fork-url> barracuda && cd barracuda
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bin/barracuda
```

GLFW is built with **both** the Wayland and X11 backends when their headers are present. On a Wayland session it uses Wayland natively; you can force X11/XWayland by launching with `XDG_SESSION_TYPE=x11 ./build/bin/barracuda` (GLFW follows `XDG_SESSION_TYPE` when both backends are compiled in). Use `CC=clang CXX=clang++ cmake ...` to build with Clang.

### 1.2 Ubuntu (GCC/Clang, X11)

```bash
sudo apt update
sudo apt install build-essential clang cmake ninja-build git pkg-config \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev \
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev
```

```bash
git clone <your-fork-url> barracuda && cd barracuda
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bin/barracuda
```

The Wayland packages are optional on Ubuntu. If they are missing, configuration detects that and builds an X11-only GLFW instead of failing. Ubuntu 22.04 ships CMake 3.22 and GCC 11, which are both new enough.

### 1.3 Windows 10/11, Visual Studio 2022 (MSVC)

1. Install **Visual Studio 2022** (Community is fine) with the **"Desktop development with C++"** workload. CMake and Ninja are included.
2. Install **Git for Windows**, either with `winget install Git.Git` or from git-scm.com.
3. Open **"x64 Native Tools Command Prompt for VS 2022"**:

```bat
git clone <your-fork-url> barracuda
cd barracuda
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
build\bin\Release\barracuda.exe
```

Alternatively, choose **File → Open → Folder…** in Visual Studio. It reads `CMakePresets.json`, so you can pick `msvc-release` and press F5. With a single-config generator such as Ninja (`cmake --preset release`), the binaries land in `build\bin\` with no `Release\` subfolder.

### 1.4 Windows, MSYS2 / MinGW-w64

From the **MSYS2 UCRT64** shell:

```bash
pacman -S --needed git mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

```bash
git clone <your-fork-url> barracuda && cd barracuda
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/barracuda.exe
```

The MinGW executables link `libgcc`, `libstdc++` and `winpthread` statically (`BARRACUDA_STATIC_RUNTIME=ON`), so they run outside the MSYS2 shell.

### 1.5 CMake presets (all platforms)

```bash
cmake --list-presets
cmake --preset release && cmake --build --preset release
cmake --preset headless && cmake --build --preset headless   # no GUI deps needed
```

### 1.6 Offline / air-gapped builds

Put the dependency sources in `third_party/glfw` and `third_party/imgui` (for example as git submodules at tags `3.4` and `v1.91.5`). CMake picks them up automatically. You can also point at existing checkouts with `-DFETCHCONTENT_SOURCE_DIR_GLFW=<dir> -DFETCHCONTENT_SOURCE_DIR_IMGUI=<dir>`.

---

## 2. Running Barracuda

### 2.1 Dashboard (`barracuda`)

| Panel | What it shows |
|---|---|
| **Board** (left) | **Play vs Barracuda**: drag or click to move, with legal-move hints, promotion picker, eval bar, chess clocks (1+0 up to 15+10, or fixed engine time), undo, swap sides and move list. **Observe Training**: the self-play game currently being generated, with an optional watch delay. |
| **NNUE Architecture** (top right) | A node-link graph of `inputs → feature transformer → L1 → L2 → output`. Node brightness is the ClippedReLU activation. Edge colour is the weight sign (cyan +, rose −), and edge brightness is \|w·a\| or \|w\|. Hover a node to highlight its connections. During training the graph refreshes about once a second from the live weights. |
| **Metrics** (bottom middle) | Barracuda's **Elo** (also shown in the top bar) and its history, search stats (depth, NPS, nodes, score, PV), training stats, the loss curve and per-epoch train/val loss. The **Training** tab holds the self-play, training and network-file controls; the **Elo Match** tab plays the network against the classical evaluator and reports Elo with a 95% margin; the **Benchmark** tab runs WAC/STS-style EPD suites. |
| **Console** (bottom right) | A live log of epochs, losses, search depth, NPS and PV, and benchmark results, with a filter and a command line (`help`). |

**Quick start loop** (all inside the GUI):
1. **Training → Self-play + train.** The worker plays games and writes `selfplay.txt`, then trains `barracuda.bnn` on them. It saves a checkpoint and runs a benchmark after every epoch.
2. Watch the games in **Observe Training** and the loss and Elo curves in **Metrics**.
3. Switch to **Play vs Barracuda**. In *Auto* mode the engine uses the NNUE once the network has seen at least 1M training samples; you can force *Classical* or *NNUE* in **Training → Evaluator**.

On startup the dashboard loads `barracuda.bnn` from the working directory if one exists. Useful flags: `--exec "<console command>"` (repeatable), `--no-vsync`, `--fps N`, and `--screenshot out.ppm --frames N` for headless smoke tests.

### 2.2 Command line / UCI (`barracuda-cli`)

```bash
./build/bin/barracuda-cli                       # UCI mode: plug into Cute Chess, Arena, BanksiaGUI, ...
./build/bin/barracuda-cli perft 6               # move generator check
./build/bin/barracuda-cli bench 12              # fixed-depth node count / speed
./build/bin/barracuda-cli selfplay --games 2000 --depth 7 --threads 8 --out selfplay.txt
./build/bin/barracuda-cli train --data selfplay.txt --epochs 15 --out barracuda.bnn
./build/bin/barracuda-cli suite --movetime 1000             # built-in WAC subset + Elo estimate
./build/bin/barracuda-cli match --games 100 --movetime 100   # NNUE vs classical: Elo +/- margin
./build/bin/barracuda-cli suite --file wac.epd --movetime 500 --net barracuda.bnn
```

The UCI options are `Hash`, `EvalFile` and `EvalMode` (Auto/Classical/NNUE). The non-standard commands `d`, `eval`, `perft N` and `bench N` are also accepted.

---

## 3. Project file blueprint

```
barracuda/
├── CMakeLists.txt                 # root: options, subdirectories, summary
├── CMakePresets.json              # release / debug / headless / msvc / native presets
├── cmake/
│   ├── BarracudaCompilerFlags.cmake   # -O3 / /O2, warnings, AVX2/SSE4.1 detection, LTO
│   ├── BarracudaDependencies.cmake    # FetchContent GLFW + Dear ImGui (X11/Wayland autodetect)
│   └── detect_avx2.cpp                # host CPU probe (try_run)
├── engine/                        # ── barracuda::core + barracuda::engine ──
│   ├── include/barracuda/engine/
│   │   ├── types.h         # Square/Piece/Move(16-bit)/Value
│   │   ├── bitboard.h      # uint64 bitboards, fancy-magic slider attacks
│   │   ├── zobrist.h       # hashing keys
│   │   ├── position.h      # bitboards + mailbox, make/unmake, SEE, repetition
│   │   ├── movegen.h       # legal generator (pins, checks, ep, castling, promotions), perft
│   │   ├── notation.h      # UCI / SAN, game result
│   │   ├── evaluate.h      # hand-crafted fallback eval (tapered PST, pawns, mobility)
│   │   ├── tt.h            # transposition table
│   │   ├── search.h        # iterative-deepening PVS + quiescence
│   │   ├── selfplay.h      # training-data generator
│   │   └── suite.h         # EPD suites (WAC subset built-in) + Elo estimate
│   └── src/*.cpp
├── nnue/                          # ── barracuda::nnue ──
│   ├── include/barracuda/nnue/
│   │   ├── architecture.h  # 768 → 2×256 → 32 → 32 → 1, quantisation constants
│   │   ├── network.h       # int16/int8 inference, Accumulator(+Stack), add/sub_feature
│   │   ├── float_network.h # float master weights, .bnn file format
│   │   ├── dataset.h       # FEN | cp | WDL parser, packed samples
│   │   ├── trainer.h       # multi-threaded backprop + Adam
│   │   └── inspector.h     # activation/weight snapshot for the visualiser
│   └── src/*.cpp
├── gui/                           # ── barracuda (dashboard executable) ──
│   └── src/
│       ├── main.cpp                # GLFW/OpenGL3/ImGui render loop (60 FPS cap)
│       ├── app.{h,cpp}             # layout, top bar, console commands
│       ├── theme.{h,cpp}           # dark minimalist palette + font discovery
│       ├── widgets.{h,cpp}         # vector chess pieces, charts, eval bar, tiles
│       ├── runtime/                # thread boundary - no ImGui in here
│       │   ├── concurrency.h       # lock-free TripleBuffer + MPMC ring queue
│       │   ├── shared_state.h      # snapshots, Logger, NetworkRegistry
│       │   ├── search_worker.*     # play/analysis search thread
│       │   └── training_worker.*   # self-play / dataset / training / benchmark thread
│       └── panels/                 # board, network, metrics, console
├── app/main.cpp                   # barracuda-cli (UCI + perft/bench/selfplay/train/suite)
├── tests/test_main.cpp            # perft, hashing, SAN, search, NNUE, trainer, lock-free tests
├── docs/                          # ARCHITECTURE.md, screenshot
└── .github/workflows/ci.yml       # Ubuntu GCC/Clang, Fedora, Windows MSVC + MinGW
```

Dependency direction: `core ← nnue ← engine ← {cli, gui}`. The GUI's `runtime/` layer never includes ImGui, and the panels never touch engine internals from the render thread.

---

## 4. Architecture

A deeper write-up is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). In brief:

**Engine core.** The board is held as 64-bit bitboards (6 piece-type sets plus 2 colour sets) with a mailbox alongside. Slider attacks use fancy magic bitboards, with the magics generated deterministically at startup. The move generator is fully legal: it detects pins via x-raying snipers, restricts check evasions to the checker and blocking squares, verifies en passant by simulating the capture (which catches the horizontal-pin case), and checks castling through attacked squares. It is validated with perft on 6 standard positions (including Kiwipete and positions 3 to 6).

**Search.** Iterative deepening with aspiration windows drives a negamax/PVS alpha-beta search. Leaves go to quiescence search with SEE and delta pruning. Move ordering is TT move → good captures (MVV-LVA) → queen promotions → two killer moves per ply → history heuristic. The transposition table is Zobrist-keyed with 2-way buckets (depth-preferred plus always-replace) and mate-distance-adjusted scores. Pruning and extensions: null move, reverse futility, futility, late move pruning/reductions, internal iterative reduction, check extension and mate-distance pruning. Time management handles movetime, clock plus increment, and node limits.

**NNUE.** The 768 binary inputs encode colour × piece × square relative to each perspective. The feature transformer uses int16 weights, and two int16 accumulators are updated incrementally from the 1–3 dirty pieces of each move (`add_feature`/`sub_feature`, fused into one SIMD pass). Activation is ClippedReLU `min(max(0,x),127)` producing uint8, followed by int8 dense layers using `_mm256_maddubs_epi16` → `_mm256_madd_epi16`, with >>6 requantisation and a centipawn output. Training happens in float space with exactly the same clipping semantics, so quantisation is just rounding. The test suite measures float-vs-int drift.

**Trainer.** The target is `λ·σ(cp/400) + (1−λ)·WDL`, optimised as MSE in win-probability space with Adam and per-epoch learning-rate decay. Mini-batches are split across threads, and each thread keeps sparse FT gradient rows. Weights are clamped to their quantisable ranges, and a checkpoint is written after every epoch.

**Threads.**
```
 render thread (60 FPS)            search worker                    training worker (+ trainer helper threads)
 ─────────────────────             ─────────────                    ──────────────────────────────────────────
 ImGui panels  ──commands──▶  Go / Inspect / NewGame         SelfPlay / Train / Benchmark / Save / Load
     ▲  (mutex+cv queue, push only)         │                               │
     │                                      ▼                               ▼
     └── TripleBuffer::update() ◀── SearchSnapshot, MoveResult,   TrainingSnapshot, ObservationSnapshot,
         (lock-free, never blocks)       NetworkView                 NetworkView
     └── MpmcQueue::try_pop()   ◀────────── log lines from every thread (lock-free) ──────────┘
                    NetworkRegistry: shared_ptr<const Network> swap between workers (never on the render path)
```

---

## 5. Implementation milestones

| # | Milestone | Contents | Verification |
|---|---|---|---|
| M1 | **Engine core** | types, bitboards, magics, Zobrist, position, legal movegen, SAN/UCI, classical eval, TT, search, UCI CLI | perft suite (6 positions) passes; mate/tactics tests; built-in WAC subset |
| M2 | **NNUE** | quantised network + SIMD kernels, accumulator stack, float master + file format, dataset parser, multi-threaded trainer, self-play generator | incremental == refresh over 20 random games; float-vs-int drift < 25 cp; trainer loss drops > 30 % on a toy set |
| M3 | **GUI & visualiser** | GLFW/ImGui shell, theme, vector pieces, board (play/observe), network graph, metrics/charts, console | rendered under Xvfb (`--screenshot`) |
| M4 | **Integration** | worker threads, lock-free snapshots, NetworkRegistry, live training view, per-epoch benchmark/Elo, CI | ThreadSanitizer run of the full GUI doing self-play + training + play search at once: 0 reports |

All four milestones are implemented in this repository.

---

## 6. CMake options

| Option | Default | Meaning |
|---|---|---|
| `BARRACUDA_BUILD_GUI` | `ON` | Build the dashboard (fetches GLFW + ImGui) |
| `BARRACUDA_BUILD_CLI` | `ON` | Build `barracuda-cli` |
| `BARRACUDA_BUILD_TESTS` | `ON` | Build `barracuda_tests` (CTest) |
| `BARRACUDA_SIMD` | `AUTO` | `AUTO` probes the build host via cpuid (`AVX2` → `-mavx2 -mbmi2 -mfma` / `/arch:AVX2`, else `SSE41`); also `AVX2`, `SSE41` or `NONE` (scalar) |
| `BARRACUDA_NATIVE` | `OFF` | `-march=native` (GCC/Clang) |
| `BARRACUDA_LTO` | `ON` | IPO/LTO in Release when supported |
| `BARRACUDA_STATIC_RUNTIME` | `ON` | MinGW: static libgcc/libstdc++/winpthread |
| `BARRACUDA_WINDOWED` | `OFF` | Windows: GUI subsystem (no console window) |

Release flags are `-O3 -funroll-loops -fno-math-errno` on GCC/Clang and `/O2 /Oi /Ot /Gy /GS-` on MSVC. **Note:** `AUTO` tunes for the machine that builds the binary. If you want to distribute binaries to older CPUs, use `-DBARRACUDA_SIMD=SSE41` or `NONE`.

---

## 7. Data formats

**Training data** has one position per line. The score is in centipawns from **White's** point of view, and the result is from White's point of view:
```
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | 35 | 0.5
8/8/4k3/8/8/4K3/4P3/8 w - - 0 1 [1.0] 250          # "legacy" layout is accepted too
```
Results may be written as `1.0/0.5/0.0`, `1-0/1/2-1/2/0-1` or `w/d/b`.

**Network files** (`.bnn`) contain `"BRCDNNUE"`, a version, the dimensions, epoch and sample counters, then the float32 tensors. The file stores float master weights, which are quantised on load, so a checkpoint can be both played with and trained further.

**EPD suites** use standard EPD with `bm`/`am`/`id` (SAN moves). Entries with an illegal FEN or move are skipped and counted.

---

## 8. Testing & verification

```bash
ctest --test-dir build --output-on-failure      # or ./build/bin/barracuda_tests [filter]
./build/bin/barracuda-cli perft 6               # startpos: 119060324
```

Checks run while developing this repository (Ubuntu 24.04, GCC 13.3 and Clang 18):
- `barracuda_tests` passes with the AVX2, SSE4.1 and scalar kernels, and under ASan+UBSan.
- The lock-free primitives and the multi-threaded trainer are clean under TSan, and so is the full dashboard running self-play, training and a play search at the same time.
- The GCC and Clang builds produce zero warnings with `-Wall -Wextra -Wpedantic`.
- On the built-in 28-position WAC subset at 1 s/move, the classical evaluator solves 26 of 28.
- End to end: 3,000 self-play games at depth 7 produced 320K positions, and 12 epochs of training on them took validation loss from 0.0045 to 0.0034 at about 250K positions/s on 4 threads. The resulting NNUE searches at about 0.94M nps on the bench positions and solves as many suite positions as the classical eval (25 of 28 each at 500 ms).

**Not verified:** Windows (MSVC and MinGW) and Fedora builds are covered by the CI workflow (`.github/workflows/ci.yml`) but have not been run yet. Please check the first CI run.

---

## 9. Known limitations

- **How the Elo is computed.** The *Elo Match* plays the network against Barracuda's own classical evaluator in colour-swapped games, which is a real measurement with a 95% margin (it needs 100+ games to get under about ±60). The headline number adds that to an anchor for the classical evaluator taken from the 28-position tactical suite, and that anchor is only a heuristic. Treat the result as a trend indicator rather than an official rating; for that, play matches against other engines with `barracuda-cli` in Cute Chess or fastchess.
- **No network weights are shipped.** You bootstrap one with self-play from the classical eval. A network trained on only a few thousand positions is much weaker than the classical eval, which is why *Auto* mode waits until the net has seen 1M samples.
- The search is single-threaded (no Lazy SMP yet), and the engine supports standard chess only (no Chess960).
- The GUI uses OpenGL 3.3 core. Very old GPU drivers, or VMs without 3D acceleration, may need Mesa's software renderer (`LIBGL_ALWAYS_SOFTWARE=1`).

---

## 10. Play online: website on GitHub Pages

`web/` is a static site where anyone can play against Barracuda. The engine is compiled to **WebAssembly** and runs inside the visitor's browser (in a Web Worker), so there is no server to run or pay for. The site has the same black and purple look as the dashboard, a drag-and-click board, an evaluation bar, undo, and a panel showing the published network (epochs trained, Elo, update date).

**Publishing the site (one time)**
1. Merge your branch into `main` (the deploy workflow in `.github/workflows/pages.yml` runs on `main`).
2. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**. GitHub Pages on a *private* repository needs a paid GitHub plan; if the repository is free and private, make it public.
3. Push to `main`. The workflow builds the WebAssembly module, runs a smoke test, and deploys. The site appears at `https://<your-user>.github.io/<repo>/`.

**Updating the site after more training**

From the dashboard: **Metrics → Training → 3. Publish to the website → Publish to website**. It saves the network, runs the match below, writes `web/net/`, then commits and pushes that folder (uncheck *commit and push* to only write the files). Start the dashboard from inside your clone (`cd ~/barracuda && ./build/bin/barracuda`) so it finds the repository, and log in to GitHub once (`gh auth login`, or an SSH key) so the push works. The console command `publish 40` does the same.

From the command line:
```bash
./build/bin/barracuda-cli publish --net barracuda.bnn --games 100 --push
```
(without `--push`, commit `web/net` yourself).
`publish` plays the network against the classical evaluator, then writes `web/net/manifest.json` (epochs, samples, Elo with its margin, date) and copies the network next to it. **The site only switches to the network when it beats the classical evaluator in that match**; otherwise the manifest keeps the site on the classical evaluator and says so. `--force-nnue` overrides that, and `--games 0` skips the match. The network file is always published, because the page draws it in the **Neural network** panel: the real activations and weights, refreshed every time the search finishes a depth (the network evaluates the end of the line being analysed), with pulses travelling along the strongest connections while the engine thinks. When the network is not playing yet, the panel says so.

**Always full strength, in its own strange style.** There are no levels: the site always searches at maximum (up to 5 s per move, usually 2 to 3 s), and reads every reply the opponent can make before moving. On top of that it plays in Barracuda's **strange style** (see section 11): when an odd-looking move (an early rook or king walk, a flank pawn push, a piece retreat, a quiet move instead of the obvious capture) is proven by a full-depth search to be almost as good as the best move, it plays the odd one. Such moves are marked *Strange move!* under the board. The concession is tiny while the game is balanced and only grows when Barracuda is clearly winning, and it is never used when Barracuda is worse or when a mate is on the board.

**Building and testing locally**
```bash
web/build.sh                                  # needs Emscripten (emcc) on PATH
node web/test/wasm_test.mjs barracuda.bnn     # engine checks under Node
python3 -m http.server -d web 8000            # then open http://localhost:8000
```
The WebAssembly build uses the same SSE4.1 NNUE kernels (mapped to WASM SIMD), and its network evaluations match the native build exactly. Visitors' games are not collected: training only happens on your machine.

**One screen, any screen.** On a desktop-sized window the whole page fits without scrolling (board, controls, the neural network and the move list). On phones, tablets and narrow or short windows the board and status line stay on screen and three tabs under it (Game, Network, Moves) fill the rest; on a phone held sideways the board sits next to the tabs. Nothing overflows horizontally, the page itself never scrolls, and a tab scrolls on its own only when a very small screen needs it. Tested from 320 x 568 to 2560 x 1300, in both languages. Adding `?fen=<FEN>` to the address starts a game from that position.

**Pace.** While Barracuda thinks, the network drawing animates. When Barracuda picks its move, the moving piece assembles from particles drawn out of the network's connections.

**Learning from visitors' games (optional).** When a game ends, the page can send an anonymous record of it (moves, result, a random id: no name, no IP stored by the site) to a collector you set up in `web/config.json`. Visitors can switch this off with the checkbox in the footer. If Barracuda wins and the game was really sent, the board turns black and says *"Thank you! Barracuda has been taught from the way you think, and your mind has become part of its neural network."* (Arabic: *شكرًا لك! لقد تم تعليم باراكودا على طريقة تفكيرك، وأصبح عقلك ضمن شبكته العصبية.*); when nothing was sent (no collector configured, sharing off, offline) it only says thanks for playing, so the page never claims something that did not happen.

Nothing changes on the site by itself: the games become training data when you run the steps below.

1. Create a Google Form with a single **Paragraph** question. Open the form's pre-filled link to find the field name (`entry.123456789`), and take the form address ending in `/formResponse`. Put them in `web/config.json`:
   ```json
   { "collect": { "type": "googleform",
                  "formUrl": "https://docs.google.com/forms/d/e/<FORM_ID>/formResponse",
                  "entry": "entry.123456789" } }
   ```
   (any endpoint that accepts a plain-text POST also works: `{ "collect": { "type": "http", "url": "https://..." } }`). Commit and push.
2. Download the responses (Form → Responses → Sheets → File → Download → CSV).
3. Turn them into training positions, mix them with your self-play data, train and publish:
```bash
./build/bin/barracuda-cli import-games --in responses.csv --out human.txt --depth 6
cat selfplay.txt human.txt > all.txt
./build/bin/barracuda-cli train --data all.txt --net barracuda.bnn --epochs 10
./build/bin/barracuda-cli publish --net barracuda.bnn --games 100
```
`import-games` replays every game, skips the opening, and labels each quiet position with the engine's evaluation and the game's result. It accepts the CSV export or JSON lines and ignores corrupt records.

**Human games without visitors: the Lichess database.** `import-pgn` reads any PGN, including the free [Lichess database](https://database.lichess.org) (monthly `.pgn.zst` files, millions of rated games), and turns it into training data. It reads as a stream and stops after the first N games, so a huge file is never loaded:
```bash
sudo apt install zstd
zstdcat lichess_db_standard_rated_2024-01.pgn.zst | ./build/bin/barracuda-cli import-pgn --in - --games 50000 --min-elo 1200 --max-elo 1800 --out human.txt
```
(or `--in games.pgn` for a plain file). To keep the data small and the training fast, without hurting what the network learns:

| Filter | Default | Why |
|---|---|---|
| `--min-elo` / `--max-elo` | none | both players inside the window (1200 to 1800 is where most opponents are) |
| `--skip-plies` | 10 | opening moves are memorised book, not understanding |
| quiet positions only | always | no checks and no captures: the score of a position in the middle of an exchange is noise |
| `--every` | 3 | one position in three: neighbouring positions are nearly identical |
| duplicates | removed | the same position is stored once |
| `--min-plies` | 20 | drops aborted games |
| scores | clamped to +-25 pawns | larger scores add nothing and slow training |

50,000 games give about 0.7 to 1 million positions (roughly 60 MB); raise `--every` to 5 for about half of that. Scores come from the PGN's own engine evaluation (`[%eval]`) when a game has one (free, and deeper than our search), and from a depth-5 search (`--depth`) on all cores (`--threads`) otherwise, about 2 ms per position: a few minutes for 50,000 games. Reading is fast (20,000 games in 3 seconds). The size of this file has no effect on how fast Barracuda plays: only the network's shape (768-256-32-32-1) does, and training never changes it. Then, as usual:
```bash
cat base_*.txt style_*.txt human.txt > all.txt
./build/bin/barracuda-cli train --data all.txt --net barracuda.bnn --out barracuda.bnn --epochs 4 --lambda 0.6
```

**Right-click and text selection are disabled** on the website (`contextmenu`/`selectstart` handlers at the top of `web/js/app.js`, and the `user-select: none` rule at the end of `web/css/style.css`). This only stops casual copying; the page source stays readable in the browser. Delete those two pieces to bring both back.

**Stylesheet check.** `node web/test/css_check.mjs` fails on an unbalanced brace in `web/css/style.css` (the deploy workflow runs it): browsers silently drop the rules after such a mistake.

**Languages and fonts.** The page is available in English and Arabic (button in the header, remembered per browser, and picked automatically from the browser language on first visit). Arabic switches the layout to right-to-left while the chess board itself is never mirrored. English text is set in Roca Two Bold and Arabic in IBM Plex Sans Arabic, both self-hosted in `web/fonts/` (see `web/fonts/NOTICE.txt` for licences; confirm that your Roca Two licence allows web embedding before publishing).

## 11. Strange style: teaching Barracuda a personality

Elo is not the goal here: the goal is an engine whose moves look wrong to people and still work.

**How the style works.** After the normal search, Barracuda ranks the other legal moves by how unusual they look (rook lifts and king walks in the middlegame, flank pawns, retreats, quiet moves where a capture is expected, moving the same piece again, early queen trips). It then proves, with null-window searches at the same depth as the main search, that the strange move keeps at least `best score - margin`. The first one that passes is played. The margin is 0 when Barracuda is worse, about 0.1 pawn in a balanced game, and up to 1.1 pawns when it is clearly winning. The proofs use the time the main search leaves unused (it stops instead of starting an iteration it could not finish), so the main search is never shortened. Mates are never replaced.

Switch it on or off: dashboard **Metrics → Training → Strange playing style** (or console `style on` / `style off`), UCI option `Style`, website always on.

Measure it:
```bash
./build/bin/barracuda-cli style-test --games 40 --movetime 300
```
It plays the strange style against normal play and prints how many moves were unusual and what that costs in Elo. `--margin 200` doubles the allowed concession (stranger, weaker), `--margin 50` halves it; `--verify 20` holds back 20% of the move time for the proofs. In our tests at 150 ms/move the default setting played about 5% unusual moves at no measurable cost (+35 +/- 115 Elo over 20 games), while holding back 30% of the time cost about 50 Elo and wider margins cost more.

**Training the style into the network.** The search picks the odd moves; training makes the network *like* the positions they lead to, so over time the engine plays them on its own, even with the style off:

1. **Self-play in the strange style** (dashboard: *1. Self-play*, keep *in its strange style* checked; or `barracuda-cli selfplay --games 2000 --style --out selfplay.txt`). The games are full of positions that only an engine with this taste reaches.
2. **Add people's games** (section 10, `import-games`): the network learns what humans play against it, so its surprises are tuned to people rather than to engines.
3. **Train with more weight on game results than on evaluations** (dashboard: lower *lambda* to about 0.3 to 0.5; CLI `train --lambda 0.4`). Results reward strange moves that actually win; evaluations only teach the classical taste.
4. **Repeat**: train, self-play again with the new network, train again. Each round is a little more "Barracuda".
5. **Publish** with the button. Use *force the network* only if you prefer its style even when it does not beat the classical evaluator.

**Reading the opponent and finishing fast.** Three more parts of the style, all on for the website and for play in the dashboard:

- **Prediction.** Before each of the player's moves Barracuda predicts two replies: the best one (from its own search) and the *natural* one, the move a person tends to play (a 2-ply search that sees captures and threats but no deep tactics). On the website the *Read you* chip counts how many of the player's moves it predicted, and a note under the board says *Barracuda saw Nf3 coming* or *Barracuda expected d4*.
- **Traps.** A trap is a move nearly as good as the best one (at most 0.4 pawn worse, proven at full depth) after which the natural reply loses at least 1.5 pawns. How much it may risk for a trap follows an opponent model: the website remembers, in the player's browser, how often this player picked the natural move when it was not the best one, and sets traps more eagerly against players who fall for them. When a trap works, the note says *Trap!*.
- **No draws, hunt the king.** Draws are worth -0.4 pawn to Barracuda (contempt), so it avoids repetitions and keeps playing for the win; once clearly winning it adds a bonus for driving the losing king to the edge and bringing its own king closer, which points the search at a mate rather than at more material.

- **Attack hunger.** From the opening on, Barracuda values its own pieces bearing on the squares around the enemy king (queens and rooks more, several attackers together far more). Only its own pressure is rewarded, so it is optimistic about attacks and keeps building them.
- **Attack and defence modes.** When its last search says the game has turned against it (worse than -0.6 pawn), it switches to **defence**: the attack bonus is replaced by a penalty for the opponent's pressure on its own king, draws become welcome, strange moves stop, and only risk-free counter-traps remain. Once the danger has passed (better than -0.1 pawn) it switches back to **attack**. On the website the *Mode* chip shows the current mode in red or blue, and a note announces each switch.

In 62 games against the same engine without any of this (150 ms/move), the full style (traps, contempt, hunt, attack hunger) scored +22 =19 -21: the personality costs no measurable strength.

`barracuda-cli selfplay --style` (and the dashboard's self-play *in its strange style*) plays with the full personality except traps (they need a human opponent), so training on that data teaches the network to like the positions this style leads to.

Measure it against a human-like opponent: `barracuda-cli trap-test --games 16 --movetime 100` (`--trap 0..100`, `--margin`, `--hunt 0|1`). In our runs Barracuda won every game; the opponent fell into about 3 of every 4 traps; games lasted 26 to 30 moves with every combination of settings, so the differences between them were within noise. That opponent loses quickly whatever happens: the traps are aimed at people, who avoid simple blunders but miss the move behind the obvious one.

No engine can promise to beat every human in every game. What it does: it searches every reply before moving, never plays a strange move that a full-depth search does not justify, and never gives anything away when it is worse.
