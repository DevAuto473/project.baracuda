// barracuda-cli: UCI engine + headless tools (perft, bench, self-play, training, tactical suites).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "barracuda/engine/match.h"
#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"
#include "barracuda/engine/search.h"
#include "barracuda/engine/selfplay.h"
#include "barracuda/engine/suite.h"
#include "barracuda/nnue/dataset.h"
#include "barracuda/nnue/network.h"
#include "barracuda/nnue/trainer.h"

using namespace barracuda;

namespace {

constexpr const char* kVersion = "Barracuda 1.0";
std::atomic<bool> g_stop{false};

// --------------------------------------------------------------------------- args
struct Args {
    std::vector<std::string> positional;
    std::map<std::string, std::string> opts;
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = opts.find(k);
        return it == opts.end() ? def : it->second;
    }
    long long num(const std::string& k, long long def) const {
        auto it = opts.find(k);
        return it == opts.end() ? def : std::stoll(it->second);
    }
    double real(const std::string& k, double def) const {
        auto it = opts.find(k);
        return it == opts.end() ? def : std::stod(it->second);
    }
    bool has(const std::string& k) const { return opts.count(k) != 0; }
};

Args parse_args(int argc, char** argv, int first) {
    Args a;
    for (int i = first; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            const std::string key = s.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) a.opts[key] = argv[++i];
            else a.opts[key] = "1";
        } else {
            a.positional.push_back(s);
        }
    }
    return a;
}

std::shared_ptr<const nnue::Network> try_load_net(const std::string& path, bool verbose) {
    std::string err;
    auto net = nnue::load_network(path, &err);
    if (!net) {
        if (verbose) std::cerr << "info string cannot load network: " << err << "\n";
        return nullptr;
    }
    if (verbose)
        std::cerr << "info string loaded network " << path << " (epochs " << net->epochs_trained() << ", "
                  << nnue::simd_backend() << ")\n";
    return net;
}

std::shared_ptr<const nnue::Network> autoload_net(const char* argv0) {
    std::vector<std::filesystem::path> candidates = {"barracuda.bnn"};
    std::error_code ec;
    const auto exe_dir = std::filesystem::absolute(argv0, ec).parent_path();
    if (!ec) candidates.push_back(exe_dir / "barracuda.bnn");
    for (const auto& c : candidates)
        if (std::filesystem::exists(c, ec)) return try_load_net(c.string(), false);
    return nullptr;
}

std::string pv_string(const std::vector<Move>& pv) {
    std::string s;
    for (Move m : pv) s += m.to_uci() + " ";
    if (!s.empty()) s.pop_back();
    return s;
}

void print_info(const SearchInfo& i) {
    std::printf("info depth %d seldepth %d score %s nodes %llu nps %llu time %lld hashfull %d pv %s\n", i.depth,
                i.seldepth, score_to_uci(i.score).c_str(), (unsigned long long)i.nodes, (unsigned long long)i.nps,
                (long long)i.time_ms, i.hashfull, pv_string(i.pv).c_str());
    std::fflush(stdout);
}

// --------------------------------------------------------------------------- bench
const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1",
};

int cmd_bench(int depth, std::shared_ptr<const nnue::Network> net) {
    Search search(16);
    search.set_network(net);
    std::uint64_t nodes = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const char* fen : BENCH_FENS) {
        search.new_game();
        SearchLimits lim;
        lim.depth = depth;
        const SearchResult r = search.go(Position(fen), lim);
        nodes += r.nodes;
        std::printf("%-72s depth %2d best %-6s score %s\n", fen, r.depth, r.best.to_uci().c_str(),
                    score_to_string(r.score).c_str());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("===========================\nTotal time (ms) : %lld\nNodes searched  : %llu\nNodes/second    : %llu\n",
                (long long)(s * 1000), (unsigned long long)nodes, (unsigned long long)(double(nodes) / std::max(s, 1e-9)));
    std::printf("Evaluator       : %s\n", search.nnue_active() ? "NNUE" : "classical");
    return 0;
}

int cmd_perft(int depth, const std::string& fen) {
    Position pos;
    if (!fen.empty() && !pos.set_fen(fen)) {
        std::cerr << "bad fen\n";
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t total = 0;
    MoveList list;
    generate_moves(pos, list);
    for (Move m : list) {
        pos.do_move(m);
        const std::uint64_t n = depth > 1 ? perft(pos, depth - 1) : 1;
        pos.undo_move();
        total += n;
        std::printf("%s: %llu\n", m.to_uci().c_str(), (unsigned long long)n);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\nNodes: %llu  (%.2f Mnps)\n", (unsigned long long)total, double(total) / std::max(s, 1e-9) / 1e6);
    return 0;
}

int cmd_selfplay(const Args& a, std::shared_ptr<const nnue::Network> net) {
    SelfPlayConfig cfg;
    cfg.games = int(a.num("games", 10));
    cfg.depth = int(a.num("depth", 6));
    cfg.nodes = std::uint64_t(a.num("nodes", 0));
    cfg.random_plies = int(a.num("random-plies", 8));
    cfg.output_path = a.get("out", "selfplay.txt");
    cfg.seed = std::uint64_t(a.num("seed", 0));
    const int threads = std::max(1, int(a.num("threads", 1)));

    std::mutex io;
    std::atomic<int> games_done{0};
    std::atomic<std::uint64_t> positions{0};
    auto worker = [&](int id) {
        SelfPlayConfig c = cfg;
        c.games = cfg.games / threads + (id < cfg.games % threads ? 1 : 0);
        if (c.seed) c.seed += std::uint64_t(id) * 7919;
        else c.seed = std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count()) + std::uint64_t(id) * 7919;
        Search search(16);
        search.set_network(net);
        SelfPlayCallbacks cb;
        cb.on_game_end = [&](const SelfPlayStats& st, const std::string& res) {
            std::lock_guard<std::mutex> lk(io);
            (void)st;
            const int g = ++games_done;
            std::printf("game %d/%d  %s\n", g, cfg.games, res.c_str());
            std::fflush(stdout);
        };
        SelfPlayStats st;
        std::string err;
        if (!run_selfplay(search, c, g_stop, cb, &st, &err)) {
            std::lock_guard<std::mutex> lk(io);
            std::cerr << "selfplay error: " << err << "\n";
        }
        positions += st.positions;
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker, i);
    for (auto& t : pool) t.join();
    std::printf("wrote %llu positions to %s\n", (unsigned long long)positions.load(), cfg.output_path.c_str());
    return 0;
}

int cmd_train(const Args& a) {
    const std::string data_path = a.get("data");
    const std::string out_path = a.get("out", "barracuda.bnn");
    if (data_path.empty()) {
        std::cerr << "usage: barracuda-cli train --data <file> [--out barracuda.bnn] [--net init.bnn] [--epochs N] "
                     "[--batch N] [--lr X] [--lambda X] [--threads N] [--max-samples N]\n";
        return 1;
    }
    nnue::FloatNetwork net;
    if (a.has("net")) {
        std::string err;
        if (!net.load(a.get("net"), &err)) {
            std::cerr << err << "\n";
            return 1;
        }
        std::printf("resuming from %s (%u epochs)\n", a.get("net").c_str(), net.epochs_trained);
    } else {
        net.init_random(std::uint64_t(a.num("seed", 1)));
    }

    nnue::Dataset data;
    nnue::Dataset::LoadStats ls;
    std::string err;
    std::printf("loading %s ...\n", data_path.c_str());
    if (!data.load_text(data_path, std::uint64_t(a.num("max-samples", 0)), &ls, &err, &g_stop)) {
        std::cerr << err << "\n";
        return 1;
    }
    std::printf("loaded %llu samples (%llu rejected lines)\n", (unsigned long long)ls.accepted,
                (unsigned long long)ls.rejected);

    nnue::TrainerConfig cfg;
    cfg.epochs = int(a.num("epochs", 10));
    cfg.batch_size = int(a.num("batch", 8192));
    cfg.learning_rate = float(a.real("lr", 1e-3));
    cfg.lambda = float(a.real("lambda", 0.75));
    cfg.threads = int(a.num("threads", 0));
    cfg.lr_decay = float(a.real("lr-decay", 0.9));
    nnue::Trainer trainer(net, cfg);
    auto last = std::chrono::steady_clock::now();
    trainer.run(
        data, g_stop,
        [&](const nnue::TrainingProgress& p) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last < std::chrono::milliseconds(500)) return;
            last = now;
            std::printf("\repoch %d/%d  batch %llu/%llu  loss %.6f  %.0f pos/s   ", p.epoch, p.epochs,
                        (unsigned long long)p.batch, (unsigned long long)p.batches_per_epoch, p.train_loss,
                        p.samples_per_sec);
            std::fflush(stdout);
        },
        [&](const nnue::TrainingProgress& p) {
            std::printf("\repoch %d/%d  train %.6f  val %.6f  lr %.2e  %.0f pos/s\n", p.epoch, p.epochs, p.train_loss,
                        p.val_loss, p.learning_rate, p.samples_per_sec);
            std::string e;
            if (!net.save(out_path, &e)) std::cerr << e << "\n";
        });
    std::printf("saved %s\n", out_path.c_str());
    return 0;
}

int cmd_match(const Args& a, std::shared_ptr<const nnue::Network> net) {
    if (!net) {
        std::cerr << "match: no network loaded (use --net file.bnn or place barracuda.bnn next to the binary)\n";
        return 1;
    }
    MatchConfig cfg;
    cfg.games = int(a.num("games", 40));
    cfg.movetime_ms = int(a.num("movetime", 100));
    cfg.seed = std::uint64_t(a.num("seed", 0));
    Search nn(16), classical(16);
    nn.set_network(net);
    nn.set_eval_mode(EvalMode::NNUE);
    classical.set_eval_mode(EvalMode::CLASSICAL);
    std::printf("NNUE vs classical: %d games at %d ms/move\n", cfg.games, cfg.movetime_ms);
    MatchCallbacks cb;
    cb.on_game_end = [](const MatchStats& st, const std::string& d) {
        std::printf("game %3d  %s   +%d =%d -%d  Elo %+.0f +/- %.0f\n", st.games_done, d.c_str(), st.wins, st.draws,
                    st.losses, st.elo.elo, st.elo.error);
        std::fflush(stdout);
    };
    MatchStats st;
    run_match(nn, classical, cfg, g_stop, cb, &st);
    std::printf("\nresult: +%d =%d -%d  score %.1f%%  ->  NNUE is %+.0f +/- %.0f Elo vs classical\n", st.wins, st.draws,
                st.losses, st.elo.score * 100.0, st.elo.elo, st.elo.error);
    return 0;
}

int cmd_suite(const Args& a, std::shared_ptr<const nnue::Network> net) {
    std::vector<SuitePosition> suite;
    if (a.has("file")) {
        std::string err;
        int skipped = 0;
        if (!load_epd_file(a.get("file"), suite, &err, &skipped)) {
            std::cerr << err << "\n";
            return 1;
        }
        if (skipped) std::printf("skipped %d invalid EPD lines\n", skipped);
    } else {
        suite = builtin_wac_suite();
    }
    Search search(int(a.num("hash", 64)));
    search.set_network(net);
    const int ms = int(a.num("movetime", 1000));
    const SuiteResult r = run_suite(search, suite, ms, g_stop, [](const SuiteEntryResult& e, int i, int n) {
        std::printf("[%3d/%3d] %-10s expected %-12s found %-8s %s  (d%d %s)\n", i + 1, n, e.id.c_str(),
                    e.expected.c_str(), e.found.c_str(), e.solved ? "OK  " : "MISS", e.depth,
                    score_to_string(e.score).c_str());
        std::fflush(stdout);
    });
    std::printf("\nsolved %d / %d  ->  estimated Elo ~%.0f (heuristic, %d ms/pos)\n", r.solved, r.total, r.elo_estimate, ms);
    return 0;
}

// --------------------------------------------------------------------------- UCI
class UciEngine {
public:
    explicit UciEngine(std::shared_ptr<const nnue::Network> net) : search_(64) { search_.set_network(std::move(net)); }
    ~UciEngine() { stop(); }

    void loop() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!handle(line)) break;
        }
        stop();
    }

private:
    void stop() {
        search_.stop();
        if (thread_.joinable()) thread_.join();
    }

    bool handle(const std::string& line) {
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "uci") {
            std::printf("id name %s\nid author Barracuda Team\n", kVersion);
            std::printf("option name Hash type spin default 64 min 1 max 65536\n");
            std::printf("option name EvalFile type string default barracuda.bnn\n");
            std::printf("option name EvalMode type combo default Auto var Auto var Classical var NNUE\n");
            std::printf("uciok\n");
        } else if (cmd == "isready") {
            std::printf("readyok\n");
        } else if (cmd == "ucinewgame") {
            stop();
            search_.new_game();
        } else if (cmd == "setoption") {
            stop();
            setoption(ss);
        } else if (cmd == "position") {
            stop();
            position(ss);
        } else if (cmd == "go") {
            stop();
            go(ss);
        } else if (cmd == "stop") {
            stop();
        } else if (cmd == "quit") {
            return false;
        } else if (cmd == "d") {
            std::printf("%s", pos_.pretty().c_str());
        } else if (cmd == "eval") {
            std::printf("static eval (stm): %s  [%s]\n", score_to_string(search_.static_eval(pos_)).c_str(),
                        search_.nnue_active() ? "NNUE" : "classical");
        } else if (cmd == "perft") {
            int d = 5;
            ss >> d;
            cmd_perft(d, pos_.fen());
        } else if (cmd == "bench") {
            int d = 10;
            ss >> d;
            cmd_bench(d, search_.network());
        } else if (!cmd.empty()) {
            std::printf("info string unknown command: %s\n", cmd.c_str());
        }
        std::fflush(stdout);
        return true;
    }

    void setoption(std::istringstream& ss) {
        std::string tok, name, value;
        bool in_value = false;
        while (ss >> tok) {
            if (tok == "name") continue;
            if (tok == "value") { in_value = true; continue; }
            (in_value ? value : name) += (in_value ? (value.empty() ? "" : " ") : (name.empty() ? "" : " ")) + tok;
        }
        if (name == "Hash") {
            search_.set_hash_mb(std::size_t(std::max(1, std::atoi(value.c_str()))));
        } else if (name == "EvalFile") {
            if (auto net = try_load_net(value, true)) search_.set_network(net);
        } else if (name == "EvalMode") {
            search_.set_eval_mode(value == "Classical" ? EvalMode::CLASSICAL : value == "NNUE" ? EvalMode::NNUE : EvalMode::AUTO);
        }
    }

    void position(std::istringstream& ss) {
        std::string tok;
        ss >> tok;
        if (tok == "startpos") {
            pos_.set_fen(Position::START_FEN);
            ss >> tok; // "moves" (optional)
        } else if (tok == "fen") {
            std::string fen;
            while (ss >> tok && tok != "moves") fen += tok + " ";
            if (!pos_.set_fen(fen)) std::printf("info string invalid fen\n");
        }
        while (ss >> tok) {
            const Move m = parse_uci_move(pos_, tok);
            if (!m) {
                std::printf("info string illegal move %s\n", tok.c_str());
                break;
            }
            pos_.do_move(m);
        }
    }

    void go(std::istringstream& ss) {
        SearchLimits lim;
        std::string tok;
        while (ss >> tok) {
            long long v = 0;
            if (tok == "infinite") { lim.infinite = true; continue; }
            if (!(ss >> v)) break;
            if (tok == "depth") lim.depth = int(v);
            else if (tok == "movetime") lim.movetime_ms = v;
            else if (tok == "wtime") lim.time_ms[WHITE] = v;
            else if (tok == "btime") lim.time_ms[BLACK] = v;
            else if (tok == "winc") lim.inc_ms[WHITE] = v;
            else if (tok == "binc") lim.inc_ms[BLACK] = v;
            else if (tok == "movestogo") lim.movestogo = int(v);
            else if (tok == "nodes") lim.nodes = std::uint64_t(v);
        }
        const Position root = pos_;
        thread_ = std::thread([this, root, lim] {
            const SearchResult r = search_.go(root, lim, print_info);
            std::printf("bestmove %s", r.best.to_uci().c_str());
            if (r.ponder) std::printf(" ponder %s", r.ponder.to_uci().c_str());
            std::printf("\n");
            std::fflush(stdout);
        });
    }

    Search search_;
    Position pos_;
    std::thread thread_;
};

void usage() {
    std::printf(
        "%s - command line\n\n"
        "  barracuda-cli                       UCI mode (for Cute Chess, Arena, BanksiaGUI, ...)\n"
        "  barracuda-cli perft <depth> [fen]   move generator verification\n"
        "  barracuda-cli bench [depth]         fixed-depth node/speed benchmark\n"
        "  barracuda-cli selfplay --games N --depth D [--nodes N] [--threads T] [--out selfplay.txt]\n"
        "  barracuda-cli train --data selfplay.txt [--out barracuda.bnn] [--net init.bnn] [--epochs N]\n"
        "                      [--batch N] [--lr X] [--lambda X] [--threads N] [--max-samples N]\n"
        "  barracuda-cli suite [--file wac.epd] [--movetime ms]   tactical suite + Elo estimate\n"
        "  barracuda-cli match [--games N] [--movetime ms] [--net f.bnn]   NNUE vs classical, Elo +/- margin\n\n"
        "  Global: --net <file.bnn> (default: ./barracuda.bnn if present)  --classical\n",
        kVersion);
}

} // namespace

int main(int argc, char** argv) {
    Position::init();
    std::ios::sync_with_stdio(false);
    const std::string sub = argc > 1 ? argv[1] : "";
    const bool has_sub = !sub.empty() && sub.rfind("--", 0) != 0;
    const Args args = parse_args(argc, argv, has_sub ? 2 : 1);

    std::shared_ptr<const nnue::Network> net =
        args.has("net") ? try_load_net(args.get("net"), true) : autoload_net(argv[0]);
    if (args.has("classical")) net = nullptr;

    if (!has_sub) {
        std::printf("%s by the Barracuda team (NNUE %s, SIMD %s)\n", kVersion, net ? "loaded" : "not loaded",
                    nnue::simd_backend());
        std::fflush(stdout);
        UciEngine uci(net);
        uci.loop();
        return 0;
    }
    if (sub == "perft") {
        const int d = args.positional.empty() ? 5 : std::stoi(args.positional[0]);
        std::string fen;
        for (std::size_t i = 1; i < args.positional.size(); ++i) fen += args.positional[i] + " ";
        return cmd_perft(d, fen);
    }
    if (sub == "bench") return cmd_bench(args.positional.empty() ? 10 : std::stoi(args.positional[0]), net);
    if (sub == "selfplay") return cmd_selfplay(args, net);
    if (sub == "train") return cmd_train(args);
    if (sub == "suite") return cmd_suite(args, net);
    if (sub == "match") return cmd_match(args, net);
    usage();
    return sub == "help" || sub == "-h" ? 0 : 1;
}
