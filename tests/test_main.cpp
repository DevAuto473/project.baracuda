// Barracuda unit tests (dependency-free mini framework, run via CTest).
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "barracuda/engine/match.h"
#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"
#include "barracuda/engine/human_model.h"
#include "barracuda/engine/import_pgn.h"
#include "barracuda/engine/search.h"
#include "barracuda/engine/suite.h"
#include "barracuda/nnue/dataset.h"
#include "barracuda/nnue/inspector.h"
#include "barracuda/nnue/network.h"
#include "barracuda/nnue/trainer.h"
#include "runtime/concurrency.h"

using namespace barracuda;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) {                                                                 \
            ++g_failures;                                                              \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
        }                                                                              \
    } while (0)

struct TestCase {
    const char* name;
    std::function<void()> fn;
};
std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
#define TEST(name)                                     \
    static void name();                                \
    static Registrar reg_##name(#name, name);          \
    static void name()

// ---------------------------------------------------------------------------
TEST(perft_suite) {
    struct P { const char* fen; int depth; std::uint64_t nodes; };
    const P cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
    };
    for (const P& c : cases) {
        Position pos(c.fen);
        const std::uint64_t n = perft(pos, c.depth);
        if (n != c.nodes) std::printf("  perft mismatch %s: %llu vs %llu\n", c.fen, (unsigned long long)n, (unsigned long long)c.nodes);
        CHECK(n == c.nodes);
    }
}

TEST(fen_roundtrip_and_hash) {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/4k3/8/2p5/8/B2P2K1/8 b - - 12 40",
    };
    for (const char* f : fens) {
        Position p(f);
        CHECK(p.fen() == f);
    }
    // incremental key == key recomputed from FEN, along a random game
    std::mt19937 rng(42);
    Position p;
    for (int i = 0; i < 300; ++i) {
        MoveList l;
        generate_moves(p, l);
        if (l.size == 0) break;
        p.do_move(l[int(rng() % unsigned(l.size))]);
        Position q(p.fen());
        CHECK(q.key() == p.key());
        if (game_result(p) != GameResult::ONGOING) break;
    }
    std::string err;
    Position bad;
    CHECK(!bad.set_fen("not a fen", &err));
    CHECK(bad.fen() == Position::START_FEN); // unchanged on failure
}

TEST(san_notation) {
    Position p;
    Move m = parse_san_move(p, "Nf3");
    CHECK(m && m.to_uci() == "g1f3");
    Position k("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    Move c = parse_san_move(k, "O-O-O");
    CHECK(c && c.flags() == QUEEN_CASTLE);
    CHECK(move_to_san(k, parse_uci_move(k, "e5f7")) == "Nxf7");
    Position mate("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1");
    CHECK(move_to_san(mate, parse_uci_move(mate, "d1d8")) == "Rd8#");
}

TEST(search_finds_mates_and_tactics) {
    Search s(16);
    SearchLimits lim;
    lim.depth = 6;
    // back-rank mate in one
    SearchResult r = s.go(Position("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"), lim);
    CHECK(r.best.to_uci() == "d1d8");
    CHECK(r.score == mate_in(1));
    // mate in two (Qf7+ style)
    s.new_game();
    lim.depth = 8;
    r = s.go(Position("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"), lim);
    CHECK(r.best.to_uci() == "h5f7");
    // winning a hanging queen
    s.new_game();
    lim.depth = 5;
    r = s.go(Position("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"), lim);
    CHECK(r.best.to_uci() == "d2d5");
    // stalemate / checkmate root handling
    r = s.go(Position("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"), lim);
    CHECK(!r.best && r.score == 0);
}

TEST(builtin_suite_valid) {
    const auto& suite = builtin_wac_suite();
    CHECK(suite.size() >= 25);
    for (const auto& p : suite) CHECK(!p.best.empty());
}

TEST(nnue_incremental_matches_refresh) {
    nnue::FloatNetwork f;
    f.init_random(7);
    f.epochs_trained = 1;
    auto net = nnue::Network::from_float(f);
    std::mt19937 rng(1234);
    for (int game = 0; game < 20; ++game) {
        Position p;
        nnue::AccumulatorStack stack;
        stack.reset(*net, p);
        for (int ply = 0; ply < MAX_PLY; ++ply) { // stack depth is bounded by the search ply limit
            MoveList l;
            generate_moves(p, l);
            if (l.size == 0 || game_result(p) != GameResult::ONGOING) break;
            p.do_move(l[int(rng() % unsigned(l.size))]);
            stack.push(*net, p.dirty());
            nnue::Accumulator fresh;
            net->refresh(p, fresh);
            bool same = true;
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < nnue::kHidden; ++i) same &= fresh.v[c][i] == stack.current().v[c][i];
            CHECK(same);
            CHECK(net->evaluate(stack.current(), p.side_to_move()) == net->evaluate(p));
            if (!same) return;
        }
    }
}

TEST(nnue_add_sub_feature) {
    nnue::FloatNetwork f;
    f.init_random(3);
    auto net = nnue::Network::from_float(f);
    Position p;
    nnue::Accumulator a, b;
    net->refresh(p, a);
    b = a;
    const int feat = nnue::feature_index(WHITE, W_QUEEN, E4);
    net->add_feature(b, WHITE, feat);
    net->sub_feature(b, WHITE, feat);
    bool same = true;
    for (int i = 0; i < nnue::kHidden; ++i) same &= a.v[WHITE][i] == b.v[WHITE][i];
    CHECK(same);
}

TEST(nnue_quantised_matches_float) {
    nnue::FloatNetwork f;
    f.init_random(11);
    auto net = nnue::Network::from_float(f);
    std::mt19937 rng(5);
    double max_err = 0.0;
    Position p;
    for (int ply = 0; ply < 60; ++ply) {
        MoveList l;
        generate_moves(p, l);
        if (l.size == 0) break;
        p.do_move(l[int(rng() % unsigned(l.size))]);
        nnue::PackedSample s;
        nnue::Dataset::pack_fen(p.fen(), s);
        int fs[32], fn[32];
        const int ns = s.features(p.side_to_move(), fs);
        const int nn = s.features(~p.side_to_move(), fn);
        const double fl = double(f.forward(fs, ns, fn, nn)) * nnue::kOutputScale;
        const double q = net->evaluate(p);
        max_err = std::max(max_err, std::abs(fl - q));
    }
    std::printf("  max |float - int| = %.2f cp\n", max_err);
    // A random network's rounding error varies a little between platforms/libm (20-30 cp seen); a real
    // quantisation bug shows up as hundreds of cp, so this bound only guards against gross errors.
    CHECK(max_err < 60.0);
}

TEST(network_save_load) {
    nnue::FloatNetwork a;
    a.init_random(99);
    a.epochs_trained = 3;
    const std::string path = "barracuda_test_net.bnn";
    CHECK(a.save(path));
    nnue::FloatNetwork b;
    CHECK(b.load(path));
    CHECK(b.epochs_trained == 3);
    CHECK(b.ft_w == a.ft_w && b.l1_w == a.l1_w && b.out_b == a.out_b);
    std::remove(path.c_str());
}

TEST(dataset_parsing) {
    nnue::PackedSample s;
    CHECK(nnue::Dataset::parse_line("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | 35 | 0.5", s));
    CHECK(s.count == 32 && s.stm == BLACK && s.score == 35 && s.result == 1);
    CHECK(nnue::Dataset::parse_line("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1 [1.0] 250", s));
    CHECK(s.count == 3 && s.score == 250 && s.result == 2);
    CHECK(!nnue::Dataset::parse_line("garbage", s));
    CHECK(!nnue::Dataset::parse_line("# comment", s));
}

TEST(trainer_reduces_loss) {
    // Teach the network the classical evaluation of random positions.
    nnue::Dataset data;
    std::mt19937 rng(77);
    Search teacher(4);
    for (int g = 0; g < 60; ++g) {
        Position p;
        for (int ply = 0; ply < 40; ++ply) {
            MoveList l;
            generate_moves(p, l);
            if (l.size == 0 || game_result(p) != GameResult::ONGOING) break;
            p.do_move(l[int(rng() % unsigned(l.size))]);
            if (p.in_check()) continue;
            const Value v = teacher.static_eval(p);
            data.add(p, p.side_to_move() == WHITE ? v : -v, 1);
        }
    }
    nnue::FloatNetwork net;
    net.init_random(1);
    nnue::TrainerConfig cfg;
    cfg.epochs = 6;
    cfg.batch_size = 256;
    cfg.learning_rate = 2e-3f;
    cfg.lambda = 1.0f;
    cfg.threads = 2;
    nnue::Trainer tr(net, cfg);
    const double before = tr.loss(data, 0, data.size());
    std::atomic<bool> stop{false};
    double after = before;
    tr.run(data, stop, {}, [&](const nnue::TrainingProgress& p) { after = p.train_loss; });
    const double final_loss = tr.loss(data, 0, data.size());
    std::printf("  samples %d  loss %.5f -> %.5f (last epoch running %.5f)\n", int(data.size()), before, final_loss, after);
    CHECK(final_loss < before * 0.7);
    CHECK(net.epochs_trained == 6);
}

TEST(inspector_view) {
    nnue::FloatNetwork f;
    f.init_random(5);
    auto net = nnue::Network::from_float(f);
    nnue::NetworkView v;
    nnue::build_network_view(*net, Position(), v);
    CHECK(v.valid);
    CHECK(v.layers[0].activation.size() == 32);
    CHECK(v.edges[0].size() == 32u * nnue::kViewFtNodes);
    CHECK(v.edges[1].size() == std::size_t(nnue::kViewFtNodes) * nnue::kL1);
    CHECK(v.layers[4].activation.size() == 1);
}

TEST(triple_buffer_and_queue) {
    runtime::TripleBuffer<std::vector<int>> tb;
    std::atomic<bool> done{false};
    std::thread writer([&] {
        for (int i = 1; i <= 20000; ++i) {
            auto& w = tb.write_buffer();
            w.assign(16, i); // consistent payload: all elements equal
            tb.publish();
        }
        done = true;
    });
    int last = 0;
    bool consistent = true, monotonic = true;
    while (!done || tb.has_new()) {
        if (tb.update()) {
            const auto& r = tb.read_buffer();
            for (int x : r) consistent &= x == r[0];
            monotonic &= r[0] >= last;
            last = r[0];
        }
    }
    writer.join();
    tb.update();
    CHECK(consistent);
    CHECK(monotonic);
    CHECK(tb.read_buffer().size() == 16 && tb.read_buffer()[0] == 20000);

    runtime::MpmcQueue<int> q(1024);
    std::atomic<int> produced{0};
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t)
        producers.emplace_back([&] {
            for (int i = 0; i < 5000; ++i) {
                while (!q.try_push(1)) std::this_thread::yield();
                ++produced;
            }
        });
    int consumed = 0, v = 0;
    while (consumed < 20000)
        if (q.try_pop(v)) consumed += v;
    for (auto& t : producers) t.join();
    CHECK(consumed == 20000);
}

TEST(elo_and_match) {
    CHECK(std::abs(elo_from_results(60, 0, 40).elo - 70.4) < 0.2);
    CHECK(std::abs(elo_from_results(10, 20, 10).elo) < 1e-6);
    CHECK(elo_from_results(60, 0, 40).error > 0 && elo_from_results(600, 0, 400).error < elo_from_results(60, 0, 40).error);
    CHECK(elo_from_results(0, 0, 0).elo == 0.0);
    Search a(4), b(4);
    MatchConfig cfg;
    cfg.games = 2;
    cfg.movetime_ms = 15;
    cfg.max_plies = 30;
    cfg.random_plies = 4;
    cfg.seed = 7;
    std::atomic<bool> stop{false};
    MatchStats st;
    run_match(a, b, cfg, stop, {}, &st);
    CHECK(st.games_done == 2 && st.wins + st.draws + st.losses == 2);
}

TEST(strange_style) {
    StyleOptions so;
    so.enabled = true;
    so.seed = 3;
    // a forced mate is never traded for style
    Search s(8);
    s.set_style(so);
    SearchLimits lim;
    lim.depth = 6;
    SearchResult r = s.go(Position("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"), lim);
    CHECK(r.best.to_uci() == "d1d8" && !r.styled);
    // strangeness features
    Position king_walk("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    CHECK(move_strangeness(king_walk, parse_uci_move(king_walk, "e1e2")) >= 30);
    CHECK(move_strangeness(king_walk, parse_uci_move(king_walk, "f3h4")) >= 15);
    CHECK(move_strangeness(king_walk, parse_uci_move(king_walk, "d2d4")) < 10);
    CHECK(style_margin(-300) == 0 && style_margin(0) > 0 && style_margin(1000) > style_margin(0));
    // whenever the style picks another move, an independent deeper search agrees it was a small concession
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    };
    int styled = 0;
    for (const char* f : fens) {
        Position pos(f);
        Search a(8), judge(8);
        a.set_style(so);
        lim.depth = 7;
        const SearchResult res = a.go(pos, lim);
        if (!res.styled) continue;
        ++styled;
        SearchLimits deep;
        deep.depth = 8;
        auto score_after = [&](Move m) { Position p = pos; p.do_move(m); judge.new_game(); return -judge.go(p, deep).score; };
        const int best = score_after(res.objective_best), chosen = score_after(res.best);
        const int allowed = style_margin(best) + 40; // + tolerance: different depths disagree a little
        if (chosen < best - allowed) std::printf("  %s: %s (%d) vs %s (%d)\n", f, res.best.to_uci().c_str(), chosen, res.objective_best.to_uci().c_str(), best);
        CHECK(chosen >= best - allowed);
    }
    std::printf("  style changed the move in %d of %zu positions\n", styled, std::size(fens));
}

TEST(traps_and_prediction) {
    // the human model takes a hanging queen at once
    Search s(8);
    Position hang("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1");
    CHECK(s.predict_human(hang).to_uci() == "d2d5");
    // a trap never replaces a forced mate, and is never set when the engine is worse
    StyleOptions so;
    so.enabled = true;
    so.trap = 100;
    so.contempt = 30;
    so.seed = 5;
    s.set_style(so);
    SearchLimits lim;
    lim.depth = 6;
    SearchResult r = s.go(Position("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"), lim);
    CHECK(r.best.to_uci() == "d1d8");
    // whenever a trap is set, the natural reply really loses and the trap itself stays sound
    const char* fens[] = {
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "r2qkbnr/ppp2ppp/2np4/4p3/2B1P1b1/5N2/PPPP1PPP/RNBQ1RK1 w kq - 2 5",
    };
    int traps = 0;
    for (const char* f : fens) {
        Position pos(f);
        Search a(8), judge(8);
        a.set_style(so);
        lim.depth = 7;
        const SearchResult res = a.go(pos, lim);
        if (!res.trap_reply) continue;
        ++traps;
        SearchLimits deep;
        deep.depth = 7;
        Position after = pos;
        after.do_move(res.best);
        const int trap_score = -judge.go(after, deep).score;
        judge.new_game();
        Position fell = after;
        fell.do_move(res.trap_reply);
        const int punished = judge.go(fell, deep).score;
        std::printf("  trap %s, natural reply %s: %d after best defence, %d if it falls\n", res.best.to_uci().c_str(),
                    res.trap_reply.to_uci().c_str(), trap_score, punished);
        CHECK(punished > trap_score + 60);
    }
    std::printf("  traps set in %d of %zu positions\n", traps, std::size(fens));
}

// A file name in the system's temporary folder (works on Windows too, where /tmp does not exist).
std::string temp_file(const char* name) { return (std::filesystem::temp_directory_path() / name).string(); }

TEST(import_pgn) {
    // Lichess-style PGN: comments with [%eval], clock tags, a game outside the rating window, a short game
    const std::string game =
        "1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] } 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 "
        "6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 "
        "15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0\n\n";
    const std::string pgn = "[Result \"1-0\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1450\"]\n\n" + game +
                            "[Result \"1-0\"]\n[WhiteElo \"2500\"]\n[BlackElo \"2500\"]\n\n" + game +
                            "[Result \"1-0\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1500\"]\n\n1. e4 e5 2. Qh5 1-0\n\n";
    std::istringstream in(pgn);
    ImportPgnOptions o;
    o.min_elo = 1200;
    o.max_elo = 1800;
    o.skip_plies = 1;
    o.every = 1;
    o.depth = 3;
    ImportPgnStats st;
    std::atomic<bool> stop{false};
    std::string err;
    const std::string out = temp_file("barracuda_import_pgn_test.txt");
    CHECK(import_pgn(in, out, o, nullptr, stop, st, &err));
    CHECK(st.games_seen == 3 && st.games_used == 1 && st.rejected_elo == 1 && st.rejected_short == 1);
    CHECK(st.positions > 10 && st.from_pgn_eval == 2); // the two evaluated opening positions cost no search
    std::ifstream f(out);
    std::string line;
    std::getline(f, line);
    CHECK(line == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 | 17 | 1.0");
    std::remove(out.c_str());

    // moves for the human model: every played move, captures and checks included, only the accepted game
    std::istringstream in2(pgn);
    ImportPgnOptions o2 = o;
    o2.moves_only = true;
    o2.moves_every = 1;
    o2.skip_plies = 0;
    o2.moves_out = temp_file("barracuda_import_moves_test.txt");
    ImportPgnStats st2;
    CHECK(import_pgn(in2, temp_file("never_written.txt"), o2, nullptr, stop, st2, &err));
    CHECK(st2.games_used == 1 && st2.moves_written == 45 && st2.positions == 0);
    std::ifstream mf(o2.moves_out);
    int legal_lines = 0, captures = 0;
    while (std::getline(mf, line)) {
        const auto bar = line.find(" | ");
        Position pp(line.substr(0, bar));
        const Move mv = parse_uci_move(pp, line.substr(bar + 3));
        legal_lines += mv ? 1 : 0;
        captures += mv && mv.is_capture();
    }
    CHECK(legal_lines == 45 && captures >= 5);
    std::remove(o2.moves_out.c_str());
}

TEST(human_model) {
    // feature sanity
    Position pos("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3");
    float f[kHumanFeatures];
    HumanModel::features(pos, parse_uci_move(pos, "f6e4"), f);
    CHECK(f[1] == 1.0f && f[6] == 1.0f && f[7] > 0.0f); // a knight capturing a pawn
    HumanModel::features(pos, parse_uci_move(pos, "d7d6"), f);
    CHECK(f[0] == 1.0f && f[6] == 0.0f && f[19] == 1.0f);
    CHECK(pos.fen() == "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"); // restored

    // stand-in for human games: a 2-ply player that errs 15% of the time, from varied openings
    Search player(4);
    player.set_eval_mode(EvalMode::CLASSICAL);
    std::mt19937_64 rng(7);
    const std::string path = temp_file("barracuda_human_moves_test.txt");
    {
        std::ofstream out(path);
        for (int g = 0; g < 150; ++g) {
            Position p;
            p.set_fen(Position::START_FEN);
            for (int ply = 0; ply < 60 && game_result(p) == GameResult::ONGOING; ++ply) {
                const MoveList l = legal_moves(p);
                Move m = rng() % 100 < 15 || ply < 4 ? l[std::size_t(rng() % std::size_t(l.size))] : player.predict_human(p, 2);
                if (ply >= 4 && ply % 2 == 0) out << p.fen() << " | " << m.to_uci() << "\n";
                p.do_move(m);
            }
        }
    }
    HumanModel model;
    HumanTrainOptions o;
    o.epochs = 4;
    o.threads = 2;
    HumanTrainStats st;
    std::atomic<bool> stop{false};
    std::string err;
    CHECK(train_human_model(path, model, o, stop, st, &err));
    std::printf("  %llu samples: held-out top-1 %.1f%% (uniform %.1f%%), top-3 %.1f%%, NLL %.3f (uniform %.3f)\n",
                (unsigned long long)st.samples, 100 * st.val_top1, 100 * st.baseline_top1, 100 * st.val_top3, st.val_nll, st.baseline_nll);
    CHECK(st.val_nll < st.baseline_nll - 0.2 && st.val_top1 > st.baseline_top1 * 1.8);
    // probabilities
    std::vector<RankedMove> r;
    Position start;
    start.set_fen(Position::START_FEN);
    {
        Search ls(1);
        ls.set_eval_mode(EvalMode::CLASSICAL);
        std::vector<ShallowScore> sc;
        ls.shallow_scores(start, 2, sc);
        CHECK(sc.size() == 20);
        model.rank(start, r, &sc);
    }
    float sum = 0;
    for (const RankedMove& x : r) sum += x.p;
    CHECK(r.size() == 20 && std::fabs(sum - 1.0f) < 1e-4f && r[0].p >= r[1].p);
    // save / load round trip
    CHECK(model.save(temp_file("barracuda_human_test.mdl")));
    const auto back = HumanModel::load(temp_file("barracuda_human_test.mdl"));
    CHECK(back && std::fabs(back->w[6] - model.w[6]) < 1e-5f && back->samples == model.samples);
    CHECK(!HumanModel::load(temp_file("barracuda_human_moves_test.txt")));
    std::remove(path.c_str());
    std::remove(temp_file("barracuda_human_test.mdl").c_str());
}

} // namespace

int main(int argc, char** argv) {
    Position::init();
    const std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0;
    for (const auto& t : registry()) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) continue;
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        t.fn();
        std::printf("[ %s ] %s\n", g_failures == before ? " OK " : "FAIL", t.name);
        ++ran;
    }
    std::printf("\n%d tests, %d checks, %d failures (SIMD backend: %s)\n", ran, g_checks, g_failures, nnue::simd_backend());
    return g_failures == 0 ? 0 : 1;
}
