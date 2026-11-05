// Barracuda unit tests (dependency-free mini framework, run via CTest).
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "barracuda/engine/match.h"
#include "barracuda/engine/movegen.h"
#include "barracuda/engine/notation.h"
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
    CHECK(max_err < 25.0);
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
    std::printf("  samples %zu  loss %.5f -> %.5f (last epoch running %.5f)\n", data.size(), before, final_loss, after);
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
