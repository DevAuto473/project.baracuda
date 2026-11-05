#include "barracuda/engine/suite.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "barracuda/engine/notation.h"

namespace barracuda {

namespace {

// Curated subset of the classic "Win At Chess" (Reinfeld) tactical suite.
// Every entry is validated on load (FEN parses, best move is legal) so a typo can never
// poison the benchmark; drop in the full wac.epd / STS files via load_epd_file() for more.
constexpr const char* WAC_EPD = R"EPD(
2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - bm Qg6; id "WAC.001";
8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - bm Rxb2; id "WAC.002";
r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - bm Qxh7+; id "WAC.004";
5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - bm Qc4+; id "WAC.005";
7k/p7/1R5K/6r1/6p1/6P1/8/8 w - - bm Rb7; id "WAC.006";
rnbqkb1r/pppp1ppp/8/4P3/6n1/7P/PPPNPPP1/R1BQKBNR b KQkq - bm Ne3; id "WAC.007";
r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - - bm Rf7; id "WAC.008";
3q1rk1/p4pp1/2pb3p/3p4/6Pr/1PNQ4/P1PB1PP1/4RRK1 b - - bm Bh2+; id "WAC.009";
2br2k1/2q3rn/p2NppQ1/2p1P3/Pp5R/4P3/1P3PPP/3R2K1 w - - bm Rxh7; id "WAC.010";
r1b1kb1r/3q1ppp/pBp1pn2/8/Np3P2/5B2/PPP3PP/R2Q1RK1 w kq - bm Bxc6; id "WAC.011";
4k1r1/2p3r1/1pR1p3/3pP2p/3P2qP/P4N2/1PQ4P/5R1K b - - bm Qxf3+; id "WAC.012";
5rk1/pp4p1/2n1p2p/2Npq3/2p5/6P1/P3P1BP/R4Q1K w - - bm Qxf8+; id "WAC.013";
r2rb1k1/pp1q1p1p/2n1p1p1/2bp4/5P2/PP1BPR1Q/1BPN2PP/R5K1 w - - bm Qxh7+; id "WAC.014";
1R6/1brk2p1/4p2p/p1P1Pp2/P7/6P1/1P4P1/2R3K1 w - - bm Rxb7; id "WAC.015";
r4rk1/ppp2ppp/2n5/2bqp3/8/P2PB3/1PP1NPPP/R2Q1RK1 w - - bm Nc3; id "WAC.016";
R7/P4k2/8/8/8/8/r7/6K1 w - - bm Rh8; id "WAC.018";
r1b2rk1/ppbn1ppp/4p3/1QP4q/3P4/N4N2/5PPP/R1B2RK1 w - - bm c6; id "WAC.019";
r2qkb1r/1ppb1ppp/p7/4p3/P1Q1P3/2P5/5PPP/R1B2KNR b kq - bm Bb5; id "WAC.020";
5rk1/1b3p1p/pp3p2/3n1N2/1P6/P1qB1PP1/3Q3P/4R1K1 w - - bm Qh6; id "WAC.021";
r1bqk2r/ppp1nppp/4p3/n5N1/2BPp3/P1P5/2P2PPP/R1BQK2R w KQkq - bm Ba2 Nxf7; id "WAC.022";
r3nrk1/2p2p1p/p1p1b1p1/2NpPq2/3R4/P1N1Q3/1PP2PPP/4R1K1 w - - bm g4; id "WAC.023";
6k1/1b1nqpbp/pp4p1/5P2/1PN5/4Q3/P5PP/1B2B1K1 b - - bm Bd4; id "WAC.024";
3R1rk1/8/5Qpp/2p5/2P1p1q1/P3P3/1P2PK2/8 b - - bm Qh4+; id "WAC.025";
3r2k1/1p1b1pp1/pq5p/8/3NR3/2PQ3P/PP3PP1/6K1 b - - bm Bf5; id "WAC.026";
7k/pp4np/2p3p1/3pN1q1/3P4/Q7/1r3rPP/2R2RK1 w - - bm Qf8+; id "WAC.027";
1r1r2k1/4pp1p/2p1b1p1/p3R3/RqBP4/4P3/1PQ2PPP/6K1 b - - bm Qe1+; id "WAC.028";
r2q2k1/pp1rbppp/4pn2/2P5/1P3B2/6P1/P3QPBP/1R3RK1 w - - bm c6; id "WAC.029";
1r3r2/4q1kp/b1pp2p1/5p2/pPn1N3/6P1/P3PPBP/2QRR1K1 w - - bm Nxd6; id "WAC.030";
)EPD";

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool parse_epd_line(const std::string& raw, SuitePosition& out) {
    const std::string line = trim(raw);
    if (line.empty() || line[0] == '#') return false;
    std::istringstream ss(line);
    std::string f[4];
    for (auto& x : f)
        if (!(ss >> x)) return false;
    out = SuitePosition{};
    out.fen = f[0] + " " + f[1] + " " + f[2] + " " + f[3] + " 0 1";
    Position pos;
    if (!pos.set_fen(out.fen)) return false;

    std::string rest;
    std::getline(ss, rest);
    std::stringstream ops(rest);
    std::string op;
    while (std::getline(ops, op, ';')) {
        op = trim(op);
        if (op.empty()) continue;
        std::istringstream os(op);
        std::string opcode;
        os >> opcode;
        std::string arg;
        if (opcode == "bm" || opcode == "am") {
            std::string text;
            while (os >> arg) {
                const Move m = parse_san_move(pos, arg);
                if (!m) return false; // illegal move in suite: reject the whole entry
                (opcode == "bm" ? out.best : out.avoid).push_back(m);
                text += (text.empty() ? "" : " ") + arg;
            }
            out.expected += (out.expected.empty() ? "" : "; ") + (opcode == "am" ? "avoid " + text : text);
        } else if (opcode == "id") {
            std::getline(os, arg);
            arg = trim(arg);
            if (arg.size() >= 2 && arg.front() == '"' && arg.back() == '"') arg = arg.substr(1, arg.size() - 2);
            out.id = arg;
        }
    }
    return !out.best.empty() || !out.avoid.empty();
}

} // namespace

std::vector<SuitePosition> parse_epd(const std::string& text, int* skipped) {
    std::vector<SuitePosition> out;
    std::istringstream ss(text);
    std::string line;
    int bad = 0;
    while (std::getline(ss, line)) {
        if (trim(line).empty() || trim(line)[0] == '#') continue;
        SuitePosition p;
        if (parse_epd_line(line, p)) {
            if (p.id.empty()) p.id = "#" + std::to_string(out.size() + 1);
            out.push_back(std::move(p));
        } else {
            ++bad;
        }
    }
    if (skipped) *skipped = bad;
    return out;
}

bool load_epd_file(const std::string& path, std::vector<SuitePosition>& out, std::string* error, int* skipped) {
    std::ifstream f(path);
    if (!f) {
        if (error) *error = "cannot open '" + path + "'";
        return false;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    out = parse_epd(buf.str(), skipped);
    if (out.empty()) {
        if (error) *error = "no valid EPD entries in '" + path + "'";
        return false;
    }
    return true;
}

const std::vector<SuitePosition>& builtin_wac_suite() {
    static const std::vector<SuitePosition> suite = [] {
        Position::init();
        return parse_epd(WAC_EPD);
    }();
    return suite;
}

double estimate_elo(int solved, int total) {
    if (total <= 0) return 0.0;
    const double r = std::clamp(double(solved) / double(total), 0.02, 0.995);
    // logit fit: ~50% -> 1400, ~83% -> 2000, ~98% -> 2850 on WAC at ~1s/position
    return 1400.0 + 870.0 * std::log10(r / (1.0 - r));
}

SuiteResult run_suite(Search& search, const std::vector<SuitePosition>& suite, int movetime_ms,
                      const std::atomic<bool>& stop, const SuiteCallback& cb) {
    SuiteResult res;
    res.total = int(suite.size());
    for (int i = 0; i < int(suite.size()); ++i) {
        if (stop.load()) break;
        const SuitePosition& sp = suite[std::size_t(i)];
        Position pos(sp.fen);
        search.new_game();
        SearchLimits lim;
        lim.movetime_ms = std::max(10, movetime_ms);
        lim.move_overhead_ms = 0;
        const SearchResult r = search.go(pos, lim);

        SuiteEntryResult e;
        e.id = sp.id;
        e.expected = sp.expected;
        e.depth = r.depth;
        e.score = r.score;
        Position tmp = pos;
        e.found = r.best ? move_to_san(tmp, r.best) : "-";
        const bool in_best = sp.best.empty() || std::find(sp.best.begin(), sp.best.end(), r.best) != sp.best.end();
        const bool in_avoid = std::find(sp.avoid.begin(), sp.avoid.end(), r.best) != sp.avoid.end();
        e.solved = in_best && !in_avoid;
        if (e.solved) ++res.solved;
        res.entries.push_back(e);
        if (cb) cb(e, i, res.total);
    }
    res.total = int(res.entries.size()) > 0 ? int(res.entries.size()) : res.total;
    res.elo_estimate = estimate_elo(res.solved, res.total);
    return res;
}

} // namespace barracuda
