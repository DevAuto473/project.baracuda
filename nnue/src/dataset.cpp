#include "barracuda/nnue/dataset.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <random>

#include "barracuda/nnue/architecture.h"

namespace barracuda::nnue {

namespace {

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// returns 0/1/2 or -1
int parse_result(std::string r) {
    r = trim(r);
    if (!r.empty() && r.front() == '"') r = r.substr(1);
    if (!r.empty() && r.back() == '"') r.pop_back();
    if (r == "1.0" || r == "1" || r == "1-0" || r == "w" || r == "W") return 2;
    if (r == "0.0" || r == "0" || r == "0-1" || r == "b" || r == "B") return 0;
    if (r == "0.5" || r == "1/2-1/2" || r == "1/2" || r == "d" || r == "D" || r == "=") return 1;
    return -1;
}

bool parse_int(const std::string& s, int& v) {
    const std::string t = trim(s);
    if (t.empty()) return false;
    const char* b = t.data();
    if (*b == '+') ++b;
    auto [p, ec] = std::from_chars(b, t.data() + t.size(), v);
    if (ec != std::errc()) {
        // accept floats like "35.0"
        char* end = nullptr;
        const double d = std::strtod(t.c_str(), &end);
        if (end == t.c_str()) return false;
        v = int(d);
    }
    return true;
}

} // namespace

int PackedSample::features(Color perspective, int* out) const {
    for (int i = 0; i < count; ++i) {
        const Piece pc = Piece(pieces[i] >> 6);
        const Square sq = Square(pieces[i] & 63);
        out[i] = feature_index(perspective, pc, sq);
    }
    return count;
}

bool Dataset::pack_fen(const std::string& fen, PackedSample& out) {
    out.count = 0;
    int rank = 7, file = 0;
    std::size_t i = 0;
    for (; i < fen.size() && fen[i] != ' '; ++i) {
        const char c = fen[i];
        if (c == '/') {
            --rank;
            file = 0;
            if (rank < 0) return false;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            int p = -1;
            for (int k = 0; k < 12; ++k)
                if (PIECE_CHARS[k] == c) p = k;
            if (p < 0 || file > 7 || out.count >= 32) return false;
            out.pieces[out.count++] = std::uint16_t((p << 6) | make_square(file, rank));
            ++file;
        }
    }
    while (i < fen.size() && fen[i] == ' ') ++i;
    if (i >= fen.size()) return false;
    out.stm = fen[i] == 'b' ? BLACK : WHITE;
    return rank == 0 && out.count >= 2;
}

bool Dataset::parse_line(const std::string& raw, PackedSample& out) {
    const std::string line = trim(raw);
    if (line.empty() || line[0] == '#') return false;

    std::string fen, score_s, result_s;
    if (line.find('|') != std::string::npos) {
        const auto a = line.find('|');
        const auto b = line.find('|', a + 1);
        if (b == std::string::npos) return false;
        fen = trim(line.substr(0, a));
        score_s = line.substr(a + 1, b - a - 1);
        result_s = line.substr(b + 1);
    } else if (line.find('[') != std::string::npos) {
        const auto a = line.find('[');
        const auto b = line.find(']', a);
        if (b == std::string::npos) return false;
        fen = trim(line.substr(0, a));
        result_s = line.substr(a + 1, b - a - 1);
        score_s = line.substr(b + 1);
    } else {
        return false;
    }

    int score = 0;
    const int result = parse_result(result_s);
    if (!parse_int(score_s, score) || result < 0) return false;
    if (!pack_fen(fen, out)) return false;
    out.score = std::int16_t(std::clamp(score, -32000, 32000));
    out.result = std::uint8_t(result);
    return true;
}

bool Dataset::load_text(const std::string& path, std::uint64_t max_samples, LoadStats* stats, std::string* error,
                        const std::atomic<bool>* stop, const ProgressFn& progress) {
    std::ifstream f(path);
    if (!f) {
        if (error) *error = "cannot open dataset '" + path + "'";
        return false;
    }
    LoadStats local;
    LoadStats& st = stats ? *stats : local;
    std::string line;
    PackedSample s;
    while (std::getline(f, line)) {
        ++st.lines;
        if (parse_line(line, s)) {
            samples_.push_back(s);
            ++st.accepted;
            if (max_samples && st.accepted >= max_samples) break;
        } else if (!line.empty() && line[0] != '#') {
            ++st.rejected;
        }
        if ((st.lines & 0xFFFF) == 0) {
            if (progress) progress(st.lines, st.accepted);
            if (stop && stop->load(std::memory_order_relaxed)) break;
        }
    }
    if (progress) progress(st.lines, st.accepted);
    return true;
}

void Dataset::add(const Position& pos, int score_white, int result_white) {
    PackedSample s;
    Bitboard occ = pos.pieces();
    while (occ && s.count < 32) {
        const Square sq = pop_lsb(occ);
        s.pieces[s.count++] = std::uint16_t((pos.piece_on(sq) << 6) | sq);
    }
    s.stm = std::uint8_t(pos.side_to_move());
    s.score = std::int16_t(std::clamp(score_white, -32000, 32000));
    s.result = std::uint8_t(std::clamp(result_white, 0, 2));
    samples_.push_back(s);
}

void Dataset::shuffle(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::shuffle(samples_.begin(), samples_.end(), rng);
}

} // namespace barracuda::nnue
