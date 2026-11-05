// Barracuda - EPD tactical test suites (WAC / STS style) and a heuristic Elo estimate.
#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "barracuda/engine/search.h"

namespace barracuda {

struct SuitePosition {
    std::string id;
    std::string fen;
    std::vector<Move> best;  // "bm"
    std::vector<Move> avoid; // "am"
    std::string expected;    // original text of bm/am for display
};

struct SuiteEntryResult {
    std::string id;
    std::string expected;
    std::string found;
    bool solved = false;
    int depth = 0;
    Value score = 0;
};

struct SuiteResult {
    int solved = 0;
    int total = 0;
    double elo_estimate = 0.0;
    std::vector<SuiteEntryResult> entries;
};

// Parses EPD text ("<4 FEN fields> bm Qg6; id \"WAC.001\";"). Invalid/illegal entries are skipped.
std::vector<SuitePosition> parse_epd(const std::string& text, int* skipped = nullptr);
bool load_epd_file(const std::string& path, std::vector<SuitePosition>& out, std::string* error, int* skipped = nullptr);

// Built-in curated Win-At-Chess subset (positions validated at load time).
const std::vector<SuitePosition>& builtin_wac_suite();

// Heuristic: maps a solve ratio at a fixed time per move to an approximate rating.
double estimate_elo(int solved, int total);

using SuiteCallback = std::function<void(const SuiteEntryResult&, int index, int total)>;
SuiteResult run_suite(Search& search, const std::vector<SuitePosition>& suite, int movetime_ms,
                      const std::atomic<bool>& stop, const SuiteCallback& cb = {});

} // namespace barracuda
