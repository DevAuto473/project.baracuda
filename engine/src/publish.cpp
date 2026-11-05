#include "barracuda/engine/publish.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "barracuda/engine/suite.h"
#include "barracuda/nnue/network.h"

#ifdef _WIN32
#define BRC_POPEN _popen
#define BRC_PCLOSE _pclose
#else
#define BRC_POPEN popen
#define BRC_PCLOSE pclose
#endif

namespace barracuda {

namespace fs = std::filesystem;

PublishResult publish_network(const PublishOptions& opt, const std::atomic<bool>& stop, const PublishLog& log) {
    PublishResult res;
    auto say = [&](const std::string& s) { if (log) log(s); };
    std::error_code ec;
    const fs::path out = opt.out_dir;
    fs::create_directories(out, ec);
    if (ec) {
        res.error = "cannot create " + out.string() + ": " + ec.message();
        return res;
    }

    std::shared_ptr<const nnue::Network> net;
    if (fs::exists(opt.net_path, ec)) {
        std::string err;
        net = nnue::load_network(opt.net_path, &err);
        if (!net) {
            res.error = "cannot load " + opt.net_path + ": " + err;
            return res;
        }
        res.has_net = true;
        res.epochs = net->epochs_trained();
        res.samples = net->samples_trained();
    }

    // anchor: tactical-suite estimate of the classical evaluator
    Search classical(32);
    classical.set_eval_mode(EvalMode::CLASSICAL);
    say("calibrating the classical evaluator on the tactical suite ...");
    const SuiteResult cal = run_suite(classical, builtin_wac_suite(), 150, stop);
    if (stop.load()) { res.error = "stopped"; return res; }
    res.elo = cal.elo_estimate;

    if (net && opt.games > 0) {
        MatchConfig cfg;
        cfg.games = opt.games;
        cfg.movetime_ms = opt.movetime_ms;
        Search nn(16), cl(16);
        nn.set_network(net);
        nn.set_eval_mode(EvalMode::NNUE);
        cl.set_eval_mode(EvalMode::CLASSICAL);
        say("match: " + std::to_string(cfg.games) + " games, network vs classical evaluator ...");
        MatchCallbacks cb;
        cb.on_game_end = [&](const MatchStats& m, const std::string&) {
            say("  game " + std::to_string(m.games_done) + "/" + std::to_string(cfg.games) + "  +" + std::to_string(m.wins) +
                " =" + std::to_string(m.draws) + " -" + std::to_string(m.losses));
        };
        run_match(nn, cl, cfg, stop, cb, &res.match);
        if (stop.load()) { res.error = "stopped"; return res; }
        if (res.match.games_done > 0 && res.match.elo.elo > 0.0) {
            res.use_nnue = true;
            res.elo = cal.elo_estimate + res.match.elo.elo;
            res.elo_error = res.match.elo.error;
        }
    }
    if (net && opt.force_nnue) res.use_nnue = true;
    if (opt.force_classical) res.use_nnue = false;

    const fs::path dest = out / "barracuda.bnn";
    if (net) { // always shipped: the page draws it; `eval` says whether it also plays
        fs::copy_file(opt.net_path, dest, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            res.error = "cannot copy the network: " + ec.message();
            return res;
        }
    } else {
        fs::remove(dest, ec);
    }

    const std::time_t now = std::time(nullptr);
    char iso[32];
    std::strftime(iso, sizeof(iso), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    const bool measured = res.use_nnue && res.match.games_done > 0;
    const std::string basis = measured ? "classical anchor (tactical suite) + measured match advantage"
                              : res.use_nnue ? "network not measured (forced); Elo shown is the classical anchor"
                                             : "tactical-suite estimate of the classical evaluator";
    std::ostringstream js;
    js << "{\n"
       << "  \"version\": " << (long long)now << ",\n"
       << "  \"updated\": \"" << iso << "\",\n"
       << "  \"eval\": \"" << (res.use_nnue ? "nnue" : "classical") << "\",\n"
       << "  \"net\": " << (net ? "\"barracuda.bnn\"" : "null") << ",\n"
       << "  \"epochs\": " << res.epochs << ",\n"
       << "  \"samples\": " << (unsigned long long)res.samples << ",\n"
       << "  \"elo\": " << (long long)std::llround(res.elo) << ",\n"
       << "  \"elo_error\": " << (long long)std::llround(res.elo_error) << ",\n"
       << "  \"elo_basis\": \"" << basis << "\",\n"
       << "  \"match\": {\"games\": " << res.match.games_done << ", \"wins\": " << res.match.wins
       << ", \"draws\": " << res.match.draws << ", \"losses\": " << res.match.losses << "}\n"
       << "}\n";
    std::ofstream mf(out / "manifest.json");
    mf << js.str();
    if (!mf) {
        res.error = "cannot write manifest.json";
        return res;
    }
    res.ok = true;
    return res;
}

std::string find_repo_root(const std::string& start) {
    std::error_code ec;
    fs::path p = fs::absolute(start, ec);
    for (int i = 0; i < 12 && !p.empty(); ++i) {
        if (fs::exists(p / ".git", ec)) return p.string();
        if (p == p.parent_path()) break;
        p = p.parent_path();
    }
    return {};
}

namespace {
int run(const std::string& cmd, const PublishLog& log) {
    FILE* f = BRC_POPEN((cmd + " 2>&1").c_str(), "r");
    if (!f) {
        if (log) log("cannot run: " + cmd);
        return -1;
    }
    char buf[512];
    while (std::fgets(buf, sizeof(buf), f)) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (!line.empty() && log) log("  " + line);
    }
    return BRC_PCLOSE(f);
}
std::string quote(const std::string& s) {
    std::string q = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\' || c == '$' || c == '`') q += '\\';
        q += c;
    }
    return q + "\"";
}
} // namespace

bool git_commit_and_push(const std::string& repo_root, const std::string& path, const std::string& message,
                         const PublishLog& log) {
    const std::string git = "git -C " + quote(repo_root) + " ";
    if (run(git + "add -- " + quote(path), log) != 0) return false;
    // nothing staged: no new commit, but still push anything that is waiting
    if (run(git + "diff --cached --quiet", nullptr) != 0) {
        if (run(git + "commit -m " + quote(message), log) != 0) return false;
    } else if (log) {
        log("no changes in " + path + " since the last publish");
    }
    return run(git + "push", log) == 0;
}

} // namespace barracuda
