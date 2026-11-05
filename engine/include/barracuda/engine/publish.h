// Barracuda - publish a trained network to the website (web/net) and optionally commit + push it.
// Shared by `barracuda-cli publish` and the dashboard's "Publish to website" button.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "barracuda/engine/match.h"

namespace barracuda {

struct PublishOptions {
    std::string net_path = "barracuda.bnn";
    std::string out_dir = "web/net";
    int games = 40;          // NNUE vs classical match that decides whether the site plays with the network
    int movetime_ms = 100;
    bool force_nnue = false; // use the network on the site even if it lost the match
    bool force_classical = false;
    std::string human_path = "human.mdl"; // how people choose moves (barracuda-cli train-human); shipped when present
};

struct PublishResult {
    bool ok = false;
    std::string error;
    bool has_net = false;
    bool use_nnue = false;
    double elo = 0.0, elo_error = 0.0;
    MatchStats match;
    std::uint32_t epochs = 0;
    std::uint64_t samples = 0;
    bool has_human = false;
};

using PublishLog = std::function<void(const std::string&)>;

// Copies the network next to the page and writes web/net/manifest.json.
PublishResult publish_network(const PublishOptions& opt, const std::atomic<bool>& stop, const PublishLog& log);

// Walks up from `start` to the folder containing .git (empty string when not inside a repository).
std::string find_repo_root(const std::string& start);

// git add <path> && git commit -m <message> && git push, run inside `repo_root`. Output goes to `log`.
bool git_commit_and_push(const std::string& repo_root, const std::string& path, const std::string& message,
                         const PublishLog& log);

} // namespace barracuda
