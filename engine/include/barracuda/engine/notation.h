// Barracuda - move notation helpers (UCI long algebraic and SAN).
#pragma once

#include <string>
#include <vector>

#include "barracuda/engine/position.h"

namespace barracuda {

// Returns Move::none() if the string is not a legal move in `pos`.
Move parse_uci_move(const Position& pos, const std::string& uci);
Move parse_san_move(const Position& pos, const std::string& san);

std::string move_to_san(Position& pos, Move m); // pos is restored before return
std::string pv_to_san(const Position& pos, const std::vector<Move>& pv);

enum class GameResult { ONGOING, WHITE_WINS, BLACK_WINS, DRAW };
// Game-level adjudication using full history (three-fold, 50-move, mate, stalemate, material).
GameResult game_result(const Position& pos, std::string* reason = nullptr);

} // namespace barracuda
