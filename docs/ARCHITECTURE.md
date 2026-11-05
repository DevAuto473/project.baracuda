# Barracuda internals

## 1. Board representation (`engine/`)

- `Square`: a1 = 0 … h8 = 63. `Piece = colour * 6 + type`, and `NO_PIECE = 12`.
- `Position` stores `by_type_[6]` and `by_color_[2]` bitboards, plus a `board_[64]` mailbox for O(1) "what is on this square" queries.
- Every move pushes a `StateInfo` holding the Zobrist key, castling rights, en passant square, rule-50 counter, captured piece, checkers and a **`DirtyPiece` record**. That record lists up to three (piece, from, to) changes, which the NNUE uses for incremental updates. `undo_move()` pops the state.
- The en passant square is stored (and hashed) only when a capture is actually possible. This keeps hashes canonical, so repetition detection works.
- `Move` is 16 bits: `from | to << 6 | flags << 12`, using the chessprogramming.org flag scheme (quiet, double push, castles, capture, ep, 4 promotions × capture bit).

### Attack generation
Knight, king and pawn attacks come from lookup tables. Sliders use **fancy magic bitboards**: the magics are found at startup by a deterministic sparse-random search (seeded per rank), which takes a few milliseconds. `BETWEEN[a][b]` includes `b`, so a check evasion target is simply `between(king, checker)`.

### Legal move generation
1. The king's danger map is computed **with the king removed** from occupancy, so the king cannot "hide" behind itself along a slider ray.
2. Double check: only king moves are generated.
3. Single check: every other piece is restricted to `between(king, checker)`, which includes capturing the checker.
4. Pins: x-ray snipers from the king square; a lone friendly blocker is pinned and may move only along `line(king, piece)`.
5. En passant: the capture is simulated on a copy of the occupancy and the king is tested against sliders, knights and pawns. This catches the classic horizontal pin `K P p r` on the 5th rank.
6. Castling: no check, empty path, and the transit and destination squares are not attacked.

Perft positions 1–6 from chessprogramming.org are part of the unit tests.

## 2. Search (`engine/src/search.cpp`)

```
go():  for depth = 1..:
         aspiration window around the previous score (±25, widening by 1.5x + 10)
         negamax(alpha, beta, depth, ply=0)
         stop if: hard time limit / node limit / external stop, or soft limit * 0.6 reached
negamax: check extension -> qsearch at depth <= 0 -> draw/mate-distance -> TT cutoff
         -> static eval (TT eval reused) -> RFP -> null move (R = 3 + d/4 + ...)
         -> IIR -> ordered move loop with LMP / futility / SEE pruning
         -> PVS + LMR (log*log table, adjusted by PV/improving/killer/check/history)
         -> killers + history (gravity update, malus for quiets that failed) -> TT store
qsearch: stand pat, captures + queen promotions ordered by MVV-LVA,
         delta pruning, SEE >= 0 filter, check evasions when in check
```

The TT has 2-way buckets of 16-byte entries (32-bit key check, move, score, static eval, depth, generation/bound). Slot 0 is depth-preferred and slot 1 always-replace. Mate scores are stored relative to the node (`score_to_tt` / `score_from_tt`).

## 3. NNUE (`nnue/`)

### Topology
```
features(perspective) = colour_rel * 384 + piece_type * 64 + square_rel     (768 per perspective)
acc[c]   = ft_b + Σ ft_w[f]               int16, c ∈ {white, black}
x        = [CReLU(acc[stm]), CReLU(acc[!stm])]     uint8 [0,127], 512 values
h1       = CReLU((W1·x + b1) >> 6)        int8 weights, int32 sums, 32 outputs
h2       = CReLU((W2·h1 + b2) >> 6)       32 outputs
out      = W3·h2 + b3;  cp = out * 400 / (127 * 64)
```

### Quantisation

| tensor | float range | scale | stored as |
|---|---|---|---|
| FT weights/bias | ±4 | ×127 | int16 |
| L1/L2/output weights | ±1.98 | ×64 | int8 |
| dense biases | – | ×127·64 | int32 |

Because training uses `clamp(x, 0, 1)` in float space, the quantised `clamp(x, 0, 127)` is the same function and the only error comes from rounding. `_mm256_maddubs_epi16` saturates at 32767, and 127·127·2 = 32258 stays below that, so no overflow is possible.

### Incremental updates
`AccumulatorStack` mirrors the search's make/unmake. After `do_move`, `push(net, dirty)` computes `next = prev + Σ add_rows − Σ sub_rows` for both perspectives in one pass over 16 AVX2 registers. `do_null_move` just copies the accumulator. Unit tests compare incremental and full-refresh accumulators bit for bit across 20 random games.

### Training
- Per-sample loss is `(σ(out) − t)²` with `t = λ·σ(cp_stm/400) + (1−λ)·wdl_stm`.
- Backprop runs through the clipped activations (the gradient is 1 on the open interval (0,1) and 0 elsewhere).
- FT gradients are sparse: each thread records which feature rows it touched and only zeroes and reduces those.
- The optimiser is Adam (β = 0.9/0.999) with learning-rate decay after each epoch. Weights are clamped to their quantisable ranges after every step.
- The validation split is the shuffled tail (3 %, at most 200K samples).

## 4. Threading model (`gui/src/runtime/`)

| Data | Producer | Consumer | Mechanism |
|---|---|---|---|
| commands (Go, Inspect, SelfPlay, Train, …) | render thread | worker | `std::deque` + mutex + condition variable (the render thread only pushes; lock hold time is a few instructions) |
| `SearchSnapshot`, `MoveResult`, `NetworkView` | search worker | render | `TripleBuffer<T>` (lock-free, wait-free reader) |
| `TrainingSnapshot`, `ObservationSnapshot`, `NetworkView` | training worker | render | `TripleBuffer<T>` |
| log lines | any thread | render | `MpmcQueue<LogLine>` (Vyukov bounded ring, fixed 256-byte lines) |
| quantised network | training worker | both workers | `NetworkRegistry` (`shared_ptr<const Network>` swap) |
| stop requests | render | workers | `std::atomic<bool>` polled every 2048 nodes / batch / move |

`TripleBuffer` keeps three slots: the writer owns `back`, the reader owns `front`, and an atomic byte holds the `middle` index plus a *dirty* bit. `publish()` swaps back↔middle and sets the dirty bit; `update()` swaps middle↔front only if the bit is set. Neither side ever waits, and the reader always gets the newest complete snapshot.

Stale engine replies are discarded by request id. That handles undo, new game, or switching to observation while a search is still running.
