# Macroxue step valuations

This document explains the `valuation_type` field that appears in self-play step logs under `selfplay_logs/`. Each logged step records how the move was chosen, which heuristic supplied the numeric score that was stored as `valuation`, and what the per-branch expectations (`branch_evs`) report.

## Field summary
- `move` is the action that was actually played after evaluating all candidates.
- `valuation_type` identifies which policy or heuristic supplied the `valuation` that selected the move.
- `valuation` is the scalar returned by that policy. For tuple/policy planners it is a probability of eventually satisfying their goal. For the search backend it is an expectimax-style heuristic score scaled by `1/1000`, and can therefore be negative.
- `branch_evs` mirrors the per-move expectation used for tie-breaking/debugging. Entries are `null` when a branch is illegal (the board does not change for that move).

## Possible valuation types

### `tuple11`
- **Source**: `Tuple11` pattern policy (`tuple.h`). It looks up a pre-trained 11-tile pattern and produces a probability (`0.0 – 1.0`) that the board remains in a “regular” high-tile funnel state.
- **When used**: First tier fallback in `SuggestMove`. If the lookup probability is at least `0.9` (or `>0` when `BIG_TUPLES` is enabled) it is accepted and the search is skipped.
- **Branch EVs**: Recomputed with a one-step rollout (`RollOutWithTuple`) that averages over all empty-tile insertions weighted by tile probabilities.

### `tuple10`
- **Source**: `Tuple10` pattern policy. Similar to `tuple11` but defined on a slightly smaller 10-tile template so it applies in more mid-game positions. Returns probabilities in `[0.0, 1.0]`.
- **When used**: Second tier fallback, triggered whenever `tuple11` declined to act and `Tuple10::SuggestMove` reports a non-zero probability.
- **Branch EVs**: Also produced by `RollOutWithTuple`.

### `line_plan`
- **Source**: `LinePlan` deterministic planner (`plan.h`). It enumerates future moves until a predefined alignment goal is reached, caching the results in a compact map.
- **When used**: Only when tuple policies are disabled (`options.tuple_moves=false`) and the board matches the plan’s applicability conditions.
- **Valuation range**: Probability `[0.0, 1.0]` that the plan will succeed before leaving its valid region.

> NOTE: this will never show up in the actual selfplay data. Do not plan around it.

### `block_plan`
- **Source**: `BlockPlan` planner (`plan.h`). It covers configurations where the anchor block is near the top-left and tries to consolidate high tiles.
- **When used**: Only when tuple policies are disabled and the board passes `BlockPlan::IsApplicable`. Acts before the search fallback.
- **Valuation range**: Probability `[0.0, 1.0]` of reaching the planner’s goal configuration.

> NOTE: this will never happen in the actual selfplay data. Do not plan around it.

### `search`
- **Source**: The depth-limited expectimax search (`Node::Search` in `node.h`). It explores the move tree using `TryAllMoves`/`TryAllTiles`, caching board evaluations and pruning by `pass_score`/`game_over_score` thresholds.
- **Valuation range**: Raw expectimax scores (integers) divided by `1000.0`. Because the underlying heuristic is centred on line/column monotonicity scores, values can be negative or exceed `1`.
- **Branch EVs**: Taken directly from the per-move expectimax scores (also scaled by `1/1000`).

### Legacy entries without `valuation_type`
Some older logs omit the `valuation_type` key entirely. These records were produced by a build prior to [`ValuationTypeName` logging] and should be interpreted as the `search` fallback (the only generator at that time).

## Decision flow vs. vanilla expectimax

The move selector is not a pure expectimax agent:
- It **gates the search** behind tuple/planner policies. When either tuple policy is confident enough, or a plan applies, no lookahead is executed. Vanilla expectimax would evaluate every position with the same depth-limited tree search.
- **Tuple heuristics** supply probabilities learned offline (stored in `tuple_moves.*` tables). Expectimax typically relies on hand-crafted evaluation functions; here the learned policy can bypass search entirely.
- The **expectimax backend is modified** with cached boards, adaptive pruning (`pass_score`/`game_over_score` depend on the current max tile), and optional tuple-based rollouts for branch diagnostics. These tweaks bias the heuristic scores away from raw expected score and towards “probability of survival” around strategic anchor tiles.
- When tuple policies are disabled, **block/line planners** take the place of the tuple nets, yielding deterministic sub-strategies that vanilla expectimax does not model.

## Example log snippets
All taken from `selfplay_logs/d2_test_v2/depth02_worker00_seed0710363253_game000000.jsonl.gz`.

```json
{"step_index":0,"move":"left","valuation_type":"search","valuation":0.034000,
 "branch_evs":{"up":0.033000,"left":0.034000,"right":0.013000,"down":0.011000}}
```
Expectimax search selects `left`; scores reflect the heuristic line/column monotonicity estimate.

```json
{"step_index":262,"move":"left","valuation_type":"tuple11","valuation":0.810125,
 "branch_evs":{"up":0.797134,"left":0.810185,"right":0.0,"down":0.0}}
```
A high-confidence Tuple11 lookup drives the move choice. Branch EVs reuse tuple rollouts; illegal directions are reported as `0.0` after filtering.

```json
{"step_index":2440,"move":"down","valuation_type":"tuple10","valuation":0.099000,
 "branch_evs":{"up":0.099002,"left":0.098364,"right":0.0,"down":0.0}}
```
Later in the game the broader Tuple10 pattern applies; probabilities remain sub-1.0 because the board is near its stability threshold.

```json
{"step_index":0,"move":"up","valuation":0.035000,
 "branch_evs":{"up":0.035000,"left":0.028000,"right":-0.022000,"down":0.010000}}
```
An older log (same directory, file `depth02_worker00_seed1781025990_game000000.jsonl.gz`) that predates `valuation_type`. Treat it as a `search` valuation.

## Practical notes
- `branch_evs` use move names `up/left/right/down`, matching `Board::move_names`.
- Illegal moves remain in the JSON with a `null` EV when generated by the search fallback; tuple-based logging reports `0.0` instead because the rollout simply never accumulates probability for that branch.
- The tuple tables are persisted across runs (`tuple_moves.10a`, `tuple_moves.11a`). Their calibration determines how often the search is skipped, which is why their valuations appear as probabilities rather than heuristic scores.
