# Future Integration: population-scaling

## Current State
Studies how ternary agent dynamics change with population size (24 → 240 → 2,400 → 24,000 agents). Key findings: cluster count follows power-law (α ≈ 0.15), win rate std increases sub-linearly, entropy plateaus at ~82%. Jensen-Shannon divergence between scales decreases, indicating convergence.

## Integration Opportunities

### With room capacity planning
population-scaling's power-law (clusters ∝ N^0.15) predicts how many strategy clusters a room will have at any size. The entropy plateau (~82%) indicates the maximum strategic diversity a room can achieve regardless of cell count. These scaling laws inform room sizing: how many cells does a room need to be healthy? Answer: enough to reach the entropy plateau.

### With Forgemaster resource allocation
The Forgemaster uses population-scaling laws to size GPU allocations. A room with 1,000 cells needs X CUDA cores; a room with 100,000 cells needs Y cores. The sub-linear scaling of win rate std means larger rooms don't need proportionally more compute — the scaling is favorable.

### With conservation-matrix-rs
Conservation law 5 (avoidance ratio conserved across scales) is validated by population-scaling's scale sweeps. The scaling laws confirm that the fleet's dynamics are scale-invariant — what works at 24 agents works at 24,000.

## Dormant Ideas Now Unlockable
The scaling laws were empirical observations. Now they're engineering constraints: room capacity planning, GPU resource allocation, and fleet sizing all use population-scaling's power laws. The observations become the fleet's physics.

## Potential in Mature Systems
Every room is sized using population-scaling's laws. New rooms start at the minimum viable size (where entropy reaches ~80% of plateau), then scale up based on demand. The Forgemaster allocates GPU resources proportional to room size. The fleet operates at optimal scale.

## Cross-Pollination Ideas
- **strategy-ecology**: Species composition changes with scale
- **conservation-verify**: Scale sweeps verify conservation at fleet scale
- **forgemaster**: GPU allocation based on scaling laws

## Dependencies for Next Steps
- Room sizing guidelines based on entropy plateau
- GPU allocation formulas from scaling laws
- Fleet-scale simulation to validate scaling at 1M+ agents
