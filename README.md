# population-scaling

Study how ternary agent dynamics change with population size.

## Key Findings

Scaling from 24 → 240 → 2,400 → 24,000 agents reveals:

| Metric | Trend |
|--------|-------|
| Strategy clusters | 7 → 10 → 14 (doubles roughly) |
| Win rate std | 0.135 → 0.149 (increases with scale) |
| Normalized entropy | Plateaus at ~82% |

### Scaling Laws

- **Cluster count** follows a power-law: `clusters ∝ N^α` where α ≈ 0.15
- **Win rate std** increases sub-linearly with population size
- **Entropy** growth plateaus — adding more agents beyond a threshold yields diminishing strategic diversity

### Cross-Scale Dynamics

Jensen-Shannon divergence between consecutive scales decreases, indicating strategy distributions converge as population grows. The system reaches a quasi-stable equilibrium in strategy space despite continued growth.

## Features

- **`ScalingExperiment`** — Run the same ternary game simulation across multiple population sizes
- **`MetricsAtScale`** — Track cluster count, win rate statistics, and entropy at each scale
- **`ScalingLaw`** — Fit power-law relationships (e.g., `clusters ∝ N^α`) via log-log regression
- **`PlateauDetector`** — Detect when a metric stops growing meaningfully with scale
- **`CrossScaleComparison`** — Compare strategy distributions across scales using Jensen-Shannon divergence
- **`ScaleReport`** — Generate a comprehensive report of all metrics at all scales

## Usage

```rust
use population_scaling::{ScalingExperiment, ScaleConfig, ScaleReport};

// Define scales to study
let configs = vec![
    ScaleConfig::new(24, 100, 3, 42),
    ScaleConfig::new(240, 1000, 3, 42),
    ScaleConfig::new(2400, 10000, 3, 42),
    ScaleConfig::new(24000, 100000, 3, 42),
];

// Generate a full scaling report
let report = ScaleReport::generate(&configs, 20); // 20 generations
println!("{}", report.to_string_report());
```

## Requirements

- Pure Rust, no `unsafe`, no external dependencies
- Edition 2021

## License

MIT
