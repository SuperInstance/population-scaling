# Population Scaling

**Population scaling** studies how ternary agent dynamics — the {-1, 0, +1} action distributions — change as the number of agents grows. This crate runs systematic experiments that measure distribution stability, power-law convergence, and plateau detection across population sizes from 10 to 100,000+.

## Why It Matters

A key question in multi-agent systems: do the macroscopic properties of a ternary population remain stable as the population scales? If the avoidance ratio (η) converges to a fixed value at large N, then results from small-scale simulations transfer to production fleets. If it doesn't, then small-scale experiments are misleading. This crate answers that question empirically by running identical experiments at exponentially increasing population sizes (10, 100, 1000, 10000, ...) and measuring how distribution metrics — mean, variance, skewness, kurtosis — evolve. The findings directly inform fleet sizing decisions: at what population size do diminishing returns set in? When does the system plateau?

## How It Works

### Scaling Experiment Design

The experiment runs N agents, each making ternary choices (avoid/neutral/choose) according to a strategy. At each scale, we measure:

- **Distribution metrics**: mean μ, standard deviation σ, skewness γ₁, kurtosis β₂
- **Avoidance ratio**: R_avoid = N(-1) / N
- **Convergence**: how close R_avoid is to the large-N limit
- **Plateau**: the population size at which further scaling yields negligible change

### Power-Law Fitting

If a metric M follows a power law in population size:

```
M(N) = a · N^b + c
```

We fit the parameters (a, b, c) via least-squares regression on log-transformed data:

```
log|M(N) - c| = log|a| + b · log(N)
```

The exponent b reveals scaling behavior:
- b < 0: metric improves (shrinks) with scale
- b ≈ 0: metric is scale-invariant (plateau reached)
- b > 0: metric worsens with scale

### Plateau Detection

A plateau is reached when the rate of change drops below a threshold:

```
plateau_at = min{ N : |dM/dN| < ε  for all subsequent N' > N }
```

The `PlateauDetector` uses a sliding window over consecutive scale measurements to identify where the derivative falls below ε, using O(window_size) memory.

### Divergence Metrics

Cross-scale comparison measures how much the distribution at scale N differs from scale 2N:

```
D(N) = D_KL(P_N || P_{2N})
```

where D_KL is the Kullback-Leibler divergence. When D(N) → 0, the distribution has converged.

## Quick Start

```rust
use population_scaling::{ScalingExperiment, ScaleConfig};

fn main() {
    let config = ScaleConfig {
        scales: vec![10, 100, 1000, 10000],
        strategy_length: 32,
        trials_per_scale: 100,
    };

    let experiment = ScalingExperiment::new(config);
    let report = experiment.run();

    println!("Scaling Report:");
    for metric in &report.metrics {
        println!("  N={}: mean={:.4}, std={:.4}, avoid_ratio={:.4}",
                 metric.population, metric.mean, metric.std_dev, metric.avoid_ratio);
    }

    if let Some(plateau) = report.plateau {
        println!("Plateau detected at N = {}", plateau.population);
    }

    if let Some(law) = report.power_law_fit {
        println!("Power law: {:.3} * N^{:.3} + {:.3}", law.a, law.b, law.c);
    }
}
```

```bash
cargo build
cargo run --release  # release mode for large populations
```

## API

| Type | Description |
|------|-------------|
| `ScaleConfig` | Experiment parameters (scales, strategy length, trials) |
| `ScalingExperiment` | Runs experiments across configured scales |
| `MetricsAtScale` | Mean, std, skewness, kurtosis, avoid ratio at one scale |
| `DistributionMetrics` | Full distribution summary for one population |
| `PowerLawFit` | Fitted parameters (a, b, c) of the scaling law |
| `PlateauDetector` | Sliding-window derivative threshold detector |
| `PlateauResult` | Plateau population and confidence |
| `ScaleReport` | Complete experiment results with all metrics |
| `CrossScaleComparison` | KL-divergence between consecutive scales |

## Architecture Notes

Population Scaling validates the SuperInstance thesis that γ + η = C holds at arbitrary scale. If the avoidance ratio (η) and constructive ratio (γ) remain stable as population grows, then C — the fleet's aggregate competence — scales linearly with N. The power-law exponent directly measures this: b ≈ 0 confirms scale-invariance. The plateau tells us the minimum viable fleet size for stable behavior. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Newman, M. E. J. (2005). "Power laws, Pareto distributions and Zipf's law." *Contemporary Physics*, 46(5), 323–351.
2. Vicsek, T. (2001). "Complexity: The Bigger Picture." *Nature*, 411, 421.
3. Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media. — On scaling in cellular automata and rule-based systems.

## License

MIT
