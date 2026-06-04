/// Configuration for a single scale level.
#[derive(Debug, Clone)]
pub struct ScaleConfig {
    pub population_size: usize,
    pub num_games: usize,
    pub num_strategies: usize,
    pub seed: u64,
}

impl ScaleConfig {
    pub fn new(population_size: usize, num_games: usize, num_strategies: usize, seed: u64) -> Self {
        Self { population_size, num_games, num_strategies, seed }
    }
}

/// Simple deterministic RNG (xorshift64) — no external deps.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Generate a value in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// A ternary agent with a strategy in {0, 1, 2}.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Agent {
    pub strategy: u8,
    pub fitness: f64,
}

impl Agent {
    pub fn new(strategy: u8) -> Self {
        Self { strategy, fitness: 0.0 }
    }
}

/// Result of a single game between two agents.
fn play_game(a_strategy: u8, b_strategy: u8, rng: &mut Rng) -> (f64, f64) {
    // Ternary rock-paper-scissors with noise:
    // 0 beats 1, 1 beats 2, 2 beats 0
    let noise = rng.next_f64() * 0.1;
    if a_strategy == (b_strategy + 1) % 3 {
        (1.0 + noise, 0.0 + noise)
    } else if b_strategy == (a_strategy + 1) % 3 {
        (0.0 + noise, 1.0 + noise)
    } else {
        (0.5 + noise, 0.5 + noise)
    }
}

/// A population of agents at one scale.
#[derive(Debug, Clone)]
pub struct Population {
    pub agents: Vec<Agent>,
}

impl Population {
    pub fn new(size: usize, num_strategies: usize, rng: &mut Rng) -> Self {
        let agents = (0..size)
            .map(|_| Agent::new(rng.next_usize(num_strategies) as u8))
            .collect();
        Self { agents }
    }

    /// Run all pairwise games and update fitness.
    pub fn run_games(&mut self, num_games: usize, rng: &mut Rng) {
        let n = self.agents.len();
        for _ in 0..num_games {
            let i = rng.next_usize(n);
            let j = rng.next_usize(n);
            if i == j {
                continue;
            }
            let (fit_i, fit_j) = play_game(self.agents[i].strategy, self.agents[j].strategy, rng);
            self.agents[i].fitness += fit_i;
            self.agents[j].fitness += fit_j;
        }
    }

    /// Evolve: select top 50% and replicate them (with mutation).
    pub fn evolve(&mut self, rng: &mut Rng, num_strategies: usize) {
        let half = self.agents.len() / 2;
        if half == 0 {
            return;
        }

        // Sort by fitness descending
        self.agents.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<Agent> = self.agents[..half].to_vec();

        // Replicate top agents, with 10% mutation chance
        for i in 0..half {
            let mut strat = top[i].strategy;
            if rng.next_f64() < 0.1 {
                strat = rng.next_usize(num_strategies) as u8;
            }
            self.agents[half + i] = Agent::new(strat);
        }

        // Reset fitness
        for agent in &mut self.agents {
            agent.fitness = 0.0;
        }
    }

    /// Count agents per strategy.
    pub fn strategy_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; 3];
        for a in &self.agents {
            counts[a.strategy as usize] += 1;
        }
        counts
    }

    /// Strategy distribution as proportions.
    pub fn strategy_distribution(&self) -> Vec<f64> {
        let total = self.agents.len() as f64;
        self.strategy_counts().iter().map(|&c| c as f64 / total).collect()
    }
}

/// Run a scaling experiment across multiple population sizes.
#[derive(Debug, Clone)]
pub struct ScalingExperiment {
    pub configs: Vec<ScaleConfig>,
    pub generations: usize,
}

impl ScalingExperiment {
    pub fn new(configs: Vec<ScaleConfig>, generations: usize) -> Self {
        Self { configs, generations }
    }

    /// Default configs matching the study: 24, 240, 2400, 24000.
    pub fn default_configs() -> Vec<ScaleConfig> {
        vec![
            ScaleConfig::new(24, 100, 3, 42),
            ScaleConfig::new(240, 1000, 3, 42),
            ScaleConfig::new(2400, 10000, 3, 42),
            ScaleConfig::new(24000, 100000, 3, 42),
        ]
    }

    /// Run the experiment and return final populations at each scale.
    pub fn run(&self) -> Vec<Population> {
        self.configs.iter().map(|cfg| {
            let mut rng = Rng::new(cfg.seed);
            let mut pop = Population::new(cfg.population_size, cfg.num_strategies, &mut rng);
            for _ in 0..self.generations {
                pop.run_games(cfg.num_games, &mut rng);
                pop.evolve(&mut rng, cfg.num_strategies);
            }
            pop
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = Rng::new(99);
        for _ in 0..1000 {
            let v = rng.next_usize(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn test_agent_new() {
        let a = Agent::new(2);
        assert_eq!(a.strategy, 2);
        assert_eq!(a.fitness, 0.0);
    }

    #[test]
    fn test_population_creation() {
        let mut rng = Rng::new(42);
        let pop = Population::new(100, 3, &mut rng);
        assert_eq!(pop.agents.len(), 100);
        for a in &pop.agents {
            assert!(a.strategy < 3);
        }
    }

    #[test]
    fn test_strategy_counts() {
        let agents = vec![Agent::new(0), Agent::new(0), Agent::new(1), Agent::new(2)];
        let pop = Population { agents };
        let counts = pop.strategy_counts();
        assert_eq!(counts, vec![2, 1, 1]);
    }

    #[test]
    fn test_strategy_distribution_sums_to_one() {
        let mut rng = Rng::new(7);
        let pop = Population::new(500, 3, &mut rng);
        let dist = pop.strategy_distribution();
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evolution_preserves_size() {
        let mut rng = Rng::new(11);
        let mut pop = Population::new(100, 3, &mut rng);
        pop.run_games(500, &mut rng);
        pop.evolve(&mut rng, 3);
        assert_eq!(pop.agents.len(), 100);
    }

    #[test]
    fn test_evolution_resets_fitness() {
        let mut rng = Rng::new(22);
        let mut pop = Population::new(100, 3, &mut rng);
        pop.run_games(500, &mut rng);
        pop.evolve(&mut rng, 3);
        for a in &pop.agents {
            assert_eq!(a.fitness, 0.0);
        }
    }
}
