use crate::scaling::Population;

/// Distribution-level metrics for one scale.
#[derive(Debug, Clone)]
pub struct DistributionMetrics {
    pub cluster_count: usize,
    pub win_rate_mean: f64,
    pub win_rate_std: f64,
    pub entropy: f64,
    pub max_entropy: f64,
    pub normalized_entropy: f64,
}

impl DistributionMetrics {
    /// Compute metrics from a population.
    pub fn from_population(pop: &Population) -> Self {
        let dist = pop.strategy_distribution();

        // Cluster count: strategies with >5% representation
        let threshold = 0.05;
        let cluster_count = dist.iter().filter(|&&p| p > threshold).count();

        // Win rates per strategy (fitness proxy from distribution weight)
        let fitnesses: Vec<f64> = pop.agents.iter().map(|a| a.fitness).collect();
        let n = fitnesses.len() as f64;
        let win_rate_mean = if n > 0.0 { fitnesses.iter().sum::<f64>() / n } else { 0.0 };
        let variance = if n > 0.0 {
            fitnesses.iter().map(|f| (f - win_rate_mean).powi(2)).sum::<f64>() / n
        } else {
            0.0
        };
        let win_rate_std = variance.sqrt();

        // Shannon entropy of strategy distribution
        let mut entropy = 0.0;
        for &p in &dist {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        let num_strategies = dist.len();
        let max_entropy = if num_strategies > 0 { (num_strategies as f64).ln() } else { 0.0 };
        let normalized_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };

        Self {
            cluster_count,
            win_rate_mean,
            win_rate_std,
            entropy,
            max_entropy,
            normalized_entropy,
        }
    }
}

/// Metrics tracked at each population scale.
#[derive(Debug, Clone)]
pub struct MetricsAtScale {
    pub population_size: usize,
    pub metrics: DistributionMetrics,
}

impl MetricsAtScale {
    pub fn new(population_size: usize, pop: &Population) -> Self {
        Self {
            population_size,
            metrics: DistributionMetrics::from_population(pop),
        }
    }

    /// Extract cluster counts across all scales.
    pub fn cluster_counts(scales: &[MetricsAtScale]) -> Vec<(usize, usize)> {
        scales.iter().map(|s| (s.population_size, s.metrics.cluster_count)).collect()
    }

    /// Extract win rate stds across all scales.
    pub fn win_rate_stds(scales: &[MetricsAtScale]) -> Vec<(usize, f64)> {
        scales.iter().map(|s| (s.population_size, s.metrics.win_rate_std)).collect()
    }

    /// Extract normalized entropies across all scales.
    pub fn normalized_entropies(scales: &[MetricsAtScale]) -> Vec<(usize, f64)> {
        scales.iter().map(|s| (s.population_size, s.metrics.normalized_entropy)).collect()
    }
}

// Helper needed from scaling module
fn _strategy_distribution(pop: &Population) -> Vec<f64> {
    pop.strategy_distribution()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaling::{Population, Rng};

    fn make_pop(size: usize, seed: u64) -> Population {
        let mut rng = Rng::new(seed);
        let mut pop = Population::new(size, 3, &mut rng);
        pop.run_games(size * 5, &mut rng);
        pop
    }

    #[test]
    fn test_cluster_count_with_uniform() {
        // Create a perfectly uniform population
        let agents: Vec<crate::scaling::Agent> = (0..30)
            .map(|i| crate::scaling::Agent::new((i % 3) as u8))
            .collect();
        let pop = Population { agents };
        let m = DistributionMetrics::from_population(&pop);
        assert_eq!(m.cluster_count, 3);
    }

    #[test]
    fn test_cluster_count_with_dominant() {
        let agents: Vec<crate::scaling::Agent> = (0..100)
            .map(|_| crate::scaling::Agent::new(0))
            .collect();
        let pop = Population { agents };
        let m = DistributionMetrics::from_population(&pop);
        assert_eq!(m.cluster_count, 1);
    }

    #[test]
    fn test_entropy_uniform_is_max() {
        let agents: Vec<crate::scaling::Agent> = (0..300)
            .map(|i| crate::scaling::Agent::new((i % 3) as u8))
            .collect();
        let pop = Population { agents };
        let m = DistributionMetrics::from_population(&pop);
        assert!((m.normalized_entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_dominated_is_low() {
        let agents: Vec<crate::scaling::Agent> = (0..100)
            .map(|_| crate::scaling::Agent::new(0))
            .collect();
        let pop = Population { agents };
        let m = DistributionMetrics::from_population(&pop);
        assert!(m.normalized_entropy < 0.01);
    }

    #[test]
    fn test_metrics_at_scale() {
        let pop = make_pop(240, 42);
        let ms = MetricsAtScale::new(240, &pop);
        assert_eq!(ms.population_size, 240);
        assert!(ms.metrics.cluster_count > 0);
    }

    #[test]
    fn test_extraction_helpers() {
        let pop1 = make_pop(24, 42);
        let pop2 = make_pop(240, 42);
        let scales = vec![
            MetricsAtScale::new(24, &pop1),
            MetricsAtScale::new(240, &pop2),
        ];
        let cc = MetricsAtScale::cluster_counts(&scales);
        assert_eq!(cc.len(), 2);
        let wr = MetricsAtScale::win_rate_stds(&scales);
        assert_eq!(wr.len(), 2);
        let ne = MetricsAtScale::normalized_entropies(&scales);
        assert_eq!(ne.len(), 2);
    }
}
