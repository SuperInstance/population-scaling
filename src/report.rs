use crate::scaling::{ScalingExperiment, ScaleConfig};
use crate::metrics::MetricsAtScale;
use crate::law::{ScalingLaw, PowerLawFit};
use crate::plateau::{PlateauDetector, PlateauResult};
use crate::divergence::CrossScaleComparison;

/// Comprehensive scaling report.
#[derive(Debug)]
pub struct ScaleReport {
    pub population_sizes: Vec<usize>,
    pub metrics: Vec<MetricsAtScale>,
    pub cluster_fit: PowerLawFit,
    pub std_fit: PowerLawFit,
    pub entropy_plateau: PlateauResult,
    pub cross_scale: CrossScaleComparison,
}

impl ScaleReport {
    /// Generate a full report from experiment configs and generation count.
    pub fn generate(configs: &[ScaleConfig], generations: usize) -> Self {
        let experiment = ScalingExperiment::new(configs.to_vec(), generations);
        let populations = experiment.run();

        let population_sizes: Vec<usize> = configs.iter().map(|c| c.population_size).collect();

        // Compute metrics at each scale
        let metrics: Vec<MetricsAtScale> = population_sizes.iter()
            .zip(populations.iter())
            .map(|(&size, pop)| MetricsAtScale::new(size, pop))
            .collect();

        // Fit scaling laws
        let cluster_counts: Vec<usize> = metrics.iter().map(|m| m.metrics.cluster_count).collect();
        let win_rate_stds: Vec<f64> = metrics.iter().map(|m| m.metrics.win_rate_std).collect();
        let cluster_fit = ScalingLaw::fit_cluster_scaling(&population_sizes, &cluster_counts);
        let std_fit = ScalingLaw::fit_win_rate_std_scaling(&population_sizes, &win_rate_stds);

        // Detect plateau in normalized entropy
        let entropy_values: Vec<(usize, f64)> = metrics.iter()
            .map(|m| (m.population_size, m.metrics.normalized_entropy))
            .collect();
        let detector = PlateauDetector::default();
        let entropy_plateau = detector.detect("normalized_entropy", &entropy_values);

        // Cross-scale comparison
        let distributions: Vec<Vec<f64>> = populations.iter()
            .map(|p| p.strategy_distribution())
            .collect();
        let cross_scale = CrossScaleComparison::from_distributions(&population_sizes, &distributions);

        Self {
            population_sizes,
            metrics,
            cluster_fit,
            std_fit,
            entropy_plateau,
            cross_scale,
        }
    }

    /// Format the report as a human-readable string.
    pub fn to_string_report(&self) -> String {
        let mut lines = Vec::new();

        lines.push("=" .repeat(60).to_string());
        lines.push("  POPULATION SCALING REPORT".to_string());
        lines.push("=".repeat(60).to_string());
        lines.push(String::new());

        // Per-scale metrics
        lines.push("Per-Scale Metrics:".to_string());
        lines.push("-".repeat(40).to_string());
        for m in &self.metrics {
            lines.push(format!(
                "  N={:>6} | clusters={:>2} | win_rate_std={:.4} | entropy={:.4} ({:.1}%)",
                m.population_size,
                m.metrics.cluster_count,
                m.metrics.win_rate_std,
                m.metrics.entropy,
                m.metrics.normalized_entropy * 100.0,
            ));
        }
        lines.push(String::new());

        // Scaling laws
        lines.push("Scaling Laws:".to_string());
        lines.push("-".repeat(40).to_string());
        lines.push(format!("  Cluster count: {}", self.cluster_fit.description));
        lines.push(format!("    Quality: {}", self.cluster_fit.quality()));
        lines.push(format!("  Win rate std:  {}", self.std_fit.description));
        lines.push(format!("    Quality: {}", self.std_fit.quality()));
        lines.push(String::new());

        // Plateau detection
        lines.push("Plateau Detection:".to_string());
        lines.push("-".repeat(40).to_string());
        lines.push(format!("  {}", self.entropy_plateau.summary()));
        if !self.entropy_plateau.growth_rates.is_empty() {
            lines.push("  Growth rates:".to_string());
            for (i, &rate) in self.entropy_plateau.growth_rates.iter().enumerate() {
                lines.push(format!("    {} → {}: {:.4} ({:.1}%)",
                    self.population_sizes.get(i).unwrap_or(&0),
                    self.population_sizes.get(i + 1).unwrap_or(&0),
                    rate, rate * 100.0));
            }
        }
        lines.push(String::new());

        // Cross-scale comparison
        lines.push("Cross-Scale Comparison:".to_string());
        lines.push("-".repeat(40).to_string());
        lines.push(self.cross_scale.summary());

        lines.push(String::new());
        lines.push("=".repeat(60).to_string());

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaling::ScaleConfig;

    #[test]
    fn test_report_generation() {
        let configs = vec![
            ScaleConfig::new(24, 50, 3, 42),
            ScaleConfig::new(240, 500, 3, 42),
            ScaleConfig::new(2400, 5000, 3, 42),
        ];
        let report = ScaleReport::generate(&configs, 10);
        assert_eq!(report.population_sizes.len(), 3);
        assert_eq!(report.metrics.len(), 3);
    }

    #[test]
    fn test_report_string_output() {
        let configs = vec![
            ScaleConfig::new(24, 50, 3, 42),
            ScaleConfig::new(240, 500, 3, 42),
        ];
        let report = ScaleReport::generate(&configs, 5);
        let s = report.to_string_report();
        assert!(s.contains("POPULATION SCALING REPORT"));
        assert!(s.contains("Scaling Laws"));
        assert!(s.contains("Plateau"));
        assert!(s.contains("Cross-Scale"));
    }

    #[test]
    fn test_report_cluster_fit_is_populated() {
        let configs = vec![
            ScaleConfig::new(24, 50, 3, 42),
            ScaleConfig::new(240, 500, 3, 42),
            ScaleConfig::new(2400, 5000, 3, 42),
        ];
        let report = ScaleReport::generate(&configs, 10);
        // Should have a non-zero description
        assert!(!report.cluster_fit.description.is_empty());
    }
}
