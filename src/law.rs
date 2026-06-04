/// A fitted power-law: y = a * x^b.
#[derive(Debug, Clone)]
pub struct PowerLawFit {
    pub exponent: f64,
    pub coefficient: f64,
    pub r_squared: f64,
    pub description: String,
}

impl PowerLawFit {
    pub fn predict(&self, x: f64) -> f64 {
        self.coefficient * x.powf(self.exponent)
    }

    /// Quality of fit: 0.0 = none, 1.0 = perfect.
    pub fn quality(&self) -> &'static str {
        if self.r_squared > 0.95 {
            "excellent"
        } else if self.r_squared > 0.8 {
            "good"
        } else if self.r_squared > 0.5 {
            "moderate"
        } else {
            "poor"
        }
    }
}

/// Fit power-law relationships between population size and metrics.
#[derive(Debug, Clone)]
pub struct ScalingLaw {
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
}

impl ScalingLaw {
    pub fn new(x_values: Vec<f64>, y_values: Vec<f64>) -> Self {
        assert_eq!(x_values.len(), y_values.len(), "x and y must have same length");
        Self { x_values, y_values }
    }

    /// Fit a power law using log-log linear regression.
    /// Fits ln(y) = ln(a) + b * ln(x).
    pub fn fit(&self) -> PowerLawFit {
        let n = self.x_values.len() as f64;
        if n < 2.0 {
            return PowerLawFit {
                exponent: 0.0,
                coefficient: 0.0,
                r_squared: 0.0,
                description: "Insufficient data points".to_string(),
            };
        }

        // Filter out non-positive values
        let pairs: Vec<(f64, f64)> = self.x_values.iter()
            .zip(self.y_values.iter())
            .filter(|(&x, &y)| x > 0.0 && y > 0.0)
            .map(|(&x, &y)| (x.ln(), y.ln()))
            .collect();

        let n = pairs.len() as f64;
        if n < 2.0 {
            return PowerLawFit {
                exponent: 0.0,
                coefficient: 0.0,
                r_squared: 0.0,
                description: "Insufficient positive data points".to_string(),
            };
        }

        let sum_x: f64 = pairs.iter().map(|(x, _)| *x).sum();
        let sum_y: f64 = pairs.iter().map(|(_, y)| *y).sum();
        let sum_xx: f64 = pairs.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = pairs.iter().map(|(x, y)| x * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return PowerLawFit {
                exponent: 0.0,
                coefficient: 0.0,
                r_squared: 0.0,
                description: "Degenerate data (no variance in x)".to_string(),
            };
        }

        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let ln_a = (sum_y - b * sum_x) / n;
        let a = ln_a.exp();

        // R-squared on log-transformed data
        let mean_y = sum_y / n;
        let ss_tot: f64 = pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = pairs.iter()
            .map(|(x, y)| (y - (ln_a + b * x)).powi(2))
            .sum();
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

        PowerLawFit {
            exponent: b,
            coefficient: a,
            r_squared: r_squared.max(0.0).min(1.0),
            description: format!("y = {:.4} * x^{:.4} (R² = {:.4})", a, b, r_squared),
        }
    }

    /// Fit cluster count scaling: clusters ∝ N^α.
    pub fn fit_cluster_scaling(populations: &[usize], cluster_counts: &[usize]) -> PowerLawFit {
        let sl = ScalingLaw::new(
            populations.iter().map(|&n| n as f64).collect(),
            cluster_counts.iter().map(|&c| c as f64).collect(),
        );
        sl.fit()
    }

    /// Fit win rate std scaling: std ∝ N^α.
    pub fn fit_win_rate_std_scaling(populations: &[usize], stds: &[f64]) -> PowerLawFit {
        let sl = ScalingLaw::new(
            populations.iter().map(|&n| n as f64).collect(),
            stds.to_vec(),
        );
        sl.fit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_power_law() {
        // y = 2 * x^3
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi.powi(3)).collect();
        let sl = ScalingLaw::new(x, y);
        let fit = sl.fit();
        assert!((fit.exponent - 3.0).abs() < 0.01);
        assert!((fit.coefficient - 2.0).abs() < 0.1);
        assert!(fit.r_squared > 0.99);
    }

    #[test]
    fn test_insufficient_data() {
        let sl = ScalingLaw::new(vec![1.0], vec![1.0]);
        let fit = sl.fit();
        assert_eq!(fit.exponent, 0.0);
    }

    #[test]
    fn test_negative_values_filtered() {
        let sl = ScalingLaw::new(vec![1.0, -1.0, 2.0], vec![1.0, 1.0, -1.0]);
        let fit = sl.fit();
        // Only one valid pair remains after filtering
        assert_eq!(fit.exponent, 0.0);
    }

    #[test]
    fn test_predict() {
        let fit = PowerLawFit {
            exponent: 2.0,
            coefficient: 3.0,
            r_squared: 1.0,
            description: String::new(),
        };
        assert!((fit.predict(4.0) - 48.0).abs() < 1e-10);
    }

    #[test]
    fn test_quality_labels() {
        assert_eq!(PowerLawFit { exponent: 0.0, coefficient: 0.0, r_squared: 0.99, description: String::new() }.quality(), "excellent");
        assert_eq!(PowerLawFit { exponent: 0.0, coefficient: 0.0, r_squared: 0.85, description: String::new() }.quality(), "good");
        assert_eq!(PowerLawFit { exponent: 0.0, coefficient: 0.0, r_squared: 0.6, description: String::new() }.quality(), "moderate");
        assert_eq!(PowerLawFit { exponent: 0.0, coefficient: 0.0, r_squared: 0.3, description: String::new() }.quality(), "poor");
    }

    #[test]
    fn test_cluster_scaling_convenience() {
        let pops = vec![24, 240, 2400, 24000];
        let clusters = vec![7, 10, 14, 18]; // approximate
        let fit = ScalingLaw::fit_cluster_scaling(&pops, &clusters);
        assert!(fit.exponent > 0.0);
        assert!(fit.r_squared > 0.0);
    }
}
