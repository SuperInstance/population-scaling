/// Result of plateau detection on a metric series.
#[derive(Debug, Clone)]
pub struct PlateauResult {
    pub metric_name: String,
    pub values: Vec<(usize, f64)>, // (population_size, value)
    pub plateau_start: Option<usize>, // index where plateau begins
    pub plateau_value: f64,
    pub is_plateaued: bool,
    pub growth_rates: Vec<f64>, // percent change between consecutive points
}

impl PlateauResult {
    /// Describe the plateau finding.
    pub fn summary(&self) -> String {
        if self.is_plateaued {
            if let Some(idx) = self.plateau_start {
                let (pop, _) = self.values[idx];
                format!(
                    "{} plateaus at ~{:.1}% starting from population size {}",
                    self.metric_name, self.plateau_value * 100.0, pop
                )
            } else {
                format!("{} shows no clear plateau", self.metric_name)
            }
        } else {
            format!("{} continues to grow across scales", self.metric_name)
        }
    }
}

/// Detect when a metric stops growing meaningfully with scale.
#[derive(Debug, Clone)]
pub struct PlateauDetector {
    /// Relative threshold: growth rate below this signals plateau.
    pub growth_threshold: f64,
    /// Minimum consecutive flat points to declare plateau.
    pub min_flat_points: usize,
}

impl PlateauDetector {
    pub fn new(growth_threshold: f64, min_flat_points: usize) -> Self {
        Self { growth_threshold, min_flat_points }
    }

    pub fn default() -> Self {
        Self { growth_threshold: 0.05, min_flat_points: 2 }
    }

    /// Analyze a metric series for plateau behavior.
    pub fn detect(&self, metric_name: &str, values: &[(usize, f64)]) -> PlateauResult {
        if values.len() < 2 {
            return PlateauResult {
                metric_name: metric_name.to_string(),
                values: values.to_vec(),
                plateau_start: None,
                plateau_value: 0.0,
                is_plateaued: false,
                growth_rates: vec![],
            };
        }

        // Compute growth rates between consecutive points
        let growth_rates: Vec<f64> = values.windows(2)
            .map(|w| {
                let prev = w[0].1;
                let curr = w[1].1;
                if prev.abs() < 1e-15 { 0.0 } else { (curr - prev) / prev.abs() }
            })
            .collect();

        // Find where growth drops below threshold for min_flat_points consecutive points
        let mut plateau_start: Option<usize> = None;
        let mut consecutive_flat = 0;

        for (i, &rate) in growth_rates.iter().enumerate() {
            if rate.abs() < self.growth_threshold {
                consecutive_flat += 1;
                if consecutive_flat >= self.min_flat_points {
                    plateau_start = Some(i + 1 - consecutive_flat);
                    break;
                }
            } else {
                consecutive_flat = 0;
            }
        }

        let is_plateaued = plateau_start.is_some();
        let plateau_value = if is_plateaued {
            values[plateau_start.unwrap()].1
        } else {
            values.last().map(|(_, v)| *v).unwrap_or(0.0)
        };

        PlateauResult {
            metric_name: metric_name.to_string(),
            values: values.to_vec(),
            plateau_start: plateau_start.map(|i| i + 1), // point to the value, not growth rate
            plateau_value,
            is_plateaued,
            growth_rates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_clear_plateau() {
        let values = vec![
            (10, 0.5),
            (100, 0.7),
            (1000, 0.82),
            (10000, 0.822),
            (100000, 0.823),
        ];
        let det = PlateauDetector::new(0.05, 2);
        let result = det.detect("entropy", &values);
        assert!(result.is_plateaued);
        assert!((result.plateau_value - 0.82).abs() < 0.01);
    }

    #[test]
    fn test_no_plateau() {
        let values = vec![
            (10, 1.0),
            (100, 2.0),
            (1000, 4.0),
            (10000, 8.0),
        ];
        let det = PlateauDetector::new(0.05, 2);
        let result = det.detect("clusters", &values);
        assert!(!result.is_plateaued);
    }

    #[test]
    fn test_too_few_points() {
        let values = vec![(10, 0.5)];
        let det = PlateauDetector::default();
        let result = det.detect("entropy", &values);
        assert!(!result.is_plateaued);
    }

    #[test]
    fn test_growth_rates_computed() {
        let values = vec![(10, 10.0), (100, 20.0), (1000, 25.0)];
        let det = PlateauDetector::default();
        let result = det.detect("test", &values);
        assert_eq!(result.growth_rates.len(), 2);
        assert!((result.growth_rates[0] - 1.0).abs() < 1e-10); // 100% growth
        assert!((result.growth_rates[1] - 0.25).abs() < 1e-10); // 25% growth
    }

    #[test]
    fn test_summary_format() {
        let values = vec![
            (10, 0.5),
            (100, 0.82),
            (1000, 0.822),
            (10000, 0.823),
        ];
        let det = PlateauDetector::new(0.05, 2);
        let result = det.detect("entropy", &values);
        let s = result.summary();
        assert!(s.contains("entropy"));
    }
}
