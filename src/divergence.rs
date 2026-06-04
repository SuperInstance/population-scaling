/// Jensen-Shannon divergence for comparing distributions across scales.

/// Compute the Jensen-Shannon divergence between two probability distributions.
/// JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5*(P+Q).
///
/// Returns a value in [0, ln(2)] where 0 = identical, ln(2) ≈ 1.386 = maximally different.
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    let m: Vec<f64> = p.iter().zip(q.iter())
        .map(|(&pi, &qi)| 0.5 * (pi + qi))
        .collect();

    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Kullback-Leibler divergence KL(p || q).
fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| {
            let qi_safe = if qi > 0.0 { qi } else { 1e-15 };
            pi * (pi / qi_safe).ln()
        })
        .sum()
}

/// Cross-scale comparison using JSD between consecutive population sizes.
#[derive(Debug, Clone)]
pub struct CrossScaleComparison {
    pub divergences: Vec<(usize, usize, f64)>, // (scale_i, scale_j, JSD)
    pub max_divergence: f64,
    pub mean_divergence: f64,
}

impl CrossScaleComparison {
    /// Compare consecutive pairs of distributions across scales.
    pub fn from_distributions(
        populations: &[usize],
        distributions: &[Vec<f64>],
    ) -> Self {
        assert_eq!(populations.len(), distributions.len());

        let divergences: Vec<(usize, usize, f64)> = populations.windows(2)
            .zip(distributions.windows(2))
            .map(|(pops, dists)| {
                (pops[0], pops[1], js_divergence(&dists[0], &dists[1]))
            })
            .collect();

        let max_divergence = divergences.iter()
            .map(|&(_, _, d)| d)
            .fold(0.0_f64, f64::max);

        let mean_divergence = if divergences.is_empty() {
            0.0
        } else {
            divergences.iter().map(|&(_, _, d)| d).sum::<f64>() / divergences.len() as f64
        };

        Self { divergences, max_divergence, mean_divergence }
    }

    /// Are the distributions converging? (later divergences smaller than earlier ones)
    pub fn is_converging(&self) -> bool {
        if self.divergences.len() < 2 {
            return false;
        }
        let first_half: f64 = self.divergences[..self.divergences.len()/2]
            .iter().map(|&(_, _, d)| d).sum::<f64>()
            / (self.divergences.len()/2) as f64;
        let second_half: f64 = self.divergences[self.divergences.len()/2..]
            .iter().map(|&(_, _, d)| d).sum::<f64>()
            / (self.divergences.len() - self.divergences.len()/2) as f64;
        second_half < first_half
    }

    /// Generate a summary string.
    pub fn summary(&self) -> String {
        let lines: Vec<String> = self.divergences.iter()
            .map(|(i, j, d)| format!("  JSD({} → {}): {:.6}", i, j, d))
            .collect();
        let conv = if self.is_converging() { "converging" } else { "diverging" };
        format!("Cross-scale JSD ({}):\n{}\n  Mean: {:.6}, Max: {:.6}",
            conv, lines.join("\n"), self.mean_divergence, self.max_divergence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_js_identical() {
        let p = vec![0.3, 0.4, 0.3];
        let jsd = js_divergence(&p, &p);
        assert!(jsd.abs() < 1e-10, "JSD of identical distributions should be 0, got {}", jsd);
    }

    #[test]
    fn test_js_maximally_different() {
        let p = vec![1.0, 0.0, 0.0];
        let q = vec![0.0, 1.0, 0.0];
        let jsd = js_divergence(&p, &q);
        let ln2 = 2.0_f64.ln();
        assert!((jsd - ln2).abs() < 0.01, "JSD of maximally different should be ln(2)={:.4}, got {:.4}", ln2, jsd);
    }

    #[test]
    fn test_js_symmetric() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.2, 0.5, 0.3];
        let jsd_pq = js_divergence(&p, &q);
        let jsd_qp = js_divergence(&q, &p);
        assert!((jsd_pq - jsd_qp).abs() < 1e-10);
    }

    #[test]
    fn test_js_nonnegative() {
        let p = vec![0.6, 0.3, 0.1];
        let q = vec![0.1, 0.3, 0.6];
        assert!(js_divergence(&p, &q) >= 0.0);
    }

    #[test]
    fn test_cross_scale_comparison() {
        let pops = vec![24, 240, 2400];
        let dists = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.4, 0.35, 0.25],
            vec![0.35, 0.35, 0.3],
        ];
        let csc = CrossScaleComparison::from_distributions(&pops, &dists);
        assert_eq!(csc.divergences.len(), 2);
        assert!(csc.max_divergence > 0.0);
        assert!(csc.mean_divergence > 0.0);
    }

    #[test]
    fn test_convergence_detection() {
        // Distributions getting more similar over time
        let pops = vec![24, 240, 2400];
        let dists = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.4, 0.35, 0.25],
            vec![0.35, 0.34, 0.31],
        ];
        let csc = CrossScaleComparison::from_distributions(&pops, &dists);
        // First JSD should be larger than second
        assert!(csc.divergences[0].2 > csc.divergences[1].2);
    }

    #[test]
    fn test_summary_format() {
        let pops = vec![24, 240];
        let dists = vec![vec![0.5, 0.3, 0.2], vec![0.4, 0.35, 0.25]];
        let csc = CrossScaleComparison::from_distributions(&pops, &dists);
        let s = csc.summary();
        assert!(s.contains("JSD(24"));
    }
}
