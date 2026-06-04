mod scaling;
mod metrics;
mod law;
mod plateau;
mod divergence;
mod report;

pub use scaling::{ScalingExperiment, ScaleConfig};
pub use metrics::{MetricsAtScale, DistributionMetrics};
pub use law::{ScalingLaw, PowerLawFit};
pub use plateau::{PlateauDetector, PlateauResult};
pub use divergence::CrossScaleComparison;
pub use report::ScaleReport;
