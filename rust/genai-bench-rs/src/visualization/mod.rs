//! Visualization and plotting

pub mod histogram;
pub mod throughput;
pub mod percentiles;

pub use histogram::HistogramPlotter;
pub use throughput::ThroughputPlotter;
pub use percentiles::PercentilePlotter;
