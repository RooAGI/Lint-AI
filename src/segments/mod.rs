mod catalog;
mod diagnostics;
pub mod intent;
mod model;
mod query;
pub mod relations;
mod routing;
mod segmented;

#[cfg(test)]
mod tests;

// Re-exported for the `#[cfg(test)]` suite below, which does `use super::*;`.
// (Nothing outside tests names these paths in non-test builds.)
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use catalog::*;
pub use diagnostics::*;
pub use intent::*;
pub use model::*;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use query::*;
pub use relations::*;
pub use routing::*;
pub use segmented::*;
