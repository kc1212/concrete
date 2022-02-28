use crate::backends::core::private::crypto::glwe::PackingKeyswitchKey as ImplPackingKeyswitchKey;
use crate::prelude::markers::PackingKeyswitchKeyKind;
use crate::prelude::PackingKeyswitchKeyEntity;
use crate::specification::entities::markers::BinaryKeyDistribution;
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension,
};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing a packing keyswitch key with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct PackingKeyswitchKey32(pub(crate) ImplPackingKeyswitchKey<Vec<u32>>);
impl AbstractEntity for PackingKeyswitchKey32 {
    type Kind = PackingKeyswitchKeyKind;
}
impl PackingKeyswitchKeyEntity for PackingKeyswitchKey32 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;

    fn input_lwe_dimension(&self) -> LweDimension {
        self.0.before_key_size()
    }

    fn output_glwe_dimension(&self) -> GlweDimension {
        self.0.after_key_size()
    }

    fn output_polynomial_size(&self) -> concrete_commons::parameters::PolynomialSize {
        self.0.polynomial_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_levels_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}

/// A structure representing a packing keyswitch key with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct PackingKeyswitchKey64(pub(crate) ImplPackingKeyswitchKey<Vec<u64>>);
impl AbstractEntity for PackingKeyswitchKey64 {
    type Kind = PackingKeyswitchKeyKind;
}
impl PackingKeyswitchKeyEntity for PackingKeyswitchKey64 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;

    fn input_lwe_dimension(&self) -> LweDimension {
        self.0.before_key_size()
    }

    fn output_glwe_dimension(&self) -> GlweDimension {
        self.0.after_key_size()
    }

    fn output_polynomial_size(&self) -> concrete_commons::parameters::PolynomialSize {
        self.0.polynomial_size()
    }

    fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_levels_count()
    }

    fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }
}
