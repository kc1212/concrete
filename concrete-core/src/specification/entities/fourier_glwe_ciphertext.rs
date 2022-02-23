use crate::specification::entities::markers::{FourierGlweCiphertextKind, KeyDistributionMarker};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

/// A trait implemented by types embodying a GLWE ciphertext in the Fourier Domain.
///
/// A Fourier GLWE ciphertext is associated with a
/// [`KeyDistribution`](`FourierGlweCiphertextEntity::KeyDistribution`) type, which conveys the
/// distribution of the secret key that it was encrypted with.
///
/// # Formal Definition
///
/// A GLWE ciphertext in the Fourier domain.
pub trait FourierGlweCiphertextEntity: AbstractEntity<Kind = FourierGlweCiphertextKind> {
    /// The distribution of the key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the GLWE dimension of the ciphertext.
    fn glwe_dimension(&self) -> GlweDimension;

    /// Returns the polynomial size of the ciphertext.
    fn polynomial_size(&self) -> PolynomialSize;
}
