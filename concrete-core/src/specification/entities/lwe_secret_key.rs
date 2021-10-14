use crate::specification::entities::markers::{KeyFlavorMarker, LweSecretKeyKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::LweDimension;

/// A trait implemented by types embodying an lwe secret key.
///
/// An lwe secret key is associated with a
/// [`KeyFlavor`](`LweSecretKeyEntity::KeyFlavor`) type, which conveys its flavor.
pub trait LweSecretKeyEntity: AbstractEntity<Kind = LweSecretKeyKind> {
    /// The flavor of this key
    type KeyFlavor: KeyFlavorMarker;

    /// Returns the lwe dimension of the key.
    fn lwe_dimension(&self) -> LweDimension;
}
