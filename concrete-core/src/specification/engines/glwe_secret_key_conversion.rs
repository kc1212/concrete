use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweSecretKeyEntity;

engine_error! {
    GlweSecretKeyConversionError for GlweSecretKeyConversionEngine @
}

/// A trait for engines converting glwe secret keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a glwe secret key containing the
/// conversion of the `input` glwe secret key to a different representation.
///
/// # Formal Definition
pub trait GlweSecretKeyConversionEngine<Input, Output>: AbstractEngine
where
    Input: GlweSecretKeyEntity,
    Output: GlweSecretKeyEntity<KeyFlavor = Input::KeyFlavor>,
{
    /// Converts a glwe secret key.
    fn convert_glwe_secret_key(
        &mut self,
        input: &Input,
    ) -> Result<Output, GlweSecretKeyConversionError<Self::EngineError>>;

    /// Unsafely converts a glwe secret key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweSecretKeyConversionError`]. For safety concerns _specific_ to an engine, refer to
    /// the implementer safety section.
    unsafe fn convert_glwe_secret_key_unchecked(&mut self, input: &Input) -> Output;
}
