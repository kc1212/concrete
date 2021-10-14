use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    CleartextVectorEntity, EncoderVectorEntity, PlaintextVectorEntity,
};

engine_error! {
    CleartextVectorEncodingError for CleartextVectorEncodingEngine @
}

/// A trait for engines encoding cleartext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a plaintext vector containing the
/// element-wise encodings of the `cleartext_vector` cleartext vector under the `encoder_vector`
/// encoder vector.
///
/// # Formal Definition
pub trait CleartextVectorEncodingEngine<EncoderVector, CleartextVector, PlaintextVector>:
    AbstractEngine
where
    EncoderVector: EncoderVectorEntity,
    CleartextVector: CleartextVectorEntity,
    PlaintextVector: PlaintextVectorEntity,
{
    /// Encodes a cleartext vector into a plaintext vector.
    fn encode_cleartext_vector(
        &mut self,
        encoder_vector: &EncoderVector,
        cleartext_vector: &CleartextVector,
    ) -> Result<PlaintextVector, CleartextVectorEncodingError<Self::EngineError>>;

    /// Unsafely encodes a cleartext vector into a plaintext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`CleartextVectorEncodingError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn encode_cleartext_unchecked(
        &mut self,
        encoder: &EncoderVector,
        cleartext: &CleartextVector,
    ) -> PlaintextVector;
}
