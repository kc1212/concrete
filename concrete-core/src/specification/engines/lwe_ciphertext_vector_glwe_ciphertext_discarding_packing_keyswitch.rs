use super::engine_error;
use crate::prelude::{GlweCiphertextEntity, PackingKeyswitchKeyEntity};
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::LweCiphertextVectorEntity;

engine_error! {
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError for LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine @
    InputLweDimensionMismatch => "The input ciphertext vector and keyswitch key input LWE \
                                  dimension must be the same.",
    OutputGlweDimensionMismatch => "The output ciphertext vector and keyswitch key output GLWE \
                                    dimensions must be the same.",
    OutputPolynomialSizeMismatch => "The output ciphertext vector and keyswitch key polynomial \
                                     sizes must be the same.",
    CiphertextCountMismatch => "The input ciphertext count is bigger than the output polynomial degree."
}

impl<EngineError: std::error::Error>
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError<EngineError>
{
    /// Validates the inputs
    pub fn perform_generic_checks<KeyswitchKey, InputCiphertextVector, OutputCiphertext>(
        output: &mut OutputCiphertext,
        input: &InputCiphertextVector,
        ksk: &KeyswitchKey,
    ) -> Result<(), Self>
    where
        KeyswitchKey: PackingKeyswitchKeyEntity,
        InputCiphertextVector:
            LweCiphertextVectorEntity<KeyDistribution = KeyswitchKey::InputKeyDistribution>,
        OutputCiphertext:
            GlweCiphertextEntity<KeyDistribution = KeyswitchKey::OutputKeyDistribution>,
    {
        if input.lwe_dimension() != ksk.input_lwe_dimension() {
            return Err(Self::InputLweDimensionMismatch);
        }

        if output.glwe_dimension() != ksk.output_glwe_dimension() {
            return Err(Self::OutputGlweDimensionMismatch);
        }

        if output.polynomial_size() != ksk.output_polynomial_size() {
            return Err(Self::OutputPolynomialSizeMismatch);
        }

        if input.lwe_ciphertext_count().0 > output.polynomial_size().0 {
            return Err(Self::CiphertextCountMismatch);
        }
        Ok(())
    }
}

/// A trait for engines keyswitching (discarding) LWE ciphertext vectors into a GLWE ciphertext.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE ciphertext vector
/// with the element-wise keyswitch of the `input` LWE ciphertext vector, under the `ksk` lwe
/// keyswitch key.
///
/// # Formal Definition
pub trait LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine<
    KeyswitchKey,
    InputCiphertextVector,
    OutputCiphertext,
>: AbstractEngine where
    KeyswitchKey: PackingKeyswitchKeyEntity,
    InputCiphertextVector:
        LweCiphertextVectorEntity<KeyDistribution = KeyswitchKey::InputKeyDistribution>,
    OutputCiphertext: GlweCiphertextEntity<KeyDistribution = KeyswitchKey::OutputKeyDistribution>,
{
    /// Keyswitch an LWE ciphertext vector.
    fn discard_packing_keyswitch_lwe_ciphertext_vector(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertextVector,
        ksk: &KeyswitchKey,
    ) -> Result<
        (),
        LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError<Self::EngineError>,
    >;

    /// Unsafely keyswitch an LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError`]. For safety concerns
    /// _specific_ to an engine, refer to the implementer safety section.
    unsafe fn discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertextVector,
        ksk: &KeyswitchKey,
    );
}
