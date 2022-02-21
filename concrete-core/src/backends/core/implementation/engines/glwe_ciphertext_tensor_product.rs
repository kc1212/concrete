use crate::prelude::{GlweCiphertextTensorProductEngine, AbstractEngine, GlweCiphertextTensorProductError};
use crate::backends::core::entities::GlweCiphertext32;
use crate::backends::core::engines::CoreEngine;

/// # Description:
/// Implementation of [`GlweTensorProductEngine`] for [`CoreEngine`] that operates on 32-bit
/// integers.
impl GlweCiphertextTensorProductEngine<GlweCiphertext32, GlweCiphertext32, GlweCiphertext32>
    for CoreEngine{

    fn tensor_product_glwe_ciphertext(&mut self,
                                      input1: &GlweCiphertext32,
                                      input2: &GlweCiphertext32) -> Result<GlweCiphertext32,
        GlweCiphertextTensorProductError<Self::EngineError>> {
        GlweCiphertextTensorProductError::perform_generic_checks(input1, input2)?;
        Ok(unsafe { self.tensor_product_glwe_ciphertext_unchecked(input1, input2)})
    }


    unsafe fn tensor_product_glwe_ciphertext_unchecked(&mut self,
                                                       input1: &GlweCiphertext32,
                                                       input2: &GlweCiphertext32) -> GlweCiphertext32 {
        todo!()
    }
}


