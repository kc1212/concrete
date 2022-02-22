use crate::prelude::{GlweCiphertextTensorProductEngine, AbstractEngine, GlweCiphertextTensorProductError};
use crate::backends::core::entities::GlweCiphertext32;
use crate::backends::core::engines::CoreEngine;
use crate::backends::core::private::crypto::glwe::{GlweCiphertext, FourierGlweCiphertext};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::entities::GlweCiphertextEntity;
use concrete_commons::parameters::GlweSize;

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
                                                       // we need to use this on FourierGLWECiphertexts32
                                                       input1: &FourierGlweCiphertext32,
                                                       input2: &GlweCiphertext32) -> GlweCiphertext32 {

        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u32,
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0 * (3 + input1.glwe_dimension().0) * (1/2)),
        );

        // .0 accesses GLWE ciphertext inside input1
        input1.0.tensor_product(input2)
    }
}


