#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::SignedInteger;
use concrete_commons::parameters::{
    CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    LweDimension, MonomialDegree, PlaintextCount, PolynomialSize,
};

use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::lwe::{LweCiphertext, LweList};
use crate::backends::core::private::crypto::secret::generators::EncryptionRandomGenerator;
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::decomposition::{
    DecompositionLevel, DecompositionTerm, SignedDecomposer,
};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::{GlweCiphertext, GlweList};

/// A packing keyswithing key.
///
/// A packing keyswitching key allows to  pack several LweCiphertext into a single GlweCiphertext.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackingKeyswitchKey<Cont> {
    tensor: Tensor<Cont>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    glwe_size: GlweSize,
    polynomial_size: PolynomialSize,
}

tensor_traits!(PackingKeyswitchKey);

impl<Scalar> PackingKeyswitchKey<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates a packing keyswitching key whose masks and bodies are all `value`.
    ///
    /// # Note
    ///
    /// This function does *not* generate a keyswitch key, but merely allocates a container of the
    /// right size. See [`PackingKeyswitchKey::fill_with_keyswitch_key`] to fill the container with
    /// a proper keyswitching key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(
    ///     ksk.decomposition_levels_count(),
    ///     DecompositionLevelCount(10)
    /// );
    /// assert_eq!(ksk.decomposition_base_log(), DecompositionBaseLog(16));
    /// assert_eq!(ksk.glwe_size(), GlweSize(3));
    /// assert_eq!(ksk.before_key_size(), LweDimension(10));
    /// assert_eq!(ksk.after_key_size(), GlweDimension(2));
    /// ```
    pub fn allocate(
        value: Scalar,
        decomp_size: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        input_size: LweDimension,
        output_size: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self {
        PackingKeyswitchKey {
            tensor: Tensor::from_container(vec![
                value;
                decomp_size.0
                    * output_size.to_glwe_size().0
                    * polynomial_size.0
                    * input_size.0
            ]),
            decomp_base_log,
            decomp_level_count: decomp_size,
            glwe_size: output_size.to_glwe_size(),
            polynomial_size,
        }
    }
}

impl<Cont> PackingKeyswitchKey<Cont> {
    /// Creates a packing key switching key from a container.
    ///
    /// # Notes
    ///
    /// This method does not create a keyswitching key, but merely wrap the container in the proper
    /// type. It assumes that either the container already contains a proper keyswitching key, or
    /// that [`PackingKeyswitchKey::fill_with_keyswitch_key`] will be called right after.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let input_size = LweDimension(200);
    /// let output_size = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// let decomp_log_base = DecompositionBaseLog(7);
    /// let decomp_level_count = DecompositionLevelCount(4);
    ///
    /// let ksk = PackingKeyswitchKey::from_container(
    ///     vec![
    ///         0 as u8;
    ///         input_size.0 * (output_size.0 + 1) * polynomial_size.0 * decomp_level_count.0
    ///     ],
    ///     decomp_log_base,
    ///     decomp_level_count,
    ///     output_size,
    ///     polynomial_size,
    /// );
    ///
    /// assert_eq!(ksk.decomposition_levels_count(), DecompositionLevelCount(4));
    /// assert_eq!(ksk.decomposition_base_log(), DecompositionBaseLog(7));
    /// assert_eq!(ksk.glwe_size(), GlweSize(3));
    /// assert_eq!(ksk.before_key_size(), LweDimension(200));
    /// assert_eq!(ksk.after_key_size(), GlweDimension(2));
    /// ```
    pub fn from_container(
        cont: Cont,
        decomp_base_log: DecompositionBaseLog,
        decomp_size: DecompositionLevelCount,
        output_size: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> PackingKeyswitchKey<Cont>
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => output_size.to_glwe_size().0 * polynomial_size.0, decomp_size.0);
        PackingKeyswitchKey {
            tensor,
            decomp_base_log,
            decomp_level_count: decomp_size,
            glwe_size: output_size.to_glwe_size(),
            polynomial_size,
        }
    }

    /// Return the size of the output key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(ksk.after_key_size(), GlweDimension(2));
    /// ```
    pub fn after_key_size(&self) -> GlweDimension {
        self.glwe_size.to_glwe_dimension()
    }

    /// Returns the size of the ciphertexts encoding each level of the decomposition of each bits
    /// of the input key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(ksk.glwe_size(), GlweSize(3));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the degree of the polynomials composing the ciphertexts
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(ksk.polynomial_size(), PolynomialSize(256));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }

    /// Returns the size of the input key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(ksk.before_key_size(), LweDimension(10));
    /// ```
    pub fn before_key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        LweDimension(
            self.as_tensor().len()
                / (self.glwe_size.0 * self.polynomial_size.0 * self.decomp_level_count.0),
        )
    }

    /// Returns the number of levels used for the decomposition of the input key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(
    ///     ksk.decomposition_levels_count(),
    ///     DecompositionLevelCount(10)
    /// );
    /// ```
    pub fn decomposition_levels_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        self.decomp_level_count
    }

    /// Returns the logarithm of the base used for the decomposition of the input key bits.
    ///
    /// Indeed, the basis used is always of the form $2^N$. This function returns $N$.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::*;
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(10),
    ///     GlweDimension(2),
    ///     PolynomialSize(256),
    /// );
    /// assert_eq!(ksk.decomposition_base_log(), DecompositionBaseLog(16));
    /// ```
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog
    where
        Self: AsRefTensor,
    {
        self.decomp_base_log
    }

    /// Fills the current keyswitch key container with an actual keyswitching key constructed from
    /// an input and an output key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::glwe::PackingKeyswitchKey;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    ///
    /// let input_size = LweDimension(10);
    /// let output_size = GlweDimension(3);
    /// let polynomial_size = PolynomialSize(256);
    /// let decomp_log_base = DecompositionBaseLog(3);
    /// let decomp_level_count = DecompositionLevelCount(5);
    /// let cipher_size = LweSize(55);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    ///
    /// let input_key = LweSecretKey::generate_binary(input_size, &mut secret_generator);
    /// let output_key =
    ///     GlweSecretKey::generate_binary(output_size, polynomial_size, &mut secret_generator);
    ///
    /// let mut ksk = PackingKeyswitchKey::allocate(
    ///     0 as u32,
    ///     decomp_level_count,
    ///     decomp_log_base,
    ///     input_size,
    ///     output_size,
    ///     polynomial_size,
    /// );
    /// ksk.fill_with_keyswitch_key(&input_key, &output_key, noise, &mut encryption_generator);
    ///
    /// assert!(!ksk.as_tensor().iter().all(|a| *a == 0));
    /// ```
    pub fn fill_with_keyswitch_key<InKeyCont, OutKeyCont, Scalar>(
        &mut self,
        before_key: &LweSecretKey<BinaryKeyKind, InKeyCont>,
        after_key: &GlweSecretKey<BinaryKeyKind, OutKeyCont>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, InKeyCont>: AsRefTensor<Element = Scalar>,
        GlweSecretKey<BinaryKeyKind, OutKeyCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We instantiate a buffer
        let mut messages = PlaintextList::from_container(vec![
            <Self as AsMutTensor>::Element::ZERO;
            self.decomp_level_count.0
                * self.polynomial_size.0
        ]);

        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_level_count;
        let decomp_base_log = self.decomp_base_log;
        let polynomial_size = self.polynomial_size;

        // loop over the before key blocks
        for (input_key_bit, keyswitch_key_block) in before_key
            .as_tensor()
            .iter()
            .zip(self.bit_decomp_iter_mut())
        {
            // We reset the buffer
            messages
                .as_mut_tensor()
                .fill_with_element(<Self as AsMutTensor>::Element::ZERO);

            // We fill the buffer with the powers of the key bits
            for (level, mut message) in (1..=decomp_level_count.0)
                .map(DecompositionLevel)
                .zip(messages.sublist_iter_mut(PlaintextCount(polynomial_size.0)))
            {
                *message.as_mut_tensor().first_mut() =
                    DecompositionTerm::new(level, decomp_base_log, *input_key_bit)
                        .to_recomposition_summand();
            }

            // We encrypt the buffer
            after_key.encrypt_glwe_list(
                &mut keyswitch_key_block.into_glwe_list(),
                &messages,
                noise_parameters,
                generator,
            );
        }
    }

    /// Iterates over borrowed `LweKeyBitDecomposition` elements.
    ///
    /// One `LweKeyBitDecomposition` being a set of lwe ciphertext, encrypting under the output
    /// key, the $l$ levels of the signed decomposition of a single bit of the input key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::PackingKeyswitchKey};
    /// use concrete_core::backends::core::private::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(15),
    ///     LweDimension(20)
    /// );
    /// for decomp in ksk.bit_decomp_iter() {
    ///     assert_eq!(decomp.lwe_size(), ksk.lwe_size());
    ///     assert_eq!(decomp.count().0, 10);
    /// }
    /// assert_eq!(ksk.bit_decomp_iter().count(), 15);
    /// ```
    pub(crate) fn bit_decomp_iter(
        &self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.glwe_size.0 * self.polynomial_size.0, self.decomp_level_count.0);
        let size = self.decomp_level_count.0 * self.glwe_size.0 * self.polynomial_size.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.polynomial_size;
        self.as_tensor().subtensor_iter(size).map(move |sub| {
            LweKeyBitDecomposition::from_container(sub.into_container(), glwe_size, poly_size)
        })
    }

    /// Iterates over mutably borrowed `LweKeyBitDecomposition` elements.
    ///
    /// One `LweKeyBitDecomposition` being a set of lwe ciphertext, encrypting under the output
    /// key, the $l$ levels of the signed decomposition of a single bit of the input key.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::PackingKeyswitchKey};
    /// use concrete_core::backends::core::private::math::tensor::{AsRefTensor, AsMutTensor};
    /// use concrete_core::backends::core::private::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
    /// let mut ksk = PackingKeyswitchKey::allocate(
    ///     0 as u8,
    ///     DecompositionLevelCount(10),
    ///     DecompositionBaseLog(16),
    ///     LweDimension(15),
    ///     LweDimension(20)
    /// );
    /// for mut decomp in ksk.bit_decomp_iter_mut() {
    ///     for mut ciphertext in decomp.ciphertext_iter_mut() {
    ///         ciphertext.as_mut_tensor().fill_with_element(0);
    ///     }
    /// }
    /// assert!(ksk.as_tensor().iter().all(|a| *a == 0));
    /// assert_eq!(ksk.bit_decomp_iter_mut().count(), 15);
    /// ```
    pub(crate) fn bit_decomp_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = LweKeyBitDecomposition<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.glwe_size.0 * self.polynomial_size.0, self.decomp_level_count.0);
        let chunks_size = self.decomp_level_count.0 * self.glwe_size.0 * self.polynomial_size.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.polynomial_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |sub| {
                LweKeyBitDecomposition::from_container(sub.into_container(), glwe_size, poly_size)
            })
    }

    /// Switches the key of a signel Lwe ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::*;
    /// use concrete_core::backends::core::private::crypto::glwe::*;
    /// use concrete_core::backends::core::private::crypto::lwe::*;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    ///
    /// let input_size = LweDimension(1024);
    /// let output_size = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// let decomp_log_base = DecompositionBaseLog(3);
    /// let decomp_level_count = DecompositionLevelCount(8);
    /// let noise = LogStandardDev::from_log_standard_dev(-15.);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    /// let input_key = LweSecretKey::generate_binary(input_size, &mut secret_generator);
    /// let output_key =
    ///     GlweSecretKey::generate_binary(output_size, polynomial_size, &mut secret_generator);
    ///
    /// let mut ksk = PackingKeyswitchKey::allocate(
    ///     0 as u64,
    ///     decomp_level_count,
    ///     decomp_log_base,
    ///     input_size,
    ///     output_size,
    ///     polynomial_size,
    /// );
    /// ksk.fill_with_keyswitch_key(&input_key, &output_key, noise, &mut encryption_generator);
    ///
    /// let plaintext: Plaintext<u64> = Plaintext(1432154329994324);
    /// let mut ciphertext = LweCiphertext::allocate(0. as u64, LweSize(1025));
    /// let mut switched_ciphertext =
    ///     GlweCiphertext::allocate(0. as u64, PolynomialSize(256), GlweSize(3));
    /// input_key.encrypt_lwe(
    ///     &mut ciphertext,
    ///     &plaintext,
    ///     noise,
    ///     &mut encryption_generator,
    /// );
    ///
    /// ksk.keyswitch_ciphertext(&mut switched_ciphertext, &ciphertext);
    ///
    /// let mut decrypted = PlaintextList::from_container(vec![0 as u64; 256]);
    /// output_key.decrypt_glwe(&mut decrypted, &switched_ciphertext);
    /// ```
    pub fn keyswitch_ciphertext<InCont, OutCont, Scalar>(
        &self,
        after: &mut GlweCiphertext<OutCont>,
        before: &LweCiphertext<InCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        GlweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        LweCiphertext<InCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.before_key_size().0 => before.lwe_size().to_lwe_dimension().0);
        ck_dim_eq!(self.after_key_size().0 => after.size().to_glwe_dimension().0);

        // We reset the output
        after.as_mut_tensor().fill_with(|| Scalar::ZERO);

        // We copy the body
        *after.get_mut_body().tensor.as_mut_tensor().first_mut() = before.get_body().0;

        // We allocate a buffer to hold the decomposition.
        let mut decomp = Tensor::allocate(Scalar::ZERO, self.decomp_level_count.0);

        // We instantiate a decomposer
        let decomposer = SignedDecomposer::new(self.decomp_base_log, self.decomp_level_count);

        for (block, before_mask) in self
            .bit_decomp_iter()
            .zip(before.get_mask().mask_element_iter())
        {
            let mask_rounded = decomposer.closest_representable(*before_mask);

            torus_small_sign_decompose(decomp.as_mut_slice(), mask_rounded, self.decomp_base_log.0);

            // loop over the number of levels
            for (level_key_cipher, decomposed) in block
                .as_tensor()
                .subtensor_iter(self.glwe_size.0 * self.polynomial_size.0)
                .zip(decomp.iter())
            {
                after
                    .as_mut_tensor()
                    .update_with_wrapping_sub_element_mul(&level_key_cipher, *decomposed);
            }
        }
    }

    /// Packs several LweCiphertext into a single GlweCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     LweDimension, LweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::*;
    /// use concrete_core::backends::core::private::crypto::glwe::*;
    /// use concrete_core::backends::core::private::crypto::lwe::*;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::backends::core::private::crypto::*;
    /// use concrete_core::backends::core::private::math::random::RandomGenerator;
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    /// let mut random_generator = RandomGenerator::new(None);
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    ///
    /// // fix a set of parameters
    /// let n_bit_msg = 8; // bit precision of the plaintext
    /// let nb_ct = CiphertextCount(256); // number of messages to encrypt
    /// let base_log = DecompositionBaseLog(3); // a parameter of the gadget matrix
    /// let level_count = DecompositionLevelCount(8); // a parameter of the gadget matrix
    /// let polynomial_size = PolynomialSize(256);
    /// let messages = PlaintextList::from_tensor(
    ///     random_generator.random_uniform_n_msb_tensor(nb_ct.0, n_bit_msg),
    /// );
    /// // the set of messages to encrypt
    /// let std_input = LogStandardDev::from_log_standard_dev(-10.); // standard deviation of the
    ///                                                              // encrypted messages to KS
    /// let std_ksk = LogStandardDev::from_log_standard_dev(-25.); // standard deviation of the ksk
    ///
    /// // set parameters related to the after (stands for 'after the KS')
    /// let dimension_after = GlweDimension(1);
    /// let sk_after =
    ///     GlweSecretKey::generate_binary(dimension_after, polynomial_size, &mut secret_generator);
    ///
    /// // set parameters related to the before (stands for 'before the KS')
    /// let dimension_before = LweDimension(630);
    /// let sk_before = LweSecretKey::generate_binary(dimension_before, &mut secret_generator);
    ///
    /// // create the before ciphertexts and the after ciphertexts
    /// let mut ciphertexts_before = LweList::allocate(0_u32, dimension_before.to_lwe_size(), nb_ct);
    /// let mut ciphertext_after =
    ///     GlweCiphertext::allocate(0_u32, polynomial_size, dimension_after.to_glwe_size());
    ///
    /// // key switching key generation
    /// let mut ksk = PackingKeyswitchKey::allocate(
    ///     0_u32,
    ///     level_count,
    ///     base_log,
    ///     dimension_before,
    ///     dimension_after,
    ///     polynomial_size,
    /// );
    /// ksk.fill_with_keyswitch_key(&sk_before, &sk_after, std_ksk, &mut encryption_generator);
    ///
    /// // encrypts with the before key our messages
    /// sk_before.encrypt_lwe_list(
    ///     &mut ciphertexts_before,
    ///     &messages,
    ///     std_input,
    ///     &mut encryption_generator,
    /// );
    ///
    /// // key switch before -> after
    /// ksk.packing_keyswitch(&mut ciphertext_after, &ciphertexts_before);
    /// ```
    pub fn packing_keyswitch<InCont, OutCont, Scalar>(
        &self,
        output: &mut GlweCiphertext<OutCont>,
        input: &LweList<InCont>,
    ) where
        Self: AsRefTensor<Element = Scalar>,
        LweList<InCont>: AsRefTensor<Element = Scalar>,
        GlweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        OutCont: Clone,
        Scalar: UnsignedTorus,
    {
        debug_assert!(input.count().0 <= output.polynomial_size().0);
        output.as_mut_tensor().fill_with_element(Scalar::ZERO);
        let mut buffer = output.clone();
        // for each ciphertext, call mono_key_switch
        for (degree, input_cipher) in input.ciphertext_iter().enumerate() {
            self.keyswitch_ciphertext(&mut buffer, &input_cipher);
            buffer
                .as_mut_polynomial_list()
                .polynomial_iter_mut()
                .for_each(|mut poly| {
                    poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(degree))
                });
            output
                .as_mut_tensor()
                .update_with_wrapping_add(buffer.as_tensor());
        }
    }
}

/// The encryption of a single bit of the output key.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub(crate) struct LweKeyBitDecomposition<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) glwe_size: GlweSize,
    pub(crate) poly_size: PolynomialSize,
}

tensor_traits!(LweKeyBitDecomposition);

impl<Cont> LweKeyBitDecomposition<Cont> {
    /// Creates a key bit decomposition from a container.
    ///
    /// # Notes
    ///
    /// This method does not decompose a key bit in a basis, but merely wraps a container in the
    /// right structure. See [`PackingKeyswitchKey::bit_decomp_iter`] for an iterator that returns
    /// key bit decompositions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.count(), CiphertextCount(15));
    /// assert_eq!(kbd.lwe_size(), LweSize(10));
    /// ```
    pub fn from_container(cont: Cont, glwe_size: GlweSize, poly_size: PolynomialSize) -> Self
    where
        Tensor<Cont>: AsRefSlice,
    {
        LweKeyBitDecomposition {
            tensor: Tensor::from_container(cont),
            glwe_size,
            poly_size,
        }
    }

    /// Returns the size of the lwe ciphertexts encoding each level of the key bit decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.lwe_size(), LweSize(10));
    /// ```
    #[allow(dead_code)]
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the number of ciphertexts in the decomposition.
    ///
    /// Note that this is actually equals to the number of levels in the decomposition.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// assert_eq!(kbd.count(), CiphertextCount(15));
    /// ```
    #[allow(dead_code)]
    pub fn count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.glwe_size.0 * self.poly_size.0);
        CiphertextCount(self.as_tensor().len() / (self.glwe_size.0 * self.poly_size.0))
    }

    /// Returns an iterator over borrowed `LweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// for ciphertext in kbd.ciphertext_iter(){
    ///     assert_eq!(ciphertext.lwe_size(), LweSize(10));
    /// }
    /// assert_eq!(kbd.ciphertext_iter().count(), 15);
    /// ```
    #[allow(dead_code)]
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = GlweCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.glwe_size.0 * self.poly_size.0)
            .map(move |sub| GlweCiphertext::from_container(sub.into_container(), self.poly_size))
    }

    /// Returns an iterator over mutably borrowed `LweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// use concrete_core::backends::core::private::math::tensor::{AsRefTensor, AsMutTensor};
    /// let mut kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// for mut ciphertext in kbd.ciphertext_iter_mut(){
    ///     ciphertext.as_mut_tensor().fill_with_element(9);
    /// }
    /// assert!(kbd.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(kbd.ciphertext_iter().count(), 15);
    /// ```
    #[allow(dead_code)]
    pub fn ciphertext_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GlweCiphertext<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.glwe_size.0 * self.poly_size.0;
        let poly_size = self.poly_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |sub| GlweCiphertext::from_container(sub.into_container(), poly_size))
    }

    /// Consumes the current key bit decomposition and returns an lwe list.
    ///
    /// Note that this operation is super cheap, as it merely rewraps the current container in an
    /// lwe list structure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use concrete_core::backends::core::private::crypto::{*, glwe::LweKeyBitDecomposition};
    /// let kbd = LweKeyBitDecomposition::from_container(vec![0 as u8; 150], LweSize(10));
    /// let list = kbd.into_lwe_list();
    /// assert_eq!(list.count(), CiphertextCount(15));
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn into_glwe_list(self) -> GlweList<Cont> {
        GlweList {
            tensor: self.tensor,
            rlwe_size: self.glwe_size,
            poly_size: self.poly_size,
        }
    }
}

fn torus_small_sign_decompose<Scalar>(res: &mut [Scalar], val: Scalar, base_log: usize)
where
    Scalar: UnsignedTorus,
    Scalar::Signed: SignedInteger,
{
    let mut tmp: Scalar;
    let mut carry = Scalar::ZERO;
    let mut previous_carry: Scalar;
    let block_bit_mask: Scalar = (Scalar::ONE << base_log) - Scalar::ONE;
    let msb_block_mask: Scalar = Scalar::ONE << (base_log - 1);

    // compute the decomposition from LSB to MSB (because of the carry)
    for i in (0..res.len()).rev() {
        previous_carry = carry;
        tmp = (val >> (Scalar::BITS - base_log * (i + 1))) & block_bit_mask;
        carry = tmp & msb_block_mask;
        tmp = tmp.wrapping_add(previous_carry);
        carry |= tmp & msb_block_mask; // 0000...0001000 or 0000...0000000
        res[i] = ((tmp.into_signed()) - ((carry << 1).into_signed())).into_unsigned();
        carry >>= base_log - 1; // 000...0001 or 000...0000
    }
}
