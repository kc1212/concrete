//! The public key for homomorphic computation.
//!
//! This module implements the generation of the server's public key, together with all the
//! available homomorphic Boolean gates ($\mathrm{AND}$, $\mathrm{MUX}$, $\mathrm{NAND}$,
//! $\mathrm{NOR}$,
//! $\mathrm{NOT}$, $\mathrm{OR}$, $\mathrm{XNOR}$, $\mathrm{XOR}$).

#[cfg(test)]
mod tests;

use crate::ciphertext::Ciphertext;
use crate::client_key::ClientKey;
use crate::{PLAINTEXT_LOG_SCALING_FACTOR, PLAINTEXT_TRUE};
use concrete_commons::parameters::LweDimension;
use concrete_core::crypto::bootstrap::{Bootstrap, FourierBootstrapKey, StandardBootstrapKey};
use concrete_core::crypto::encoding::Cleartext;
use concrete_core::crypto::glwe::GlweCiphertext;
use concrete_core::crypto::lwe::{LweCiphertext, LweKeyswitchKey};
use concrete_core::crypto::secret::generators::EncryptionRandomGenerator;
use concrete_core::math::fft::{AlignedVec, Complex64};
use concrete_core::math::tensor::AsMutTensor;
use serde::{Deserialize, Serialize};

/// A structure containing the server public key.
///
/// The server key is generated by the client and is meant to be published: the client
/// sends it to the server so it can compute homomorphic Boolean circuits.
///
/// In more details, it contains:
/// * `key_switching_key` - a public key, used to perform the key-switching operation.
/// * `bootstrapping_key` - a public key, used to perform the bootstrapping operation.
#[derive(Serialize, Clone, Deserialize, PartialEq, Debug)]
pub struct ServerKey {
    pub key_switching_key: LweKeyswitchKey<Vec<u32>>,
    pub bootstrapping_key: FourierBootstrapKey<AlignedVec<Complex64>, u32>,
}

impl ServerKey {
    /// Allocates and generates a server key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::client_key::ClientKey;
    /// use concrete_boolean::parameters::DEFAULT_PARAMETERS;
    /// use concrete_boolean::server_key::ServerKey;
    ///
    /// // Generate the client key:
    /// let cks = ClientKey::new(&DEFAULT_PARAMETERS);
    ///
    /// // Generate the server key:
    /// let sks = ServerKey::new(&cks);
    /// ```
    pub fn new(cks: &ClientKey) -> ServerKey {
        // Allocate and generate the key in coefficient domain:
        let mut coef_bsk = StandardBootstrapKey::allocate(
            0_u32,
            cks.parameters.glwe_dimension.to_glwe_size(),
            cks.parameters.polynomial_size,
            cks.parameters.pbs_level,
            cks.parameters.pbs_base_log,
            cks.parameters.lwe_dimension,
        );
        let mut encryption_generator = EncryptionRandomGenerator::new(None);
        coef_bsk.par_fill_with_new_key(
            &cks.lwe_secret_key,
            &cks.glwe_secret_key,
            cks.parameters.glwe_modular_std_dev,
            &mut encryption_generator,
        );

        // Allocate the bootstrapping key in Fourier domain and forward FFT:
        let mut fourier_bsk = FourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            cks.parameters.glwe_dimension.to_glwe_size(),
            cks.parameters.polynomial_size,
            cks.parameters.pbs_level,
            cks.parameters.pbs_base_log,
            cks.parameters.lwe_dimension,
        );
        fourier_bsk.fill_with_forward_fourier(&coef_bsk);

        // Allocate the key switching key:
        let mut ksk = LweKeyswitchKey::allocate(
            0_u32,
            cks.parameters.ks_level,
            cks.parameters.ks_base_log,
            LweDimension(cks.parameters.glwe_dimension.0 * cks.parameters.polynomial_size.0),
            cks.parameters.lwe_dimension,
        );

        // Convert the GLWE secret key into an LWE secret key:
        let big_lwe_secret_key = cks.glwe_secret_key.clone().into_lwe_secret_key();

        // Fill the key switching key:
        ksk.fill_with_keyswitch_key(
            &big_lwe_secret_key,
            &cks.lwe_secret_key,
            cks.parameters.lwe_modular_std_dev,
            &mut encryption_generator,
        );

        // Pack the keys in the server key set:
        let sks: ServerKey = ServerKey {
            key_switching_key: ksk,
            bootstrapping_key: fourier_bsk,
        };
        sks
    }

    /// Computes homomorphically an AND gate between two ciphertexts encrypting Boolean values:
    /// $$ ct_{out} = ct_{left}~\mathrm{AND}~ct_{right} $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt two messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    ///
    /// // Compute homomorphically an AND gate:
    /// let ct_res = sks.and(&ct1, &ct2);
    ///
    /// // Decrypt:
    /// let dec_and = cks.decrypt(&ct_res);
    /// assert_eq!(false, dec_and);
    /// ```
    pub fn and(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for AND: ct_left + ct_right + (0,...,0,-1/8)
        let mut ct_temp = ct_left.0.clone();
        ct_temp.update_with_add(&ct_right.0);
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_sub(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // -1/8

        // Create the accumulator
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate the output of the PBS
        let mut ct_pbs = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the programmable bootstrapping with fixed test polynomial
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs, &ct_temp, &accumulator);

        // Compute a key switch to get back to input key
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_left.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_pbs);

        // Result
        Ciphertext(ct_ks)
    }

    /// Computes an homomorphic MUX gate between three ciphertexts encrypting Boolean values:
    /// $$ct_{out} = (ct_{condition}?~ct_{then}:~ct_{else}) $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt three messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    /// let ct3 = cks.encrypt(true);
    ///
    /// // Compute homomorphically a MUX gate:
    /// let ct_res = sks.mux(&ct1, &ct2, &ct3);
    ///
    /// // Decrypt:
    /// let dec_mux = cks.decrypt(&ct_res);
    /// assert_eq!(false, dec_mux);
    /// ```
    pub fn mux(
        &self,
        ct_condition: &Ciphertext,
        ct_then: &Ciphertext,
        ct_else: &Ciphertext,
    ) -> Ciphertext {
        // In theory MUX gate = (ct_condition AND ct_then) + (!ct_condition AND ct_else)

        // Compute the linear combination for first AND: ct_condition + ct_then + (0,...,0,-1/8)
        let mut ct_temp_1 = ct_condition.0.clone();
        ct_temp_1.update_with_add(&ct_then.0);
        ct_temp_1.get_mut_body().0 = ct_temp_1
            .get_mut_body()
            .0
            .wrapping_sub(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // -1/8

        // Compute the linear combination for second AND: - ct_condition + ct_else + (0,...,0,-1/8)
        let mut ct_temp_2 = ct_condition.0.clone();
        ct_temp_2.update_with_neg();
        ct_temp_2.update_with_add(&ct_else.0);
        ct_temp_2.get_mut_body().0 = ct_temp_2
            .get_mut_body()
            .0
            .wrapping_sub(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // -1/8

        // Create the accumulator:
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate the output of the first PBS:
        let mut ct_pbs_1 = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Allocate the output of the second PBS:
        let mut ct_pbs_2 = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the first programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs_1, &ct_temp_1, &accumulator);

        // Compute the second programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs_2, &ct_temp_2, &accumulator);

        // Compute the linear combination to add the two results : ct_pbs_1 + ct_pbs_2 + (0,...,0,
        // +1/8)
        let mut ct_temp = ct_pbs_1;
        ct_temp.update_with_add(&ct_pbs_2);
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_add(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // +1/8

        // Compute the key switch to get back to input key
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_condition.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_temp);

        // Output the result:
        Ciphertext(ct_ks)
    }

    /// Computes homomorphically a NAND gate between two ciphertexts encrypting Boolean values:
    /// $$ct_{out} = \mathrm{NOT} (ct_{left}~\mathrm{AND}~ct_{right})$$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt two messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    ///
    /// // Compute homomorphically a NAND gate:
    /// let ct_res = sks.nand(&ct1, &ct2);
    ///
    /// // Decrypt:
    /// let dec_nand = cks.decrypt(&ct_res);
    /// assert_eq!(true, dec_nand);
    /// ```
    pub fn nand(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for NAND: - ct_left - ct_right + (0,...,0,1/8)
        let mut ct_temp = ct_left.0.clone();
        ct_temp.update_with_neg();
        ct_temp.update_with_sub(&ct_right.0);
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_add(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // 1/8

        // Create the accumulator:
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial:
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate the output of the PBS:
        let mut ct_pbs = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs, &ct_temp, &accumulator);

        // Compute the key switch to get back to input key:
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_left.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_pbs);

        // Output the result
        Ciphertext(ct_ks)
    }

    /// Computes homomorphically a NOR gate between two ciphertexts encrypting Boolean values:
    /// $$ ct_{out} = \mathrm{NOT}(ct_{left}~\mathrm{OR}~ct_{right}) $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt two messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    ///
    /// // Compute homomorphically the NOR gate:
    /// let ct_res = sks.nor(&ct1, &ct2);
    ///
    /// // Decrypt:
    /// let dec_nor = cks.decrypt(&ct_res);
    /// assert_eq!(false, dec_nor);
    /// ```
    pub fn nor(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for NOR: - ct_left - ct_right + (0,...,0,-1/8)
        let mut ct_temp = ct_left.0.clone();
        ct_temp.update_with_neg();
        ct_temp.update_with_sub(&ct_right.0);
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_sub(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // -1/8

        // Create the accumulator:
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial:
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate the output of the PBS:
        let mut ct_pbs = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the Programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs, &ct_temp, &accumulator);

        // Compute the key switch to get back to input key:
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_left.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_pbs);

        // Output the result:
        Ciphertext(ct_ks)
    }

    /// Computes homomorphically a NOT gate of a ciphertexts encrypting a Boolean value:
    /// $$ct_{out} = \mathrm{NOT}(ct_{in})$$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt a message:
    /// let ct = cks.encrypt(true);
    ///
    /// // Compute homomorphically a NOT gate:
    /// let ct_res = sks.not(&ct);
    ///
    /// // Decrypt:
    /// let dec_not = cks.decrypt(&ct_res);
    /// assert_eq!(false, dec_not);
    /// ```
    pub fn not(&self, ct: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for NOT: -ct
        let mut ct = ct.0.clone();
        ct.update_with_neg();

        // Output the result:
        Ciphertext(ct)
    }

    /// Computes homomorphically an OR gate between two ciphertexts encrypting Boolean values:
    /// $$ct_{out} = ct_{left}~\mathrm{OR}~ct_{right}$$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt two messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    ///
    /// // Compute homomorphically the OR gate:
    /// let ct_res = sks.or(&ct1, &ct2);
    ///
    /// // Decrypt:
    /// let dec_or = cks.decrypt(&ct_res);
    /// assert_eq!(true, dec_or);
    /// ```
    pub fn or(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for OR: ct_left + ct_right + (0,...,0,+1/8)
        let mut ct_temp = ct_left.0.clone();
        ct_temp.update_with_add(&ct_right.0);
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_add(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR)); // +1/8

        // Create the accumulator:
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial:
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate the output of the PBS:
        let mut ct_pbs = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs, &ct_temp, &accumulator);

        // Compute a key switch to get back to input key:
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_left.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_pbs);

        // Output the result:
        Ciphertext(ct_ks)
    }

    /// Computes homomorphically an XNOR gate (or equality test) between two ciphertexts encrypting
    /// Boolean values:
    /// $$ct_{out} = (ct_{left}~==~ct_{right}) $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encrypt two messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    ///
    /// // Compute the XNOR gate:
    /// let ct_res = sks.xnor(&ct1, &ct2);
    ///
    /// // Decrypt:
    /// let dec_xnor = cks.decrypt(&ct_res);
    /// assert_eq!(false, dec_xnor);
    /// ```
    pub fn xnor(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for XNOR: 2*(-ct_left - ct_right) + (0,...,0,-1/4)
        let mut ct_temp = ct_left.0.clone();
        ct_temp.update_with_neg();
        ct_temp.update_with_sub(&ct_right.0);
        ct_temp.update_with_scalar_mul(Cleartext(2));
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_sub(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR + 1)); // -1/4

        // Create the accumulator:
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial:
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate the output of the PBS:
        let mut ct_pbs = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute a programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs, &ct_temp, &accumulator);

        // Compute a key switching to get back to input key:
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_left.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_pbs);

        // Output the result:
        Ciphertext(ct_ks)
    }

    /// Computes homomorphically an XOR gate between two ciphertexts encrypting Boolean values:
    /// $$ct_{out}= ct_{left}~\mathrm{XOR}~ct_{right}$$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_boolean::gen_keys;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys();
    ///
    /// // Encryption of two messages:
    /// let ct1 = cks.encrypt(true);
    /// let ct2 = cks.encrypt(false);
    ///
    /// // Compute the XOR gate:
    /// let ct_res = sks.xor(&ct1, &ct2);
    ///
    /// // Decryption:
    /// let dec_xor = cks.decrypt(&ct_res);
    /// assert_eq!(true, dec_xor);
    /// ```
    pub fn xor(&self, ct_left: &Ciphertext, ct_right: &Ciphertext) -> Ciphertext {
        // Compute the linear combination for XOR: 2*(ct_left + ct_right) + (0,...,0,1/4)
        let mut ct_temp = ct_left.0.clone();
        ct_temp.update_with_add(&ct_right.0);
        ct_temp.update_with_scalar_mul(Cleartext(2));
        ct_temp.get_mut_body().0 = ct_temp
            .get_mut_body()
            .0
            .wrapping_add(1_u32 << (32 - PLAINTEXT_LOG_SCALING_FACTOR + 1)); // +1/4

        // Create the accumulator:
        let mut accumulator = GlweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.polynomial_size(),
            self.bootstrapping_key.glwe_size(),
        );

        // Fill the body of accumulator with the Test Polynomial:
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .fill_with_element(PLAINTEXT_TRUE); // 1/8

        // Allocate for the output of the PBS:
        let mut ct_pbs = LweCiphertext::allocate(
            0_u32,
            self.bootstrapping_key.output_lwe_dimension().to_lwe_size(),
        );

        // Compute the programmable bootstrapping with fixed test polynomial:
        self.bootstrapping_key
            .bootstrap(&mut ct_pbs, &ct_temp, &accumulator);

        // Compute the key switching to get back to input key:
        let mut ct_ks = LweCiphertext::allocate(0_u32, ct_left.0.lwe_size());
        self.key_switching_key
            .keyswitch_ciphertext(&mut ct_ks, &ct_pbs);

        // Output the result:
        Ciphertext(ct_ks)
    }
}
