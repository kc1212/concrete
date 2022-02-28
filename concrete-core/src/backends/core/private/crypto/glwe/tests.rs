use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::glwe::GlweList;
use crate::backends::core::private::crypto::lwe::LweList;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, UniformMsb};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{
    self, assert_delta_std_dev, assert_noise_distribution,
};
use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension,
    PlaintextCount, PolynomialSize,
};
use concrete_npe as npe;

use super::{GlweCiphertext, PackingKeyswitchKey};

fn test_glwe<T: UnsignedTorus>() {
    // random settings
    let nb_ct = test_tools::random_ciphertext_count(200);
    let dimension = test_tools::random_glwe_dimension(200);
    let polynomial_size = test_tools::random_polynomial_size(200);
    let noise_parameter = LogStandardDev::from_log_standard_dev(-20.);
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // generates a secret key
    let sk = GlweSecretKey::generate_binary(dimension, polynomial_size, &mut secret_generator);

    // generates random plaintexts
    let plaintexts = PlaintextList::from_tensor(
        random_generator.random_uniform_tensor(nb_ct.0 * polynomial_size.0),
    );

    // encrypts
    let mut ciphertext = GlweList::allocate(T::ZERO, polynomial_size, dimension, nb_ct);
    sk.encrypt_glwe_list(
        &mut ciphertext,
        &plaintexts,
        noise_parameter,
        &mut encryption_generator,
    );

    // decrypts
    let mut decryptions = PlaintextList::from_tensor(
        random_generator.random_uniform_tensor(nb_ct.0 * polynomial_size.0),
    );
    sk.decrypt_glwe_list(&mut decryptions, &ciphertext);

    // test
    assert_delta_std_dev(&plaintexts, &decryptions, noise_parameter);
}

#[test]
fn test_glwe_encrypt_decrypt_u32() {
    test_glwe::<u32>();
}

#[test]
fn test_glwe_encrypt_decrypt_u64() {
    test_glwe::<u64>();
}

fn test_packing_keyswitch<T: UnsignedTorus + RandomGenerable<UniformMsb>>() {
    //! create a KSK and key switch some LWE samples
    //! warning: not a randomized test for the parameters
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // fix a set of parameters
    let n_bit_msg = 8; // bit precision of the plaintext
    let nb_ct = CiphertextCount(256); // number of messages to encrypt
    let base_log = DecompositionBaseLog(3); // a parameter of the gadget matrix
    let level_count = DecompositionLevelCount(8); // a parameter of the gadget matrix
    let polynomial_size = PolynomialSize(256);
    let messages = PlaintextList::from_tensor(
        random_generator.random_uniform_n_msb_tensor(nb_ct.0, n_bit_msg),
    );
    // the set of messages to encrypt
    let std_input = LogStandardDev::from_log_standard_dev(-10.); // standard deviation of the
                                                                 // encrypted messages to KS
    let std_ksk = LogStandardDev::from_log_standard_dev(-25.); // standard deviation of the ksk

    // set parameters related to the after (stands for 'after the KS')
    let dimension_after = GlweDimension(1);
    let sk_after =
        GlweSecretKey::generate_binary(dimension_after, polynomial_size, &mut secret_generator);

    // set parameters related to the before (stands for 'before the KS')
    let dimension_before = LweDimension(630);
    let sk_before = LweSecretKey::generate_binary(dimension_before, &mut secret_generator);

    // create the before ciphertexts and the after ciphertexts
    let mut ciphertexts_before = LweList::allocate(T::ZERO, dimension_before.to_lwe_size(), nb_ct);
    let mut ciphertext_after =
        GlweCiphertext::allocate(T::ZERO, polynomial_size, dimension_after.to_glwe_size());

    // key switching key generation
    let mut ksk = PackingKeyswitchKey::allocate(
        T::ZERO,
        level_count,
        base_log,
        dimension_before,
        dimension_after,
        polynomial_size,
    );
    ksk.fill_with_keyswitch_key(&sk_before, &sk_after, std_ksk, &mut encryption_generator);

    // encrypts with the before key our messages
    sk_before.encrypt_lwe_list(
        &mut ciphertexts_before,
        &messages,
        std_input,
        &mut encryption_generator,
    );

    // key switch before -> after
    ksk.packing_keyswitch(&mut ciphertext_after, &ciphertexts_before);

    // decryption with the after key
    let mut dec_messages = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    sk_after.decrypt_glwe(&mut dec_messages, &ciphertext_after);

    // calls the NPE to find out the amount of noise after KS
    let output_variance = npe::estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms::<
        T,
        _,
        _,
        BinaryKeyKind,
    >(dimension_before, std_input, std_ksk, base_log, level_count);

    assert_noise_distribution(&messages, &dec_messages, output_variance);
}

#[test]
fn test_packing_keyswitch_u32() {
    test_packing_keyswitch::<u32>();
}

#[test]
fn test_packing_keyswitch_u64() {
    test_packing_keyswitch::<u64>();
}
