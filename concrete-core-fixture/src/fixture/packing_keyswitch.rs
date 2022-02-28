use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGlweCiphertext, PrototypesGlweSecretKey, PrototypesLweCiphertextVector,
    PrototypesLweSecretKey, PrototypesPackingKeyswitchKey, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesGlweCiphertext, SynthesizesLweCiphertextVector, SynthesizesPackingKeyswitchKey,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::prelude::{
    GlweCiphertextEntity, LweCiphertextVectorEntity,
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine, PackingKeyswitchKeyEntity,
};

/// A fixture for the types implementing the
/// `LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine` trait.
pub struct LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchFixture;

#[derive(Debug)]
pub struct LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub decomposition_level: DecompositionLevelCount,
    pub decomposition_base_log: DecompositionBaseLog,
}

impl<Precision, Engine, InputCiphertextVector, KeyswitchKey, OutputCiphertext>
    Fixture<Precision, Engine, (InputCiphertextVector, KeyswitchKey, OutputCiphertext)>
    for LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchFixture
where
    Precision: IntegerPrecision,
    Engine: LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine<
        KeyswitchKey,
        InputCiphertextVector,
        OutputCiphertext,
    >,
    InputCiphertextVector:
        LweCiphertextVectorEntity<KeyDistribution = KeyswitchKey::InputKeyDistribution>,
    KeyswitchKey: PackingKeyswitchKeyEntity,
    OutputCiphertext: GlweCiphertextEntity<KeyDistribution = KeyswitchKey::OutputKeyDistribution>,
    Maker: SynthesizesLweCiphertextVector<Precision, InputCiphertextVector>
        + SynthesizesGlweCiphertext<Precision, OutputCiphertext>
        + SynthesizesPackingKeyswitchKey<Precision, KeyswitchKey>,
{
    type Parameters = LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters;
    type RepetitionPrototypes = (
        <Maker as PrototypesPackingKeyswitchKey<
            Precision,
            KeyswitchKey::InputKeyDistribution,
            KeyswitchKey::OutputKeyDistribution,
        >>::PackingKeyswitchKeyProto,
        <Maker as PrototypesLweSecretKey<Precision, KeyswitchKey::InputKeyDistribution>>::LweSecretKeyProto,
        <Maker as PrototypesGlweSecretKey<Precision, KeyswitchKey::OutputKeyDistribution>>::GlweSecretKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        <Maker as PrototypesLweCiphertextVector<Precision, KeyswitchKey::InputKeyDistribution>>::LweCiphertextVectorProto,
        <Maker as PrototypesGlweCiphertext<Precision, KeyswitchKey::OutputKeyDistribution>>::GlweCiphertextProto,
    );
    type PreExecutionContext = (OutputCiphertext, InputCiphertextVector, KeyswitchKey);
    type PostExecutionContext = (OutputCiphertext, InputCiphertextVector, KeyswitchKey);
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(200),
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(256),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(200),
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(256),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(400),
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(256),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(200),
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(512),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(400),
                    glwe_dimension: GlweDimension(1),
                    polynomial_size: PolynomialSize(512),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
                LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(400),
                    glwe_dimension: GlweDimension(2),
                    polynomial_size: PolynomialSize(512),
                    decomposition_level: DecompositionLevelCount(3),
                    decomposition_base_log: DecompositionBaseLog(7),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key_input = maker.new_lwe_secret_key(parameters.lwe_dimension);
        let proto_secret_key_output =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.polynomial_size);
        let proto_ksk = maker.new_packing_keyswitch_key(
            &proto_secret_key_input,
            &proto_secret_key_output,
            parameters.decomposition_level,
            parameters.decomposition_base_log,
            parameters.noise,
        );
        (proto_ksk, proto_secret_key_input, proto_secret_key_output)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let (_, proto_input_secret_key, _) = repetition_proto;
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.polynomial_size.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector.as_slice());
        let proto_input_ciphertext_vector = <Maker as PrototypesLweCiphertextVector<
            Precision,
            InputCiphertextVector::KeyDistribution,
        >>::encrypt_plaintext_vector_to_lwe_ciphertext_vector(
            maker,
            proto_input_secret_key,
            &proto_plaintext_vector,
            parameters.noise,
        );
        let proto_output_ciphertext = <Maker as PrototypesGlweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::trivial_encrypt_zeros_to_glwe_ciphertext(
            maker,
            parameters.glwe_dimension,
            parameters.polynomial_size,
        );
        (
            proto_plaintext_vector,
            proto_input_ciphertext_vector,
            proto_output_ciphertext,
        )
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_keyswitch_key, ..) = repetition_proto;
        let (_, proto_input_ciphertext_vector, proto_output_ciphertext) = sample_proto;
        let synth_keywsitch_key = maker.synthesize_packing_keyswitch_key(proto_keyswitch_key);
        let synth_input_ciphertext_vector =
            maker.synthesize_lwe_ciphertext_vector(proto_input_ciphertext_vector);
        let synth_output_ciphertext = maker.synthesize_glwe_ciphertext(proto_output_ciphertext);
        (
            synth_output_ciphertext,
            synth_input_ciphertext_vector,
            synth_keywsitch_key,
        )
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (mut output_ciphertext, input_ciphertext_vector, ksk) = context;
        unsafe {
            engine.discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(
                &mut output_ciphertext,
                &input_ciphertext_vector,
                &ksk,
            );
        };
        (output_ciphertext, input_ciphertext_vector, ksk)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (output_ciphertext, input_ciphertext, keyswitch_key) = context;
        let (_, _, proto_output_secret_key) = repetition_proto;
        let (proto_plaintext, ..) = sample_proto;
        let proto_output_ciphertext = maker.unsynthesize_glwe_ciphertext(&output_ciphertext);
        let proto_output_plaintext = <Maker as PrototypesGlweCiphertext<
            Precision,
            OutputCiphertext::KeyDistribution,
        >>::decrypt_glwe_ciphertext_to_plaintext_vector(
            maker,
            proto_output_secret_key,
            &proto_output_ciphertext,
        );
        maker.destroy_lwe_ciphertext_vector(input_ciphertext);
        maker.destroy_glwe_ciphertext(output_ciphertext);
        maker.destroy_packing_keyswitch_key(keyswitch_key);
        (
            maker.transform_plaintext_vector_to_raw_vec(proto_plaintext),
            maker.transform_plaintext_vector_to_raw_vec(&proto_output_plaintext),
        )
    }

    fn check_sample_outcomes(parameters: &Self::Parameters, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        let means = means
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .collect::<Vec<_>>();
        let actual = actual
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .collect::<Vec<_>>();
        assert_noise_distribution(&actual, means.as_slice(), parameters.noise)
    }
}
