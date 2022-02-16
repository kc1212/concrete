use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGgswCiphertext, PrototypesGlweCiphertext, PrototypesGlweSecretKey,
    PrototypesPlaintext, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{
    SynthesizesGgswCiphertext, SynthesizesGlweCiphertext, SynthesizesGlweSecretKey,
    SynthesizesPlaintextVector,
};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};
use concrete_core::prelude::{
    GgswCiphertextEntity, GlweCiphertextEntity, GlweCiphertextGgswCiphertextExternalProductEngine,
    GlweSecretKeyEntity, PlaintextVectorEntity,
};

/// A fixture for the types implementing the `GlweCiphertextGgswCiphertextExternalProduct` trait.
pub struct GlweCiphertextGgswCiphertextExternalProductFixture;

#[derive(Debug)]
pub struct GlweCiphertextGgswCiphertextExternalProductParameters {
    pub noise: Variance,
    pub glwe_dimension: GlweDimension,
    pub polynomial_size: PolynomialSize,
    pub decomposition_base_log: DecompositionBaseLog,
    pub decomposition_level_count: DecompositionLevelCount,
}

impl<Precision, Engine, GlweInput, GgswInput, Output, SecretKey, PlaintextVector>
    Fixture<Precision, Engine, (GlweInput, GgswInput, Output, SecretKey, PlaintextVector)>
    for GlweCiphertextGgswCiphertextExternalProductFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextGgswCiphertextExternalProductEngine<GlweInput, GgswInput, Output>,
    GlweInput: GlweCiphertextEntity,
    GgswInput: GgswCiphertextEntity<KeyDistribution = GlweInput::KeyDistribution>,
    Output: GlweCiphertextEntity<KeyDistribution = GlweInput::KeyDistribution>,
    SecretKey: GlweSecretKeyEntity<KeyDistribution = GlweInput::KeyDistribution>,
    PlaintextVector: PlaintextVectorEntity,
    Maker: SynthesizesGlweSecretKey<Precision, SecretKey>
        + SynthesizesGlweCiphertext<Precision, GlweInput>
        + SynthesizesGlweCiphertext<Precision, Output>
        + SynthesizesGgswCiphertext<Precision, GgswInput>
        + SynthesizesPlaintextVector<Precision, PlaintextVector>,
{
    type Parameters = GlweCiphertextGgswCiphertextExternalProductParameters;
    type Outcome = (Vec<Precision::Raw>, Vec<Precision::Raw>);
    type RepetitionPrototypes = (
        <Maker as PrototypesGlweSecretKey<Precision, GlweInput::KeyDistribution>>::GlweSecretKeyProto,
    );
    type SamplePrototypes = (
        <Maker as PrototypesPlaintextVector<Precision>>::PlaintextVectorProto,
        Vec<Precision::Raw>,
    );
    type PreExecutionContext = (GlweInput, GgswInput);
    type PostExecutionContext = Output;

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![GlweCiphertextGgswCiphertextExternalProductParameters {
                noise: Variance(0.00000001),
                glwe_dimension: GlweDimension(200),
                polynomial_size: PolynomialSize(200),
                decomposition_base_log: DecompositionBaseLog(8),
                decomposition_level_count: DecompositionLevelCount(4),
            }]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.polynomial_size);
        (proto_secret_key,)
    }

    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let raw_plaintext_vector = Precision::Raw::uniform_vec(parameters.polynomial_size.0);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(&raw_plaintext_vector);
        (proto_plaintext_vector, raw_plaintext_vector)
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

    fn prepare_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_plaintext_vector, _) = sample_proto;
        let (proto_secret_key,) = repetition_proto;
        let proto_glwe_input = maker.encrypt_plaintext_vector_to_glwe_ciphertext(
            proto_secret_key,
            proto_plaintext_vector,
            parameters.noise,
        );
        let proto_plaintext_one = maker.transform_raw_to_plaintext(&Precision::Raw::one());
        let proto_ggsw_input = maker.encrypt_plaintext_to_ggsw_ciphertext(
            proto_secret_key,
            &proto_plaintext_one,
            parameters.noise,
            parameters.decomposition_level_count,
            parameters.decomposition_base_log,
        );
        let synth_glwe_input = maker.synthesize_glwe_ciphertext(&proto_glwe_input);
        let synth_ggsw_input = maker.synthesize_ggsw_ciphertext(&proto_ggsw_input);
        (synth_glwe_input, synth_ggsw_input)
    }

    fn execute_engine(
        _parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (glwe_input, ggsw_input) = context;
        unsafe {
            engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                &glwe_input,
                &ggsw_input,
            )
        }
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let ciphertext = context;
        let (_, raw_plaintext_vector) = sample_proto;
        let (proto_secret_key,) = repetition_proto;
        let proto_output_ciphertext = maker.unsynthesize_glwe_ciphertext(&ciphertext);
        maker.destroy_glwe_ciphertext(ciphertext);
        let proto_plaintext_vector = maker.decrypt_glwe_ciphertext_to_plaintext_vector(
            proto_secret_key,
            &proto_output_ciphertext,
        );
        (
            raw_plaintext_vector.clone(),
            maker.transform_plaintext_vector_to_raw_vec(&proto_plaintext_vector),
        )
    }
}
