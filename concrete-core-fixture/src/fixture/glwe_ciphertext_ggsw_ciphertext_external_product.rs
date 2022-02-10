use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesGgswCiphertext, PrototypesGlweCiphertext, PrototypesGlweSecretKey,
    PrototypesPlaintext, PrototypesPlaintextVector,
};
use crate::generation::synthesizing::{SynthesizesGgswCiphertext, SynthesizesGlweCiphertext};
use crate::generation::{IntegerPrecision, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::SampleSize;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
};
use concrete_core::backends::core::private::math::polynomial::Polynomial;
use concrete_core::prelude::{
    GgswCiphertextEntity, GlweCiphertextEntity, GlweCiphertextGgswCiphertextExternalProductEngine,
};

pub struct GlweCiphertextGgswCiphertextExternalProductFixture;

#[derive(Debug)]
pub struct GlweCiphertextGgswCiphertextExternalProductParameters {
    pub ggsw_noise: Variance,
    pub glwe_noise: Variance,
    pub glwe_dimension: GlweDimension,
    pub poly_size: PolynomialSize,
    pub dec_level_count: DecompositionLevelCount,
    pub dec_base_log: DecompositionBaseLog,
}

impl<Precision, Engine, GlweCiphertext, GgswCiphertext, OutputGlweCiphertext>
    Fixture<Precision, Engine, (GlweCiphertext, GgswCiphertext, OutputGlweCiphertext)>
    for GlweCiphertextGgswCiphertextExternalProductFixture
where
    Precision: IntegerPrecision,
    Engine: GlweCiphertextGgswCiphertextExternalProductEngine<
        GlweCiphertext,
        GgswCiphertext,
        OutputGlweCiphertext,
    >,
    GlweCiphertext: GlweCiphertextEntity,
    GgswCiphertext: GgswCiphertextEntity<KeyDistribution = GlweCiphertext::KeyDistribution>,
    OutputGlweCiphertext: GlweCiphertextEntity<KeyDistribution = GlweCiphertext::KeyDistribution>,
    Maker: SynthesizesGlweCiphertext<Precision, GlweCiphertext>
        + SynthesizesGgswCiphertext<Precision, GgswCiphertext>
        + SynthesizesGlweCiphertext<Precision, OutputGlweCiphertext>,
{
    type Parameters = GlweCiphertextGgswCiphertextExternalProductParameters;
    type RawInputs = (Precision::Raw, Vec<Precision::Raw>);
    type RawOutputs = (Vec<Precision::Raw>,);
    type Bypass = (<Maker as PrototypesGlweSecretKey<Precision, GlweCiphertext::KeyDistribution>>::GlweSecretKeyProto, );
    type PreExecutionContext = (GlweCiphertext, GgswCiphertext);
    type PostExecutionContext = (GlweCiphertext, GgswCiphertext, OutputGlweCiphertext);
    type Prediction = ();

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        todo!()
    }

    fn generate_random_raw_inputs(parameters: &Self::Parameters) -> Self::RawInputs {
        todo!()
    }

    fn compute_prediction(
        parameters: &Self::Parameters,
        raw_inputs: &Self::RawInputs,
        sample_size: SampleSize,
    ) -> Self::Prediction {
        todo!()
    }

    fn check_prediction(
        _parameters: &Self::Parameters,
        forecast: &Self::Prediction,
        actual: &[Self::RawOutputs],
    ) -> bool {
        todo!()
    }

    fn prepare_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        raw_inputs: &Self::RawInputs,
    ) -> (Self::Bypass, Self::PreExecutionContext) {
        let (raw_plaintext, raw_plaintext_vector) = raw_inputs;
        let proto_plaintext = maker.transform_raw_to_plaintext(raw_plaintext);
        let proto_plaintext_vector =
            maker.transform_raw_vec_to_plaintext_vector(raw_plaintext_vector);
        let proto_secret_key =
            maker.new_glwe_secret_key(parameters.glwe_dimension, parameters.poly_size);
        let proto_glwe_ciphertext = maker.encrypt_plaintext_vector_to_glwe_ciphertext(
            &proto_secret_key,
            &proto_plaintext_vector,
            parameters.glwe_noise,
        );
        let proto_ggsw_ciphertext = maker.encrypt_plaintext_to_ggsw_ciphertext(
            &proto_secret_key,
            &proto_plaintext,
            parameters.dec_level_count,
            parameters.dec_base_log,
            parameters.ggsw_noise,
        );
        let synth_glwe_ciphertext = maker.synthesize_glwe_ciphertext(&proto_glwe_ciphertext);
        let synth_ggsw_ciphertext = maker.synthesize_ggsw_ciphertext(&proto_ggsw_ciphertext);
        (
            (proto_secret_key,),
            (synth_glwe_ciphertext, synth_ggsw_ciphertext),
        )
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (glwe_ciphertext, ggsw_ciphertext) = context;
        let output_ciphertext = unsafe {
            engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                &glwe_ciphertext,
                &ggsw_ciphertext,
            )
        };
        (glwe_ciphertext, ggsw_ciphertext, output_ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        bypass: Self::Bypass,
        context: Self::PostExecutionContext,
    ) -> Self::RawOutputs {
        let (glwe_ciphertext, ggsw_ciphertext, output_glwe_ciphertext) = context;
        let (proto_secret_key,) = bypass;
        let proto_output_ciphertext = maker.unsynthesize_glwe_ciphertext(&output_glwe_ciphertext);
        maker.destroy_glwe_ciphertext(output_glwe_ciphertext);
        maker.destroy_glwe_ciphertext(glwe_ciphertext);
        maker.destroy_ggsw_ciphertext(ggsw_ciphertext);
        let decryption = maker.decrypt_glwe_ciphertext_to_plaintext_vector(
            &proto_secret_key,
            &proto_output_ciphertext,
        );
        (maker.transform_plaintext_vector_to_raw_vec(&decryption),)
    }
}
