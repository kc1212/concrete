use crate::generation::prototypes::{
    GgswCiphertextPrototype, ProtoBinaryGgswCiphertext32, ProtoBinaryGgswCiphertext64,
};
use crate::generation::prototyping::{PrototypesGlweSecretKey, PrototypesPlaintext};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::GgswCiphertextEncryptionEngine;

pub trait PrototypesGgswCiphertext<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>: PrototypesPlaintext<Precision> + PrototypesGlweSecretKey<Precision, KeyDistribution>
{
    type GgswCiphertextProto: GgswCiphertextPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn encrypt_plaintext_to_ggsw_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext: &Self::PlaintextProto,
        dec_level_count: DecompositionLevelCount,
        dec_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::GgswCiphertextProto;

    fn decrypt_ggsw_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GgswCiphertextProto,
    ) -> Self::PlaintextProto;
}

impl PrototypesGgswCiphertext<Precision32, BinaryKeyDistribution> for Maker {
    type GgswCiphertextProto = ProtoBinaryGgswCiphertext32;

    fn encrypt_plaintext_to_ggsw_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext: &Self::PlaintextProto,
        dec_level_count: DecompositionLevelCount,
        dec_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::GgswCiphertextProto {
        ProtoBinaryGgswCiphertext32(
            self.core_engine
                .encrypt_ggsw_ciphertext(
                    &secret_key.0,
                    &plaintext.0,
                    noise,
                    dec_level_count,
                    dec_base_log,
                )
                .unwrap(),
        )
    }

    fn decrypt_ggsw_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GgswCiphertextProto,
    ) -> Self::PlaintextProto {
        todo!()
    }
}

impl PrototypesGgswCiphertext<Precision64, BinaryKeyDistribution> for Maker {
    type GgswCiphertextProto = ProtoBinaryGgswCiphertext64;

    fn encrypt_plaintext_to_ggsw_ciphertext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        plaintext: &Self::PlaintextProto,
        dec_level_count: DecompositionLevelCount,
        dec_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::GgswCiphertextProto {
        ProtoBinaryGgswCiphertext64(
            self.core_engine
                .encrypt_ggsw_ciphertext(
                    &secret_key.0,
                    &plaintext.0,
                    noise,
                    dec_level_count,
                    dec_base_log,
                )
                .unwrap(),
        )
    }

    fn decrypt_ggsw_ciphertext_to_plaintext(
        &mut self,
        secret_key: &Self::GlweSecretKeyProto,
        ciphertext: &Self::GgswCiphertextProto,
    ) -> Self::PlaintextProto {
        todo!()
    }
}
