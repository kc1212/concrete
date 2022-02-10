use crate::generation::prototyping::{PrototypesGgswCiphertext, PrototypesGlweCiphertext};
use crate::generation::IntegerPrecision;
use concrete_core::prelude::{GgswCiphertextEntity, GlweCiphertextEntity};

pub trait SynthesizesGgswCiphertext<Precision: IntegerPrecision, GgswCiphertext>:
    PrototypesGgswCiphertext<Precision, GgswCiphertext::KeyDistribution>
where
    GgswCiphertext: GgswCiphertextEntity,
{
    fn synthesize_ggsw_ciphertext(
        &mut self,
        prototype: &Self::GgswCiphertextProto,
    ) -> GgswCiphertext;
    fn unsynthesize_ggsw_ciphertext(
        &mut self,
        entity: &GgswCiphertext,
    ) -> Self::GgswCiphertextProto;
    fn destroy_ggsw_ciphertext(&mut self, entity: GgswCiphertext);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoBinaryGgswCiphertext32, ProtoBinaryGgswCiphertext64};
    use crate::generation::synthesizing::SynthesizesGgswCiphertext;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{
        DestructionEngine, GgswCiphertext32, GgswCiphertext64, GgswCiphertextComplex64,
        GgswCiphertextConversionEngine,
    };

    impl SynthesizesGgswCiphertext<Precision64, GgswCiphertextComplex64> for Maker {
        fn synthesize_ggsw_ciphertext(
            &mut self,
            prototype: &Self::GgswCiphertextProto,
        ) -> GgswCiphertextComplex64 {
            self.core_engine
                .convert_ggsw_ciphertext(&prototype.0)
                .unwrap()
        }

        fn unsynthesize_ggsw_ciphertext(
            &mut self,
            entity: &GgswCiphertextComplex64,
        ) -> Self::GgswCiphertextProto {
            todo!()
        }

        fn destroy_ggsw_ciphertext(&mut self, entity: GgswCiphertextComplex64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
