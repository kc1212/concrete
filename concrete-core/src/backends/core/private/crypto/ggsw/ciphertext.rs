use crate::backends::core::private::crypto::bootstrap::FourierBskBuffers;
use crate::backends::core::private::crypto::encoding::Plaintext;
use crate::backends::core::private::crypto::glwe::{GlweCiphertext, GlweList};
use crate::backends::core::private::math::fft::{Complex64, FourierPolynomial};
use crate::backends::core::private::math::polynomial::Polynomial;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::utils::{zip, zip_args};

use super::GgswLevelMatrix;

use crate::backends::core::private::math::decomposition::{DecompositionLevel, SignedDecomposer};
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};

/// A GGSW ciphertext.
#[derive(Debug, Clone, PartialEq)]
pub struct GgswCiphertext<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    rlwe_size: GlweSize,
    decomp_base_log: DecompositionBaseLog,
}

tensor_traits!(GgswCiphertext);

impl<Scalar> GgswCiphertext<Vec<Scalar>> {
    /// Allocates a new GGSW ciphertext whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn allocate(
        value: Scalar,
        poly_size: PolynomialSize,
        rlwe_size: GlweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Copy,
    {
        GgswCiphertext {
            tensor: Tensor::from_container(vec![
                value;
                decomp_level.0
                    * rlwe_size.0
                    * rlwe_size.0
                    * poly_size.0
            ]),
            poly_size,
            rlwe_size,
            decomp_base_log,
        }
    }
}

impl<Scalar> GgswCiphertext<Vec<Scalar>>
where
    Scalar: UnsignedTorus,
{
    pub fn new_trivial_encryption(
        poly_size: PolynomialSize,
        glwe_size: GlweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        plaintext: &Plaintext<Scalar>,
    ) -> Self {
        let mut ciphertext = Self::allocate(
            Scalar::ZERO,
            poly_size,
            glwe_size,
            decomp_level,
            decomp_base_log,
        );
        ciphertext.fill_with_trivial_encryption(plaintext);
        ciphertext
    }
}

impl<Cont> GgswCiphertext<Cont> {
    /// Creates an Rgsw ciphertext from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::from_container(
    ///     vec![9 as u8; 7 * 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn from_container(
        cont: Cont,
        rlwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => rlwe_size.0, poly_size.0, rlwe_size.0 * rlwe_size.0);
        GgswCiphertext {
            tensor,
            poly_size,
            rlwe_size,
            decomp_base_log,
        }
    }

    /// Returns the size of the glwe ciphertexts composing the ggsw ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.rlwe_size
    }

    /// Returns the number of decomposition levels used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.rlwe_size.0,
            self.poly_size.0,
            self.rlwe_size.0 * self.rlwe_size.0
        );
        DecompositionLevelCount(
            self.as_tensor().len() / (self.rlwe_size.0 * self.rlwe_size.0 * self.poly_size.0),
        )
    }

    /// Returns the size of the polynomials used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a borrowed list composed of all the GLWE ciphertext composing current ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let list = ggsw.as_glwe_list();
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ```
    pub fn as_glwe_list<Scalar>(&self) -> GlweList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        GlweList::from_container(
            self.as_tensor().as_slice(),
            self.rlwe_size.to_glwe_dimension(),
            self.poly_size,
        )
    }

    /// Returns a mutably borrowed `GlweList` composed of all the GLWE ciphertext composing
    /// current ciphertext.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let mut list = ggsw.as_mut_glwe_list();
    /// list.as_mut_tensor().fill_with_element(0);
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ggsw.as_tensor().iter().for_each(|a| assert_eq!(*a, 0));
    /// ```
    pub fn as_mut_glwe_list<Scalar>(&mut self) -> GlweList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let dimension = self.rlwe_size.to_glwe_dimension();
        let size = self.poly_size;
        GlweList::from_container(self.as_mut_tensor().as_mut_slice(), dimension, size)
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns an iterator over borrowed level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// for level_matrix in ggsw.level_matrix_iter() {
    ///     assert_eq!(level_matrix.row_iter().count(), 7);
    ///     assert_eq!(level_matrix.polynomial_size(), PolynomialSize(9));
    ///     for rlwe in level_matrix.row_iter() {
    ///         assert_eq!(rlwe.glwe_size(), GlweSize(7));
    ///         assert_eq!(rlwe.polynomial_size(), PolynomialSize(9));
    ///     }
    /// }
    /// assert_eq!(ggsw.level_matrix_iter().count(), 3);
    /// ```
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GgswLevelMatrix<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let chunks_size = self.poly_size.0 * self.rlwe_size.0 * self.rlwe_size.0;
        let poly_size = self.poly_size;
        let rlwe_size = self.rlwe_size;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    rlwe_size,
                    DecompositionLevel(index + 1),
                )
            })
    }

    /// Returns an iterator over mutably borrowed level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// for mut level_matrix in ggsw.level_matrix_iter_mut() {
    ///     for mut rlwe in level_matrix.row_iter_mut() {
    ///         rlwe.as_mut_tensor().fill_with_element(9);
    ///     }
    /// }
    /// assert!(ggsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = GgswLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0 * self.rlwe_size.0 * self.rlwe_size.0;
        let poly_size = self.poly_size;
        let rlwe_size = self.rlwe_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    rlwe_size,
                    DecompositionLevel(index + 1),
                )
            })
    }

    /// Returns a parallel iterator over mutably borrowed level matrices.
    ///
    /// # Notes
    /// This iterator is hidden behind the "multithread" feature gate.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    ///
    /// let mut ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// ggsw.par_level_matrix_iter_mut()
    ///     .for_each(|mut level_matrix| {
    ///         for mut rlwe in level_matrix.row_iter_mut() {
    ///             rlwe.as_mut_tensor().fill_with_element(9);
    ///         }
    ///     });
    /// assert!(ggsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GgswLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Sync + Send,
    {
        let chunks_size = self.poly_size.0 * self.rlwe_size.0 * self.rlwe_size.0;
        let poly_size = self.poly_size;
        let rlwe_size = self.rlwe_size;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    rlwe_size,
                    DecompositionLevel(index + 1),
                )
            })
    }

    pub fn fill_with_trivial_encryption<Scalar>(&mut self, plaintext: &Plaintext<Scalar>)
    where
        Self: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We fill the ggsw with trivial glwe encryptions of zero:
        for mut glwe in self.as_mut_glwe_list().ciphertext_iter_mut() {
            let mut mask = glwe.get_mut_mask();
            mask.as_mut_tensor().fill_with_element(Scalar::ZERO);
        }
        let base_log = self.decomposition_base_log();
        for mut matrix in self.level_matrix_iter_mut() {
            let decomposition = plaintext.0.wrapping_mul(
                Scalar::ONE
                    << (<Scalar as Numeric>::BITS
                        - (base_log.0 * (matrix.decomposition_level().0))),
            );
            // We iterate over the rows of the level matrix
            for (index, row) in matrix.row_iter_mut().enumerate() {
                let rlwe_ct = row.into_glwe();
                // We retrieve the row as a polynomial list
                let mut polynomial_list = rlwe_ct.into_polynomial_list();
                // We retrieve the polynomial in the diagonal
                let mut level_polynomial = polynomial_list.get_mut_polynomial(index);
                // We get the first coefficient
                let first_coef = level_polynomial.as_mut_tensor().first_mut();
                // We update the first coefficient
                *first_coef = first_coef.wrapping_add(decomposition);
            }
        }
    }

    pub fn external_product<C1, C2, Scalar>(
        &self,
        output: &mut GlweCiphertext<C1>,
        glwe: &GlweCiphertext<C2>,
    ) where
        Self: AsRefTensor<Element = Complex64>,
        GlweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        GlweCiphertext<C2>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We check that the polynomial sizes match
        ck_dim_eq!(
            self.poly_size =>
            glwe.polynomial_size(),
            output.polynomial_size()
        );
        // We check that the glwe sizes match
        ck_dim_eq!(
            self.glwe_size() =>
            glwe.size(),
            output.size()
        );

        let mut buffers = FourierBskBuffers::new(self.polynomial_size(), self.glwe_size());
        let fft_buffers = &mut buffers.fft_buffers;
        let rounded_buffer = &mut buffers.rounded_buffer;

        // "alias" buffers to save some typing
        let fft = &mut fft_buffers.fft;
        let first_fft_buffer = &mut fft_buffers.first_buffer;
        let second_fft_buffer = &mut fft_buffers.second_buffer;
        let output_fft_buffer = &mut fft_buffers.output_buffer;
        output_fft_buffer.fill_with_element(Complex64::new(0., 0.));

        let rounded_input_glwe = rounded_buffer;

        // We round the input mask and body
        let decomposer =
            SignedDecomposer::new(self.decomp_base_log, self.decomposition_level_count());
        decomposer.fill_tensor_with_closest_representable(rounded_input_glwe, glwe);

        // ------------------------------------------------------ EXTERNAL PRODUCT IN FOURIER DOMAIN
        // In this section, we perform the external product in the fourier domain, and accumulate
        // the result in the output_fft_buffer variable.
        let mut decomposition = decomposer.decompose_tensor(rounded_input_glwe);
        // We loop through the levels (we reverse to match the order of the decomposition iterator.)
        for ggsw_decomp_matrix in self.level_matrix_iter().rev() {
            // We retrieve the decomposition of this level.
            let glwe_decomp_term = decomposition.next_term().unwrap();
            debug_assert_eq!(
                ggsw_decomp_matrix.decomposition_level(),
                glwe_decomp_term.level()
            );
            // For each levels we have to add the result of the vector-matrix product between the
            // decomposition of the glwe, and the ggsw level matrix to the output. To do so, we
            // iteratively add to the output, the product between every lines of the matrix, and
            // the corresponding (scalar) polynomial in the glwe decomposition:
            //
            //                ggsw_mat                        ggsw_mat
            //   glwe_dec   | - - - - | <        glwe_dec   | - - - - |
            //  | - - - | x | - - - - |         | - - - | x | - - - - | <
            //    ^         | - - - - |             ^       | - - - - |
            //
            //        t = 1                           t = 2                     ...
            // When possible we iterate two times in a row, to benefit from the fact that fft can
            // transform two polynomials at once.
            let mut iterator = zip!(
                ggsw_decomp_matrix.row_iter(),
                glwe_decomp_term
                    .as_tensor()
                    .subtensor_iter(self.poly_size.0)
                    .map(Polynomial::from_tensor)
            );

            //---------------------------------------------------------------- VECTOR-MATRIX PRODUCT
            loop {
                match (iterator.next(), iterator.next()) {
                    // Two iterates are available, we use the fast fft.
                    (Some(first), Some(second)) => {
                        // We unpack the iterator values
                        let zip_args!(first_ggsw_row, first_glwe_poly) = first;
                        let zip_args!(second_ggsw_row, second_glwe_poly) = second;
                        // We perform the forward fft transform for the glwe polynomials
                        fft.forward_two_as_integer(
                            first_fft_buffer,
                            second_fft_buffer,
                            &first_glwe_poly,
                            &second_glwe_poly,
                        );
                        // Now we loop through the polynomials of the output, and add the
                        // corresponding product of polynomials.
                        let iterator = zip!(
                            first_ggsw_row
                                .as_tensor()
                                .subtensor_iter(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor),
                            second_ggsw_row
                                .as_tensor()
                                .subtensor_iter(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor),
                            output_fft_buffer
                                .as_mut_tensor()
                                .subtensor_iter_mut(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor)
                        );
                        for zip_args!(first_ggsw_poly, second_ggsw_poly, mut output_poly) in
                            iterator
                        {
                            output_poly.update_with_two_multiply_accumulate(
                                &first_ggsw_poly,
                                first_fft_buffer,
                                &second_ggsw_poly,
                                second_fft_buffer,
                            );
                        }
                    }
                    // We reach the  end of the loop and one element remains.
                    (Some(first), None) => {
                        // We unpack the iterator values
                        let (first_ggsw_row, first_glwe_poly) = first;
                        // We perform the forward fft transform for the glwe polynomial
                        fft.forward_as_integer(first_fft_buffer, &first_glwe_poly);
                        // Now we loop through the polynomials of the output, and add the
                        // corresponding product of polynomials.
                        let iterator = zip!(
                            first_ggsw_row
                                .as_tensor()
                                .subtensor_iter(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor),
                            output_fft_buffer
                                .subtensor_iter_mut(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor)
                        );
                        for zip_args!(first_ggsw_poly, mut output_poly) in iterator {
                            output_poly.update_with_multiply_accumulate(
                                &first_ggsw_poly,
                                first_fft_buffer,
                            );
                        }
                    }
                    // The loop is over, we can exit.
                    _ => break,
                }
            }
        }

        // --------------------------------------------  TRANSFORMATION OF RESULT TO STANDARD DOMAIN
        // In this section, we bring the result from the fourier domain, back to the standard
        // domain, and add it to the output.
        //
        // We iterate over the polynomials in the output. Again, when possible, we process two
        // iterations simultaneously to benefit from the fft acceleration.
        let mut _output_bind = output.as_mut_polynomial_list();
        let mut iterator = zip!(
            _output_bind.polynomial_iter_mut(),
            output_fft_buffer
                .subtensor_iter_mut(self.poly_size.0)
                .map(FourierPolynomial::from_tensor)
        );
        loop {
            match (iterator.next(), iterator.next()) {
                (Some(first), Some(second)) => {
                    // We unpack the iterates
                    let zip_args!(mut first_output, mut first_fourier) = first;
                    let zip_args!(mut second_output, mut second_fourier) = second;
                    // We perform the backward transform
                    fft.add_backward_two_as_torus(
                        &mut first_output,
                        &mut second_output,
                        &mut first_fourier,
                        &mut second_fourier,
                    );
                }
                (Some(first), None) => {
                    // We unpack the iterates
                    let (mut first_output, mut first_fourier) = first;
                    // We perform the backward transform
                    fft.add_backward_as_torus(&mut first_output, &mut first_fourier);
                }
                _ => break,
            }
        }
    }
}
