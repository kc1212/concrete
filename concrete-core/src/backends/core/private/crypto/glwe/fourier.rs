use concrete_fftw::array::AlignedVec;

use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{GlweSize, PolynomialSize};

// TODO: the FFT buffers are private
use crate::backends::core::private::crypto::bootstrap::fourier::buffers::FftBuffers;
use crate::backends::core::private::crypto::glwe::GlweCiphertext;
use crate::backends::core::private::math::fft::{Complex64, FourierPolynomial};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

/// A GLWE ciphertext in the Fourier Domain.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGlweCiphertext<Cont, Scalar> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    _scalar: std::marker::PhantomData<Scalar>,
}

impl<Scalar> FourierGlweCiphertext<AlignedVec<Complex64>, Scalar> {
    /// Allocates a new GLWE ciphertext in the Fourier domain whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::FourierGlweCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let glwe: FourierGlweCiphertext<_, u32> =
    ///     FourierGlweCiphertext::allocate(Complex64::new(0., 0.), PolynomialSize(10), GlweSize(7));
    /// assert_eq!(glwe.glwe_size(), GlweSize(7));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn allocate(value: Complex64, poly_size: PolynomialSize, glwe_size: GlweSize) -> Self
    where
        Scalar: Copy,
    {
        let mut tensor = Tensor::from_container(AlignedVec::new(glwe_size.0 * poly_size.0));
        tensor.as_mut_tensor().fill_with_element(value);
        FourierGlweCiphertext {
            tensor,
            poly_size,
            glwe_size,
            _scalar: Default::default(),
        }
    }
}

impl<Cont, Scalar: UnsignedInteger + UnsignedTorus> FourierGlweCiphertext<Cont, Scalar> {
    /// Creates a GLWE ciphertext in the Fourier domain from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::FourierGlweCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let glwe: FourierGlweCiphertext<_, u32> = FourierGlweCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    /// );
    /// assert_eq!(glwe.glwe_size(), GlweSize(7));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn from_container(cont: Cont, glwe_size: GlweSize, poly_size: PolynomialSize) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => glwe_size.0, poly_size.0);
        FourierGlweCiphertext {
            tensor,
            poly_size,
            glwe_size,
            _scalar: Default::default(),
        }
    }

    /// Returns the size of the GLWE ciphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::FourierGlweCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let glwe: FourierGlweCiphertext<_, u32> =
    ///     FourierGlweCiphertext::allocate(Complex64::new(0., 0.), PolynomialSize(10), GlweSize(7));
    /// assert_eq!(glwe.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the size of the polynomials used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::FourierGlweCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let glwe: FourierGlweCiphertext<_, u32> =
    ///     FourierGlweCiphertext::allocate(Complex64::new(0., 0.), PolynomialSize(10), GlweSize(7));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Fills a GLWE ciphertext with the Fourier transform of a GLWE ciphertext in
    /// coefficient domain.
    pub fn fill_with_forward_fourier<InputCont>(
        &mut self,
        glwe: &GlweCiphertext<InputCont>,
        buffers: &mut FftBuffers,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        GlweCiphertext<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We retrieve a buffer for the fft.
        let fft_buffer = &mut buffers.first_buffer;
        let fft = &mut buffers.fft;

        // We move every polynomial to the fourier domain.
        let poly_list = glwe.as_polynomial_list();
        let iterator = self
            .tensor
            .subtensor_iter_mut(self.poly_size.0)
            .map(|t| FourierPolynomial::from_container(t.into_container()))
            .zip(poly_list.polynomial_iter());
        for (mut fourier_poly, coef_poly) in iterator {
            fft.forward_as_torus(fft_buffer, &coef_poly);
            fourier_poly
                .as_mut_tensor()
                .fill_with_one((fft_buffer).as_tensor(), |a| *a);
        }
    }

    /// Returns an iterator over references to the polynomials contained in the GLWE.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// for polynomial in list.polynomial_iter() {
    ///     assert_eq!(polynomial.polynomial_size(), PolynomialSize(2));
    /// }
    /// assert_eq!(list.polynomial_iter().count(), 4);
    /// ```
    pub fn polynomial_iter(
        &self,
    ) -> impl Iterator<Item = FourierPolynomial<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.poly_size.0)
            .map(FourierPolynomial::from_tensor)
    }

    /// Returns an iterator over mutable references to the polynomials contained in the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::{
    ///     MonomialDegree, PolynomialList,
    /// };
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// for mut polynomial in list.polynomial_iter_mut() {
    ///     polynomial
    ///         .get_mut_monomial(MonomialDegree(0))
    ///         .set_coefficient(10u8);
    ///     assert_eq!(polynomial.polynomial_size(), PolynomialSize(2));
    /// }
    /// for polynomial in list.polynomial_iter() {
    ///     assert_eq!(
    ///         *polynomial.get_monomial(MonomialDegree(0)).get_coefficient(),
    ///         10u8
    ///     );
    /// }
    /// assert_eq!(list.polynomial_iter_mut().count(), 4);
    /// ```
    pub fn polynomial_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = FourierPolynomial<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(FourierPolynomial::from_tensor)
    }

    /// Returns the tensor product of two GLWE ciphertexts
    ///
    /// # Example
    pub fn tensor_product<Container>(
        &self,
        glwe: &FourierGlweCiphertext<Container, Scalar>,
    ) -> FourierGlweCiphertext<AlignedVec<Complex64>, Scalar>
    where
        Self: AsRefTensor<Element = Complex64>,
        FourierGlweCiphertext<Container, Scalar>: AsRefTensor<Element = Complex64>,
    {
        // We check that the polynomial sizes match
        ck_dim_eq!(
            self.poly_size =>
            glwe.polynomial_size(),
            self.polynomial_size()
        );
        // We check that the glwe sizes match
        ck_dim_eq!(
            self.glwe_size() =>
            glwe.glwe_size(),
            self.glwe_size()
        );

        // we create an output Fourier GLWE ciphertext
        //let mut output = FourierGlweCiphertext::from_container(
        //    vec![Scalar::ZERO; (1 / 2) * self.poly_size.0 * (3 + self.poly_size.0)],
        //    GlweSize((1 / 2) * (3 + self.poly_size.0)),
        //    self.poly_size,
        //);
        let mut output = FourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            self.poly_size,
            GlweSize((1/2) * self.poly_size.0 * (3 + self.poly_size.0)),
        );

        let iter_glwe_1 = self.polynomial_iter();
        {
            let mut iter_output = output.polynomial_iter_mut();

            // Here the output contains all the multiplied terms
            // in the order defined by the loops
            let mut counter_1 = 0;
            let k = self.glwe_size().0;

            // 1. Get the T_i = A1i * A2i terms
            for (i, polynomial1) in iter_glwe_1.enumerate() {
                // create an iterator iter_glwe_2 (mutable)
                let iter_glwe_2 = glwe.polynomial_iter();
                // consumes the iterator object with enumerate()
                for (j, polynomial2) in iter_glwe_2.enumerate() {
                    let mut output_poly = iter_output.next().unwrap();
                    let mut iter_glwe_1_ = self.polynomial_iter();
                    let mut iter_glwe_2_ = glwe.polynomial_iter();
                    if i == j {
                        // Put A1i * A2i into the output
                        output_poly.update_with_multiply_accumulate(&polynomial1, &polynomial2);
                        // Put A1i * B2 + B1 * A2i into the output
                        // create new iterators for glwe_1 and glwe_2
                        output_poly.update_with_two_multiply_accumulate(
                            &polynomial1,
                            // TODO: make sure that this index is correct, should be [-1]
                            &iter_glwe_2_
                                .nth(polynomial1.polynomial_size().0 - 1)
                                .unwrap(),
                            &iter_glwe_1_
                                .nth(polynomial1.polynomial_size().0 - 1)
                                .unwrap(),
                            &polynomial2,
                        );
                        counter_1 += 1;
                    } else {
                        // else condition means i != j
                        if j < i {
                            // Put A1i * A2j + A1j * A2i
                            output_poly.update_with_two_multiply_accumulate(
                                &polynomial1,
                                &polynomial2,
                                &iter_glwe_1_.nth(j).unwrap(),
                                &iter_glwe_2_.nth(i).unwrap(),
                            )
                        }
                    }
                }
                if counter_1 > k {
                    break;
                }
            }
        }
        output
    }
}

impl<Element, Cont, Scalar> AsRefTensor for FourierGlweCiphertext<Cont, Scalar>
where
    Cont: AsRefSlice<Element = Element>,
    Scalar: UnsignedTorus,
{
    type Element = Element;
    type Container = Cont;
    fn as_tensor(&self) -> &Tensor<Self::Container> {
        &self.tensor
    }
}

impl<Element, Cont, Scalar> AsMutTensor for FourierGlweCiphertext<Cont, Scalar>
where
    Cont: AsMutSlice<Element = Element>,
    Scalar: UnsignedTorus,
{
    type Element = Element;
    type Container = Cont;
    fn as_mut_tensor(&mut self) -> &mut Tensor<<Self as AsMutTensor>::Container> {
        &mut self.tensor
    }
}

impl<Cont, Scalar> IntoTensor for FourierGlweCiphertext<Cont, Scalar>
where
    Cont: AsRefSlice,
    Scalar: UnsignedTorus,
{
    type Element = <Cont as AsRefSlice>::Element;
    type Container = Cont;
    fn into_tensor(self) -> Tensor<Self::Container> {
        self.tensor
    }
}
