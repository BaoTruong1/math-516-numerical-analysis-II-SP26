import numpy as np
import pytest
from numanalysislib.basis.chebyshev import ChebyshevBasis


class TestChebyshevBasis:

    def test_initialization(self):
        """check degree and DOF setup"""
        basis = ChebyshevBasis(degree=3)
        assert basis.degree == 3
        assert basis.n_dofs == 4

    def test_evaluate_basis_scalar(self):
        basis = ChebyshevBasis(degree=3)
        x = np.array([0.5])

        # T_0(x) = cos(0*arccos(x)) = 1
        val0 = basis.evaluate_basis(0, x)
        np.testing.assert_allclose(val0[0], 1.0)

        # T_1(x) = cos(1*arccos(x)) = x
        val1 = basis.evaluate_basis(1, x)
        np.testing.assert_allclose(val1[0], 0.5)

        # T_2(x) = cos(2*arccos(x))
        val2 = basis.evaluate_basis(2, x)
        np.testing.assert_allclose(val2[0], -0.5)

    def test_evaluate_basis_vector(self):
        """check vectorized evaluation"""
        basis = ChebyshevBasis(degree=2)
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        # T_2(x) = cos(2*arccos(x))
        y = basis.evaluate_basis(2, x)
        expected = np.cos(2*np.arccos(x))

        np.testing.assert_allclose(y, expected, atol=1e-14)

    def test_fit_exact_reconstruction(self):
        """
        check if we can recover a known polynomial
        target: 1*T_0 + (-2)*T_1 + 3*T_2 + 1*T_3
        """
        degree = 3
        basis = ChebyshevBasis(degree)

        # true coefficients
        true_coef = np.array([1.0, -2.0, 3.0, 1.0])

        # use Chebyshev nodes for optimal interpolation
        x_nodes = basis.chebyshev_nodes(n=4, kind="roots")
        y_nodes = basis.evaluate(true_coef, x_nodes)

        coef = basis.fit(x_nodes, y_nodes)

        # check coefficient-match
        np.testing.assert_allclose(coef, true_coef, atol=1e-12)

    def test_fit_identity(self):
        """test fitting f(x)=x using Chebyshev"""
        basis = ChebyshevBasis(degree=1)
        x = np.array([-1.0, 1.0])
        y = x

        coef = basis.fit(x, y)
        # expected: x = 0*T_0 + 1*T_1
        np.testing.assert_allclose(coef, [0.0, 1.0], atol=1e-12)

    def test_fit_quadratic(self):
        """test fitting f(x) = x^2 using Chebyshev polynomials."""
        basis = ChebyshevBasis(degree=2)
        x = np.array([-1.0, 0.0, 1.0])
        y = x**2

        coeffs = basis.fit(x, y)
        # Expected: x^2 = 0.5*T_0 + 0*T_1 + 0.5*T_2 -> [0.5, 0, 0.5]
        np.testing.assert_allclose(coeffs, [0.5, 0.0, 0.5], atol=1e-12)

    def test_evaluate_full_polynomial(self):
        """test the evaluate() method"""
        basis = ChebyshevBasis(degree=2)
        # target: 1*T_0 + 2*T_1 + 3*T_2
        coef = np.array([1.0, 2.0, 3.0])

        x = np.array([0.5])
        result = basis.evaluate(coef, x)

        # manual calculation:
        # T_0(0.5) = 1
        # T_1(0.5) = 0.5
        # T_2(0.5) = -0.5
        # target(x) = 1*1 + 2*0.5 + 3*(-0.5) = 0.5
        np.testing.assert_allclose(result[0], 0.5)

    def test_chebyshev_nodes_roots(self):
        """test Chebyshev nodes of the 1st kind (roots)"""
        basis = ChebyshevBasis(degree=3)

        nodes = basis.chebyshev_nodes(n=4, kind="roots")

        # should be 4 nodes in [-1, 1]
        assert len(nodes) == 4
        assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0)

        # known values for n=4: cos(pi/8), cos(3pi/8), cos(5pi/8), cos(7pi/8)
        expected = np.array([-0.92387953, -0.38268343, 0.38268343, 0.92387953])
        np.testing.assert_allclose(nodes, expected, atol=1e-14)

    def test_chebyshev_nodes_extrema(self):
        """test Chebyshev nodes of the second kind (extrema)"""
        basis = ChebyshevBasis(degree=3)

        nodes = basis.chebyshev_nodes(n=4, kind="extrema")

        # should be 4 nodes with endpoints included
        assert len(nodes) == 4
        assert nodes[0] == -1.0
        assert nodes[-1] == 1.0

        # known values for n=4: -1, -0.5, 0.5, 1
        expected = np.array([-1.0, -0.5, 0.5, 1.0])
        np.testing.assert_allclose(nodes, expected, atol=1e-14)

    def test_vandermonde_singularity(self):
        """ensure specific error is raised for duplicate nodes"""
        basis = ChebyshevBasis(degree=1)
        x = np.array([1.0, 1.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="singular matrix"):
            basis.fit(x, y)

    def test_invalid_basis_index(self):
        """error message when basis index is out of range"""
        basis = ChebyshevBasis(degree=2)
        x = np.array([1.0])

        with pytest.raises(ValueError):
            # index > degree, not allowed
            basis.evaluate_basis(3, x)

        with pytest.raises(ValueError):
            # negative index, not allowed
            basis.evaluate_basis(-1, x)

    def test_coefficient_mismatch(self):
        """error message when coefficient array length is wrong"""
        basis = ChebyshevBasis(degree=2)  # expects 3 coefficients
        x = np.array([1.0])

        with pytest.raises(ValueError):
            # only 2 coefficients provided
            basis.evaluate(np.array([1.0, 2.0]), x)
