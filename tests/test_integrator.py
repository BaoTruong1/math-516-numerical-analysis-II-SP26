# Test file (unchanged except for line length in one test)
import numpy as np
import pytest
from numanalysislib.basis.power import PowerBasis
from src.numanalysislib.calculus.integrator import Quadrature


class TestQuadrature:

    def test_default_initialization(self):
        quad = Quadrature()
        assert quad.rule == 'gauss-legendre'
        assert quad.n_points == 5

    def test_rule_error(self):
        with pytest.raises(ValueError):
            quad = Quadrature(rule='gauss-jacobi')

    def test_lobatto_error(self):
        with pytest.raises(ValueError):
            quad = Quadrature(rule='gauss-lobatto', n_points=1)

    def test_points_error(self):
        with pytest.raises(ValueError):
            quad = Quadrature(rule='gauss-legendre', n_points=-1)

    def test_nodes_weights_legendre_n2(self):
        quad = Quadrature('gauss-legendre', n_points = 2)
        # nodes should be {-1/3, 1/3}
        true_nodes = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        # weights should be {1,1}
        true_weights = np.ones(2)
        np.testing.assert_allclose([quad.points, quad.weights], [true_nodes, true_weights])

    def test_nodes_weights_lobatto_n4(self):
        quad = Quadrature('gauss-lobatto', n_points = 4)
   
        true_nodes = np.array([-1, -1/np.sqrt(5), 1/np.sqrt(5) , 1])

        true_weights = np.array([1/6, 5/6, 5/6, 1/6])
        np.testing.assert_allclose([quad.points, quad.weights], [true_nodes, true_weights])

    def test_gauss_legendre_n1(self):
        #n=1 is just the midpoint rule
        quad = Quadrature('gauss-legendre', n_points=1)
        # Should be exact for degree 1 (2*1-1=1)
        f = lambda x: 2*x + 1
        exact = 2 
        np.testing.assert_allclose(exact, quad.integrate(f, 0, 1))

    def test_gauss_lobatto_n2(self):
        #n=2 is just the trapezoidal rule
        quad = Quadrature('gauss-lobatto', n_points=2)
        # Should be exact for degree 1 (2*2-3=1)
        f = lambda x: 2*x + 1
        np.testing.assert_allclose(2.0, quad.integrate(f, 0, 1))
        
    def test_power_basis_gauss_legendre(self):
        #Test Gauss-Legendre exactness for polynomials up to degree 2n-1
        n_points = 3
        quad = Quadrature(rule='gauss-legendre', n_points=n_points)
        power_basis = PowerBasis(degree=2*n_points - 1)  # degree 5
        coeff = np.ones(power_basis.n_dofs)  # f(x) = 1 + x + x^2 + ... + x^5
        a, b = 0.0, 1.0
    
        exact = sum(1.0/(i+1) for i in range(2*n_points))  # i from 0 to 5
        
        result = quad.integrate_polynomial_object(power_basis, coeff, a, b)
        np.testing.assert_allclose(exact, result, rtol=1e-12)
        
        # Test one degree higher (should not be exact)
        power_basis_higher = PowerBasis(degree=2*n_points)  # degree 6
        coeff_higher = np.ones(power_basis_higher.n_dofs)
        result_higher = quad.integrate_polynomial_object(power_basis_higher, coeff_higher, a, b)
        exact_higher = sum(1.0/(i+1) for i in range(2*n_points + 1))
        
        # Should not be exact (but will be close)
        assert not np.allclose(exact_higher, result_higher, rtol=1e-12)

    def test_power_basis_gauss_lobatto(self):
        #Test Gauss-Lobatto exactness for polynomials up to degree 2n-3
        n_points = 4  # Use 4 points so 2n-3 = 5
        quad = Quadrature(rule='gauss-lobatto', n_points=n_points)
        max_exact_degree = 2*n_points - 3  # degree 5 for n=4
        power_basis = PowerBasis(degree=max_exact_degree)
        coeff = np.ones(power_basis.n_dofs)
        a, b = 0.0, 1.0
        
        exact = sum(1.0/(i+1) for i in range(max_exact_degree + 1))
        
        result = quad.integrate_polynomial_object(power_basis, coeff, a, b)
        np.testing.assert_allclose(exact, result, rtol=1e-12)
        
        # test one degree higher (should not be exact for n=4, degree 6)
        power_basis_higher = PowerBasis(degree=max_exact_degree + 1)
        coeff_higher = np.ones(power_basis_higher.n_dofs)
        result_higher = quad.integrate_polynomial_object(power_basis_higher, coeff_higher, a, b)
        exact_higher = sum(1.0/(i+1) for i in range(max_exact_degree + 2))
        
        # should not be exact
        assert not np.allclose(exact_higher, result_higher, rtol=1e-12)

    def test_compare_convergence_rates(self):
        # test that Gauss-Lobatto is slightly less accurate for same n_points
        f = lambda x: np.exp(x)
        a, b = 0.0, 1.0
        exact = np.exp(1) - 1
        
        for n in [3, 4, 5, 6]:
            quad_gl = Quadrature('gauss-legendre', n)
            quad_lob = Quadrature('gauss-lobatto', n)
            
            error_gl = abs(exact - quad_gl.integrate(f, a, b))
            error_lob = abs(exact - quad_lob.integrate(f, a, b))
            
            # Lobatto should have larger error for non-polynomial functions
            np.testing.assert_array_less(error_gl, error_lob)

    def test_type1_improper_integral(self):
        quad = Quadrature(rule='gauss-lobatto', n_points=10)
        with pytest.raises(ValueError):
            quad.integrate(lambda x: np.exp(-x), 0, np.inf)

    def test_ab_reversed(self):
        quad = Quadrature(rule='gauss-lobatto', n_points=10)
        with pytest.raises(ValueError):
            quad.integrate(lambda x: np.exp(-x), 1, 0)

    def test_runge(self):
        quad = Quadrature('gauss-legendre', n_points=40)

        # Runge function
        f = lambda x: 1 / (1 + 25*x**2)
        result = quad.integrate(f, -1, 1)
        exact = 2/5 * np.arctan(5)
        np.testing.assert_allclose(result, exact, rtol=1e-6)

    def test_improper_integral_divergent(self):
        quad = Quadrature(rule='gauss-legendre', n_points=1000)
        result = quad.integrate(lambda x: 1/x, 0, 1)
        quad_few = Quadrature(rule='gauss-legendre', n_points=100)
        result_few = quad_few.integrate(lambda x: 1/x, 0, 1)

        # With more points, the approximation should get larger
        np.testing.assert_array_less(result_few, result)

    def test_improper_integral_singular_but_finite(self):
        quad = Quadrature(rule='gauss-legendre', n_points=1000)

        # This integral is finite despite singularity at x=0
        result = quad.integrate(lambda x: 1/np.sqrt(x), 0, 1)
        exact = 2.0

        # Should be close but not exact due to singularity
        np.testing.assert_allclose(result, exact, rtol=1e-3)