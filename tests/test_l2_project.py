import numpy as np
from numanalysislib.basis.power import PowerBasis
from numanalysislib.approximation.l2_project import L2Projector


class TestL2Projector:

    def test_initialization(self):
        projector = L2Projector(rule = "gauss-lobatto", n_points = 10)
        assert projector.integrator.rule == "gauss-lobatto"
        assert projector.integrator.n_points == 10

    def test_mass_matrix_shape(self):
        basis = PowerBasis(degree=2)
        projector = L2Projector()
        M = projector.mass_matrix(basis)
        assert M.shape == (3, 3)

    def test_mass_matrix_symmetric(self):
        basis = PowerBasis(degree=2)
        projector = L2Projector()
        M = projector.mass_matrix(basis)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_load_vector_shape(self):
        basis = PowerBasis(degree=2)
        projector = L2Projector()
        f = lambda x: 1 + x
        b = projector.load_vector(basis, f)
        assert b.shape == (3,)

    def test_project_exact_linear_function(self):
        """
        Project f(x) = 1 + 2x onto PowerBasis(1).
        Since f is already in the space span{1, x},
        the coefficients should be [1, 2].
        """
        basis = PowerBasis(degree=1)
        projector = L2Projector()

        f = lambda x: 1 + 2 * x
        coeffs = projector.project(basis, f)

        expected = np.array([1.0, 2.0])
        np.testing.assert_allclose(coeffs, expected, atol=1e-6)

    def test_project_constant_function(self):
        """
        Project f(x) = 3 onto PowerBasis(1).
        The exact coefficients should be [3, 0].
        """
        basis = PowerBasis(degree=1)
        projector = L2Projector()
        f = lambda x: 3.0 * np.ones_like(x)
        coeffs = projector.project(basis, f)
        expected = np.array([3.0, 0.0])
        np.testing.assert_allclose(coeffs, expected, atol=1e-4)