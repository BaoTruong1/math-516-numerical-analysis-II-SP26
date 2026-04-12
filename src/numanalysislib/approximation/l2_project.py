import numpy as np
from typing import Callable
from numanalysislib.basis._abstract import PolynomialBasis


class L2Projector:
    """
    Compute the L2 projection of a function onto a polynomial basis space.
    The projection coefficients c solve:
        M c = b
    where
        M_ij = ∫ phi_i(x) phi_j(x) dx
        b_i  = ∫ f(x) phi_i(x) dx
    """

    def __init__(self, n_quad: int = 1000):
        """
        Args:
            n_quad: Number of quadrature points used in the trapezoidal rule.
        """
        self.n_quad = n_quad

    def _integrate(self, f: Callable[[np.ndarray], np.ndarray], a: float, b: float) -> float:
        """
        Numerically integrate f over [a, b] using the composite trapezoidal rule.
        """
        x = np.linspace(a, b, self.n_quad)
        y = f(x)
        return np.trapezoid(y, x)

    def mass_matrix(self, basis: PolynomialBasis) -> np.ndarray:
        """
        Assemble the mass matrix M with entries
            M_ij = ∫ phi_i(x) phi_j(x) dx
        """
        M = np.zeros((basis.n_dofs, basis.n_dofs), dtype=float)

        for i in range(basis.n_dofs):
            for j in range(basis.n_dofs):
                integrand = lambda x, i=i, j=j: (
                    basis.evaluate_basis(i, x) * basis.evaluate_basis(j, x)
                )
                M[i, j] = self._integrate(integrand, basis.a, basis.b)

        return M

    def load_vector(
        self, basis: PolynomialBasis, f: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the load vector b with entries
            b_i = ∫ f(x) phi_i(x) dx
        """
        b = np.zeros(basis.n_dofs, dtype=float)

        for i in range(basis.n_dofs):
            integrand = lambda x, i=i: f(x) * basis.evaluate_basis(i, x)
            b[i] = self._integrate(integrand, basis.a, basis.b)

        return b

    def project(
        self, basis: PolynomialBasis, f: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Compute the L2 projection coefficients by solving
            M c = b
        """
        M = self.mass_matrix(basis)
        b = self.load_vector(basis, f)

        try:
            coeffs = np.linalg.solve(M, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to solve the L2 projection system.") from exc

        return coeffs