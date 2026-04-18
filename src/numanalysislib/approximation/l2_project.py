import numpy as np
from typing import Callable
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.calculus.integrator import Quadrature


class L2Projector:
    """
    Compute the L2 projection of a function onto a polynomial basis space.
    The projection coefficients c solve:
        M c = b
    where
        M_ij = ∫ phi_i(x) phi_j(x) dx
        b_i  = ∫ f(x) phi_i(x) dx
    """

    def __init__(self, rule: str = "gauss-legendre", n_points: int = 5):
        """
        Parameters
        ----------
        rule : str, optional
            Quadrature rule to use.
        n_points : int, optional
            Number of quadrature points.
        Returns
        -------
        None
        """
        self.integrator = Quadrature(rule = rule, n_points = n_points)

    def mass_matrix(self, basis: PolynomialBasis) -> np.ndarray:
        """
        Assemble the mass matrix M with entries
            M_ij = ∫ phi_i(x) phi_j(x) dx
        Parameters
        ----------
        basis : PolynomialBasis
            Basis object defining the approximation space.
        Returns
        -------
        np.ndarray
            Mass matrix of shape ``(basis.n_dofs, basis.n_dofs)``.
        """
        M = np.zeros((basis.n_dofs, basis.n_dofs), dtype=float)

        for i in range(basis.n_dofs):
            for j in range(i, basis.n_dofs):
                integrand = lambda x, i=i, j=j: (
                    basis.evaluate_basis(i, x) * basis.evaluate_basis(j, x)
                )
                value = self.integrator.integrate(integrand, basis.a, basis.b)
                M[i, j] = value
                M[j, i] = value

        return M

    def load_vector(
        self, basis: PolynomialBasis, f: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Assemble the load vector b with entries
            b_i = ∫ f(x) phi_i(x) dx
        Parameters
        ----------
        basis : PolynomialBasis
            Basis object defining the approximation space.
        f : Callable[[np.ndarray], np.ndarray]
            Function to project onto the basis space.
        Returns
        -------
        np.ndarray
            Load vector of shape ``(basis.n_dofs,)``.
        """
        b = np.zeros(basis.n_dofs, dtype=float)

        for i in range(basis.n_dofs):
            integrand = lambda x, i=i: f(x) * basis.evaluate_basis(i, x)
            b[i] = self.integrator.integrate(integrand, basis.a, basis.b)

        return b

    def project(
        self, basis: PolynomialBasis, f: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Compute the L2 projection coefficients by solving
            M c = b
        Parameters
        ----------
        basis : PolynomialBasis
            Basis object defining the approximation space.
        f : Callable[[np.ndarray], np.ndarray]
            Function to project onto the basis space.
        Returns
        -------
        np.ndarray
            Coefficient vector of shape ``(basis.n_dofs,)`` containing the
            L2 projection of ``f`` onto the span of the basis functions.
        Raises
        ------
        ValueError
            If the linear system for the projection cannot be solved.
        """
        M = self.mass_matrix(basis)
        b = self.load_vector(basis, f)

        try:
            coeffs = np.linalg.solve(M, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Failed to solve the L2 projection system.") from exc

        return coeffs