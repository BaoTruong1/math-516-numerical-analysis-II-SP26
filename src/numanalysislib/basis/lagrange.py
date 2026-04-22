r"""
Goal: Handle the construction and evaluation of Lagrange interpolating
polynomials.
Methods:
- `fit(x_nodes, y_nodes)`: Stores nodes and computes Barycentric
weights.
- `evaluate(coefficients, x)`: Evaluates the full polynomial using the
Barycentric Formula.
- `evaluate_basis(index, x)`: Computes the individual i-th basis
function.
Math: L_i(x) = \\prod_{j \\neq i} \\frac{x - x_j}{x_i - x_j}
Note: This class utilizes Barycentric weights to achieve numerical
stability.
"""

import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis


class LagrangePolynomial(PolynomialBasis):
    """
    Lagrange polynomial basis defined by a set of distinct
    interpolation nodes.

    For nodes x_0, x_1,..., the i-th basis function is
    L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}

    Parameters
    ----------
    degree : int
    Degree of the polynomial basis.
    """

    def __init__(self, degree: int) -> None:
        """
        Initialize the empty Lagrange polynomial basis.

        Parameters
        ----------
        degree : int
            Degree of the polynomial basis.

        Data will be provided later via the fit() method.
        """
        super().__init__(degree, a=0.0, b=1.0)
        self.nodes = None
        self.weights = None

    def _compute_weights(self) -> None:
        """
        Compute the barycentric weights for the stored interpolation
        nodes.

        For nodes :`x_0, x_1,..., x_n`, the barycentric weights are:

            w_i = \frac{1}{\prod_{j \ne i} (x_i - x_j)}.

        """

        diff_matrix = self.nodes[:, np.newaxis] - self.nodes[np.newaxis, :]
        np.fill_diagonal(diff_matrix, 1.0)
        self.weights = 1.0 / np.prod(diff_matrix, axis=1)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the specified Lagrange basis function at given points.

        Parameters
        -------
        index : int
        Index of the basis function to evaluate.
        x : np.ndarray
        Points at which to evaluate the basis function.

        Returns
        -------
        np.ndarray
        Values of the selected basis function at the points `x`
        """
        if self.nodes is None or self.weights is None:
            raise ValueError("Call fit before evaluating the basis.")
        if not 0 <= index < self.n_dofs:
            raise IndexError("Basis index out of range.")


        x = np.asarray(x, dtype=float)
        x_flat = np.atleast_1d(x).reshape(-1)

        diff = x_flat[:, np.newaxis] - self.nodes[np.newaxis, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            terms = self.weights / diff

            denominator = np.sum(terms, axis=1)

            numerator = terms[:, index]
            values = numerator / denominator

        exact_matches = np.where(diff == 0)
        if len(exact_matches[0]) > 0:
            row_indices = exact_matches[0]
            col_indices = exact_matches[1]

            values[row_indices] = (col_indices == index).astype(float)

        return values.reshape(x.shape)

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Lagrange interpolant using the barycentric formula.

        Parameters
        ----------
        coefficients : np.ndarray
        x : np.ndarray
            Points at which to evaluate the interpolating polynomial.

        Returns
        ----------
        np.ndarray
        Values of the interpolating polynomial at the points `x`.
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients.")

        x = np.asarray(x, dtype=float)


        x_flat = np.atleast_1d(x).reshape(-1)
        diff = x_flat[:, np.newaxis] - self.nodes[np.newaxis, :]

        with np.errstate(divide='ignore', invalid='ignore'):
            terms = self.weights / diff

            numerator = np.sum(terms * coefficients, axis=1)

            denominator = np.sum(terms, axis=1)

            result = numerator / denominator

        exact_matches = np.where(diff == 0)
        if len(exact_matches[0]) > 0:
            result[exact_matches[0]] = coefficients[exact_matches[1]]

        return result.reshape(x.shape)

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Store interpolation nodes and return the nodal coefficients.

        Parameters
        ----------
        x_nodes : np.ndarray
            Distinct interpolation nodes.
        y_nodes : np.ndarray
            Function values at the interpolation nodes.

        Returns
        -------
        np.ndarray
            Coefficients of the interpolant in the Lagrange basis.
        """
        x_nodes = np.asarray(x_nodes, dtype=float)
        y_nodes = np.asarray(y_nodes, dtype=float)

        if x_nodes.ndim != 1 or y_nodes.ndim != 1:
            raise ValueError("x_nodes and y_nodes must be one-dimensional.")
        if x_nodes.size == 0:
            raise ValueError("At least one interpolation node is required.")
        if np.unique(x_nodes).size != x_nodes.size:
            raise ValueError("All interpolation nodes must be distinct.")
        if x_nodes.size != self.n_dofs or y_nodes.size != self.n_dofs:
            raise ValueError(
                f"Expected {self.n_dofs} nodes and {self.n_dofs} values."
            )

        self.nodes = x_nodes
        self._compute_weights()

        coefficients = y_nodes
        return coefficients