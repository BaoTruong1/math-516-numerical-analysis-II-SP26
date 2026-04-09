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
    r"""
    Lagrange polynomial basis defined by a set of distinct
    interpolation nodes.

    The $i$-th basis function is defined mathematically as:
    $$L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$
    """

    def __init__(self, degree: int) -> None:
        """
        Initialize the empty Lagrange polynomial basis.

        Data will be provided later via the fit() method.
        """
        super().__init__(degree, a=0.0, b=1.0)
        self.nodes = None
        self.weights = None

    def _compute_weights(self):
        r"""
        Compute the barycentric weights for the stored interpolation
        nodes.

        This method uses a vectorized matrix approach to calculate the
        weights:
        $$w_i = \frac{1}{\prod_{j \neq i} (x_i - x_j)}$$

        It updates the `self.weights` attribute in-place.
        """
        # Creating a matrix for all possible differences (xi - xj).
        diff_matrix = self.nodes[:, np.newaxis] - self.nodes[np.newaxis, :]
        np.fill_diagonal(diff_matrix, 1.0)
        self.weights = 1.0 / np.prod(diff_matrix, axis=1)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the index-th Lagrange basis function at points x.
        """
        if self.nodes is None or self.weights is None:
            raise ValueError("Call fit before evaluating the basis.")
        if not 0 <= index < self.n_dofs:
            raise IndexError("Basis index out of range.")

        # 1. Setup
        x = np.asarray(x, dtype=float)
        x_flat = np.atleast_1d(x).reshape(-1)

        # 2. Distance Matrix
        diff = x_flat[:, np.newaxis] - self.nodes[np.newaxis, :]

        # 3. The Barycentric Math (Ignoring divide-by-zero warnings temporarily)
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = self.weights / diff

            # The denominator is the sum of all terms (same as evaluate)
            denominator = np.sum(terms, axis=1)

            # The numerator is JUST the term for our specific 'index'
            numerator = terms[:, index]
            values = numerator / denominator

        # 4. --- Fix Exact Node Matches (The VIP Rule) ---
        exact_matches = np.where(diff == 0)
        if len(exact_matches[0]) > 0:
            row_indices = exact_matches[0]  # Which x points hit a node
            col_indices = exact_matches[1]  # Which specific node they hit

            # If they hit OUR target node (index), the value is 1.0.
            # If they hit any OTHER node, the value is 0.0.
            values[row_indices] = (col_indices == index).astype(float)

        return values.reshape(x.shape)

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Lagrange interpolant using the Barycentric formula.

        Parameters
        ----------
        coefficients : np.ndarray
            The y-values (coefficients) at each node, shape (n_dofs,).
        x : np.ndarray
            The points where we want to calculate the curve.

        Returns
        -------
        np.ndarray
            The evaluated polynomial p(x).
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(f"Expected {self.n_dofs} coefficients.")

        x = np.asarray(x, dtype=float)

        # This creates the (x - x_j) part of the denominator
        x_flat = np.atleast_1d(x).reshape(-1)
        diff = x_flat[:, np.newaxis] - self.nodes[np.newaxis, :]

        # We temporarily ignore "divide by zero" warnings because we will
        # manually fix exact node matches right after this math.
        with np.errstate(divide='ignore', invalid='ignore'):
            # The lambda_j / (x - x_j) piece used in both top and bottom
            terms = self.weights / diff

            # Numerator: Sum of (terms * y_j)
            numerator = np.sum(terms * coefficients, axis=1)

            # Denominator: Sum of (terms)
            denominator = np.sum(terms, axis=1)

            # The final p(x) formula
            result = numerator / denominator

        # --- Fix Exact Node Matches ---
        # If the user asks to evaluate exactly AT a node (where x - x_j = 0),
        # the formula breaks (1/0).
        exact_matches = np.where(diff == 0)
        if len(exact_matches[0]) > 0:
            result[exact_matches[0]] = coefficients[exact_matches[1]]

        return result

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Store interpolation nodes and return the nodal values as
        coefficients.
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
        return y_nodes.copy()