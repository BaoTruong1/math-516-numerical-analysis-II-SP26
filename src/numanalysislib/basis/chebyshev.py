"""
Chebyshev polynomial basis

Here we implement the Chebyshev polynomials on the reference interval
[-1, 1], where the basis is defined as T_n(x) = cos(n*arccos(x))
"""

import warnings # for showing warning messages
import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis


class ChebyshevBasis(PolynomialBasis):
    """
    Chebyshev polynomials on the reference interval [-1, 1]

    The basis fxns T_n(x) are defined as: T_n(x) = cos(n*arccos(x)), for x in [-1, 1]

    To use on a chosen interval [a, b], need to wrap with AffinePolynomialBasis

    Args:
        degree (int): the maximum degree of the polynomial basis
    """

    def __init__(self, degree: int) -> None:
        """
        Initialize the Chebyshev basis on [-1, 1]

        Args:
            degree (int): maximum polynomial deg
        """
        super().__init__(degree, a=-1.0, b=1.0)
        # for instance, if we pass degree=5 when creating a ChebyshevBasis object,
        # then the PolynomialBasis constructor would set self.degree=5, self.n_dofs=6

    def evaluate_basis(self, ind: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the i-th Chebyshev polynomial T_i(x) at points x

        Args:
            ind (int): note that 0 <= ind <= degree
            x (np.ndarray): pts at which to evaluate the basis

        Returns:
            a np.ndarray, values of T_ind(x) at each point, same shape as x

        Raises ValueError:
            if index is out of range for the basis degree
        """
        if ind < 0 or ind > self.degree:
            raise ValueError(
                f"basis index {ind} is out of range for degree {self.degree}"
            )

        # warning for high-degree evaluation which may be unstable
        if ind > 50:
            warnings.warn(
                f"evaluating high-degree Chebyshev polynomial T_{ind}(x) "
                f"using cos(n*arccos(x)) may be numerically unstable",
                RuntimeWarning
            )

        x = np.asarray(x)
        return np.cos(ind*np.arccos(x))

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Compute coefficients for Chebyshev interpolation by solving the system V*c = y, where V_ij = T_j(x_i)

        Args:
            x_nodes (np.ndarray): interpolation node coordinates w/ shape (n_dofs,)
            y_nodes (np.ndarray): fxn values at the nodes w/ shape (n_dofs,)

        Returns:
            a np.ndarray, coef c s.t. sum(c_i*T_i(x)) interpolates at nodes w/ shape (n_dofs,)

        Raises ValueError:
            if number of nodes does not match n_dofs, or if matrix is singular
        """

        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)

        if len(x_nodes) != self.n_dofs:
            raise ValueError(
                f"expected {self.n_dofs} nodes for degree {self.degree}, "
                f"got {len(x_nodes)}"
            )

        # build the matrix V using vectorized basis evaluation
        # each column j corresponds to T_j evaluated at all x_nodes
        vander = np.column_stack([
            self.evaluate_basis(j, x_nodes) for j in range(self.n_dofs)
        ])

        # check condition number
        cond_num = np.linalg.cond(vander)
        if cond_num > 1e12:
            warnings.warn(
                f"Chebyshev matrix is ill-conditioned "
                f"(w/ cond={cond_num:.2e}), results might be inaccurate",
                RuntimeWarning
            )

        try:
            coef = np.linalg.solve(vander, y_nodes)
        except np.linalg.LinAlgError:
            raise ValueError(
                "we are having singular matrix, make sure that the nodes are distinct"
            )

        return coef

    def evaluate(self, coef: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the series sum(c_i*T_i(x))

        Using vectorized operations: evaluates all basis functions at all
        points simultaneously, then sums with coefficients

        Args:
            coef (np.ndarray): coefficients c_0, c_1, ..., c_{n_dofs-1}
            x (np.ndarray): pts at which to evaluate the polynomial

        Returns:
            a np.ndarray, the evaluated polynomial values, same shape as x

        Raises ValueError:
            if number of coefficients does not match n_dofs
        """
        if len(coef) != self.n_dofs:
            raise ValueError(
                f"expected {self.n_dofs} coefficients, "
                f"got {len(coef)}"
            )

        x = np.asarray(x)

        # build matrix where row i, column j = T_j(x_i)
        # then multiply by coefficients and sum along columns
        basis_mat = np.column_stack([
            self.evaluate_basis(j, x) for j in range(self.n_dofs)
        ])

        # vectorized evaluation: sum over basis fxns
        return basis_mat @ coef

    def chebyshev_nodes(self, n: int, kind: str = "roots") -> np.ndarray:
        """
        Generate Chebyshev nodes on [-1, 1]

        Args:
            n (int): number of nodes to generate
            kind (str): optional
                 type of Chebyshev nodes:
                  - "roots": Chebyshev points of the 1st kind (roots of T_n)
                  - "extrema": Chebyshev points of the 2nd kind
                 default is "roots"

        Returns:
            a np.ndarray, array of n Chebyshev nodes in [-1, 1]

        Raises ValueError:
            if n <= 0 or kind is invalid
        """
        if n <= 0:
            raise ValueError(f"number of nodes n must be positive, got {n}")

        if kind == "roots":
            # roots of T_n: x_k = cos((2k-1)pi/(2n)) for k = 1,...,n
            k = np.arange(1, n + 1)
            nodes = np.cos((2 * k - 1) * np.pi / (2 * n))
        elif kind == "extrema":
            # extrema of T_n: x_k = cos(k*pi/(n-1)) for k = 0,...,n-1
            if n == 1:
                nodes = np.array([0.0])
            else:
                k = np.arange(n)
                nodes = np.cos(k*np.pi/(n-1))
        else:
            raise ValueError(
                f"kind must be 'roots' or 'extrema', got {kind}"
            )
        
        return np.sort(nodes)
