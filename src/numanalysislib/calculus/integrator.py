import numpy as np
from typing import Callable, Tuple
from scipy.special import roots_jacobi, eval_legendre


class Quadrature:
    '''
    Numerical integration using Gauss-Legendre or Gauss-Lobatto quadrature.

    This class precomputes quadrature points and weights for a specified rule
    and number of points, then provides methods to approximate integrals of
    functions or polynomial objects over finite intervals.

    Parameters
    ----------
    rule : str, optional
        Quadrature rule to use. Must be either ``'gauss-legendre'`` or
        ``'gauss-lobatto'``. Default is ``'gauss-legendre'``.
    n_points : int, optional
        Number of quadrature points. For Gauss-Legendre, must be >= 1.
        For Gauss-Lobatto, must be >= 2. Default is 5.
    '''

    def __init__(self, rule: str = 'gauss-legendre', n_points: int = 5):
        """
        Initialize quadrature rule with specified points.

        Parameters
        ----------
        rule : str, optional
            Quadrature rule: ``'gauss-legendre'`` or ``'gauss-lobatto'``.
        n_points : int, optional
            Number of quadrature points.
        """
        if rule != 'gauss-legendre' and rule != 'gauss-lobatto':
            raise ValueError("Unsupported quadrature rule")

        if rule == 'gauss-lobatto' and n_points < 2:
            raise ValueError("Need at least 2 points for Gauss-Lobatto "
                             "quadrature")
        elif n_points < 1:
            raise ValueError("Need at least 1 point for Gauss-Legendre "
                             "quadrature")

        self.rule = rule
        self.n_points = n_points
        self.points: np.ndarray = None
        self.weights: np.ndarray = None

        self._set_quadrature_rule()

    def _set_quadrature_rule(self) -> None:
        """
        Set reference points and weights on [-1, 1].
        """
        if self.rule == 'gauss-lobatto':
            self.points, self.weights = self._gauss_lobatto_points_and_weights()
        else:
            self.points, self.weights = self._gauss_legendre_points_and_weights()

    def _gauss_legendre_points_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Legendre points and weights on [-1, 1].

        Returns
        -------
        points : np.ndarray
            Quadrature points in the interval [-1, 1].
        weights : np.ndarray
            Corresponding quadrature weights.
        """
        return roots_jacobi(self.n_points, 0.0, 0.0)

    def _gauss_lobatto_points_and_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gauss-Lobatto points and weights on [-1, 1].

        Gauss-Lobatto quadrature includes the endpoints -1 and 1 as quadrature
        points. The interior points are the roots of P'_{n-1}(x), where P_n is
        the Legendre polynomial of degree n.

        This implementation uses SciPy's ``roots_jacobi()`` function to compute
        the roots of the Jacobi polynomial P^{(1,1)}_{n-2}(x), which are
        equivalent to the interior points. Weights are computed using the
        formula:

        .. math::
            w_j = \\frac{2}{n(n+1)} \\cdot \\frac{1}{[L_n(x_j)]^2}

        where L_n is the Legendre polynomial of degree n.

        Returns
        -------
        points : np.ndarray
            Quadrature points in [-1, 1] including endpoints.
        weights : np.ndarray
            Corresponding quadrature weights.
        """
        if self.n_points == 2:
            return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

        degree = self.n_points - 1

        # Get interior points (zeros of L'_degree)
        # These are roots of P_(degree-1)^(1,1)(x)
        interior_points, _ = roots_jacobi(degree - 1, 1.0, 1.0)
        points = np.concatenate([[-1.0], interior_points, [1.0]])

        # Compute weights using formula: 2/(n(n+1)) * 1/[L_n(x_j)]^2
        n = degree
        L_n = eval_legendre(n, points)
        weights = 2.0 / (n * (n + 1)) * (1.0 / (L_n ** 2))

        return points, weights

    def _validate_bounds(self, a: float, b: float) -> None:
        """
        Raises errors if its not a valid bounded interval [a,b].

        Parameters
        ----------
        a : float
            Lower bound of integration.
        b : float
            Upper bound of integration.
        """
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(f"Integration bounds must be finite. Got a={a}, "
                             f"b={b}")
        if a > b:
            raise ValueError(f"Lower bound a={a} must be <= upper bound b={b}")

    def _affine_map(self, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Map quadrature points and weights from [-1, 1] to [a, b].

        Parameters
        ----------
        a : float
            Lower bound of target interval.
        b : float
            Upper bound of target interval.

        Returns
        -------
        x_phys : np.ndarray
            Quadrature points mapped to [a, b].
        weights_phys : np.ndarray
            Quadrature weights scaled for the interval [a, b].
        '''
        x_phys = (b - a) / 2 * self.points + (a + b) / 2
        weights_phys = (b - a) / 2 * self.weights
        return x_phys, weights_phys

    def integrate(self, f: Callable, a: float, b: float) -> float:
        """
        Computes :math:`\\int_a^b f(x) dx` using the precomputed quadrature rule.

        Parameters
        ----------
        f : Callable
            Function to integrate. Must accept a numpy array of points and
            return an array of function values.
        a : float
            Lower bound of integration.
        b : float
            Upper bound of integration.

        Returns
        -------
        float
            Approximate value of the definite integral.
        """
        self._validate_bounds(a, b)
        x_phys, weights_phys = self._affine_map(a, b)
        f_vals = f(x_phys)
        return np.sum(weights_phys * f_vals)

    def integrate_polynomial_object(self, basis, coefficients: np.ndarray,
                                    a: float = None, b: float = None) -> float:
        """
        Convenience method to integrate a polynomial from a basis + coefficients.

        This is just a wrapper around integrate() that creates a callable
        function for you.

        Parameters
        ----------
        basis : object
            Basis object with an ``evaluate(coefficients, x)`` method.
        coefficients : np.ndarray
            Array of shape ``(n_dofs,)`` containing basis coefficients.
        a : float, optional
            Lower bound of integration. If ``None``, uses ``basis.a``.
        b : float, optional
            Upper bound of integration. If ``None``, uses ``basis.b``.

        Returns
        -------
        float
            Approximate value of the definite integral.
        """
        if a is None:
            a = basis.a
        if b is None:
            b = basis.b
        self._validate_bounds(a, b)

        f = lambda x: basis.evaluate(coefficients, x)
        return self.integrate(f, a, b)