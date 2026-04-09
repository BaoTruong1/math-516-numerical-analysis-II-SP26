import numpy as np
import pytest

from numanalysislib.basis.lagrange import LagrangePolynomial


def quadratic(x: np.ndarray) -> np.ndarray:
    return x ** 2 - 3.0 * x + 2.0


def test_fit_returns_nodal_values() -> None:
    basis = LagrangePolynomial(2)
    x_nodes = np.array([-1.0, 0.5, 2.0])
    y_nodes = np.array([3.0, -1.0, 4.0])

    coefficients = basis.fit(x_nodes, y_nodes)

    np.testing.assert_allclose(coefficients, y_nodes, atol=1e-12)


def test_reconstructs_quadratic_exactly() -> None:
    basis = LagrangePolynomial(2)
    x_nodes = np.array([-1.0, 0.5, 2.0])
    y_nodes = quadratic(x_nodes)
    coefficients = basis.fit(x_nodes, y_nodes)

    x = np.linspace(-1.0, 2.0, 100)
    actual = basis.evaluate(coefficients, x)
    expected = quadratic(x)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_basis_is_kronecker_delta_at_nodes() -> None:
    basis = LagrangePolynomial(2)
    x_nodes = np.array([-1.0, 0.5, 2.0])
    basis.fit(x_nodes, np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(
        basis.evaluate_basis(0, x_nodes),
        np.array([1.0, 0.0, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        basis.evaluate_basis(1, x_nodes),
        np.array([0.0, 1.0, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        basis.evaluate_basis(2, x_nodes),
        np.array([0.0, 0.0, 1.0]),
        atol=1e-12,
    )


def test_repeated_nodes_raise_value_error() -> None:
    basis = LagrangePolynomial(2)
    x_nodes = np.array([0.0, 0.0, 1.0])
    y_nodes = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="distinct"):
        basis.fit(x_nodes, y_nodes)


def test_degree_zero_returns_constant() -> None:
    basis = LagrangePolynomial(0)
    coefficients = basis.fit(np.array([2.0]), np.array([5.0]))

    x = np.array([-10.0, 0.0, 7.5])
    actual = basis.evaluate(coefficients, x)
    expected = np.array([5.0, 5.0, 5.0])

    np.testing.assert_allclose(actual, expected, atol=1e-12)



