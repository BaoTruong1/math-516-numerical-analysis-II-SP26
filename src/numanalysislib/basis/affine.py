from numanalysislib.basis._abstract import PolynomialBasis
import numpy as np

class AffinePolynomialBasis(PolynomialBasis):
    def __init__(self, basis: PolynomialBasis, a: float, b: float):
        """
        Initializes class mapping between [a,b] and [hat_a, hat_b]

        Args:
            basis (PolynomialBasis): the polynomial basis to use
            a (float): left endpoint of physical interval
            b (float): right endpoint of physical interval
        """
        # check for endpoint ordering
        if a > b:
            raise ValueError("b must be greater than a")
        
        # inheret necessary attributes from the basis
        self.basis = basis
        self.a_hat = basis.a
        self.b_hat = basis.b
        self.degree = basis.degree
        self.n_dofs = basis.n_dofs

        # create physical interval
        self.a = a
        self.b = b

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the i-th basis function at points x.

        Args:
            index (int): index of basis function to evaluate
            x (np.ndarray): position to evaluate at

        Returns:
            np.ndarray: basis function evaluated at x
        """
        x_hat = self.pull_back(x)
        x_hat_eval = self.basis.evaluate_basis(index, x_hat)

        return x_hat_eval
    
    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Computes the coefficients c such that p(x_nodes) = y_nodes.
        
        Args:
            x_nodes (np.ndarray): x positions in physical space to fit onto
            y_nodes (np.ndarray): function values to fit to
        
        Returns:
            np.ndarray: coefficients for fit basis
        """
        x_nodes_hat = self.pull_back(x_nodes)
        return self.basis.fit(x_nodes_hat, y_nodes)


    def pull_back(self, x: np.ndarray) -> np.ndarray:
        """
        Maps x in [a, b] to hat_x in [hat_a, hat_b]

        Args:
            x (np.ndarray): element(s) of physical interval to be mapped into reference interval

        Returns:
           (np.ndarray): corresponding element(s) in reference interval
        """
        return self.a_hat + (x - self.a)/(self.b - self.a)*(self.b_hat - self.a_hat)

    def push_forward(self, hat_x: np.ndarray) -> np.ndarray:
        """
        Maps hat_x in [hat_a, hat_b] to x in [a, b].  

        Args:
            hat_x (np.ndarray): element(s) of reference interval to be mapped into physical interval

        Returns:
            np.ndarray: corresponding element(s) in physical interval
        """
        return self.a + (hat_x - self.a_hat)/(self.b_hat - self.a_hat)*(self.b - self.a)
    