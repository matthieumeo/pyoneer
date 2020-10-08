# ############################################################################
# cpgd.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Class for the CPGD algorithm. Description and analysis of the algorithm available at:
[1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
Under review.
"""

import numpy as np
from typing import Optional
from pyoneer.algorithms.base_reconstruction_algorithm import BaseReconstructionAlgorithm
from pyoneer.algorithms.cadzow_denoising import CadzowAlgorithm
from pyoneer.operators.linear_operator import ToeplitzificationOperator, LinearOperatorFromMatrix
from scipy.sparse.linalg import eigs


class CPGDAlgorithm(BaseReconstructionAlgorithm):
    """
    Class for the CPGD algorithm, with parent class `BaseReconstructionAlgorithm`.
    :attribute linear_op: LinearOperatorFromMatrix
    Forward operator G modelling the measurement process.
    :attribute toeplitz_op: ToeplitzificationOperator
    Toeplitzification operator.
    :attribute rank: int
    Rank parameter for Cadzow denoising.
    :attribute rho: float
    Rho parameter for Cadzow denoising.
    :attribute tol: float
    Tolerance for stopping criterion.
    :attribute eig_tol: float
    Tolerance for the convergence of the low rank approximation algorithms. Only used if `cadzow_backend` is 'scipy.sparse'.
    :attribute provided_init_sol: {None,np.ndarray}
    Initial solution for warm start.
    :attribute tau_weight: {None,float}
    Factor for setting the step size tau of the gradient descent.
    :attribute tau: float
    Step size tau of the gradient descent.
    :attribute min_error: float
    Minimal data mismatch so far. Used to select best reconstruction among multiple runs with different random
    initalizations.
    :attribute best_estimate: {None, np.ndarray}
    Estimate with minimal data mismatch so far. Prior to any iteration set to None.
    :attribute denoise_verbose: bool
    Verbosity of Cazow denoising.
    :attribute nb_cadzow_iter: int
    Number of iterations of Cazow denoising.
    :attribute cadzow_backend: {numpy, scipy, scipy.sparse}
    Backend of Cazow denoising.

    denoise_verbose = denoise_verbose
        self.nb_cadzow_iter = nb_cadzow_iter
        self.cadzow_backend
    Description and analysis of the algorithm available at:
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """

    def __init__(self, nb_iter: int, linear_op: LinearOperatorFromMatrix, toeplitz_op: ToeplitzificationOperator,
                 rank: int, nb_cadzow_iter: int = 20, denoise_verbose: bool = False, rho: float = np.Inf,
                 tol: float = 1e-6, eig_tol: float = 1e-8, init_sol: np.ndarray = None, tau: float = None,
                 tau_init_type: str = 'safest', tau_weight: float = 1.5, beta: Optional[float] = None,
                 nb_init: int = 1, random_state: int = 1, cadzow_backend: str = 'scipy'):
        """
        Initialize an object of the class.
        :param nb_iter: int
        Number of iterations.
        :param linear_op: LinearOperatorFromMatrix
        Forward operator G modelling the measurement process.
        :param toeplitz_op: ToeplitzificationOperator
        Toeplitzification operator.
        :param rank: int
        Rank parameter for Cadzow denoising.
        :param nb_cadzow_iter: int
        Number of iterations of Cazow denoising.
        :param denoise_verbose: bool
        Verbosity of Cazow denoising.
        :param rho: float
        Rho parameter for Cadzow denoising.
        :param tol: float
        Tolerance for stopping criterion.
        :param eig_tol: float
        Tolerance for the convergence of the low rank approximation algorithms. Only used if `cadzow_backend` is 'scipy.sparse'.
        :param init_sol: {None, np.ndarray}
        Potential initial solution for warm start.
        :param tau: float
        Step size tau of the gradient descent.
        :param tau_init_type: str
        Method for choosing `tau`.
        :param tau_weight: float
        Weight for tau if `tau_init_type` is not one of {'safest','largest','fastest'}.
        :param nb_init: int
        Number of random initializations.
        :param random_state: int
        Seed the random number generator for reproducibility.
        :param cadzow_backend: {numpy, scipy, scipy.sparse}
        Backend of Cazow denoising.
        """
        super(CPGDAlgorithm, self).__init__(nb_iter=nb_iter, nb_init=nb_init, name='CPGD', random_state=random_state)
        if not isinstance(linear_op, LinearOperatorFromMatrix):
            raise ValueError("Argument linear_op must be an instance of LinearOperatorFromMatrix class.")
        self.linear_op = linear_op
        if not isinstance(toeplitz_op, ToeplitzificationOperator):
            raise ValueError("Argument toeplitz_op must be an instance of ToeplitzificationOperator class.")
        self.toeplitz_op = toeplitz_op
        self.rank = rank
        self.rho = rho
        self.tol = tol
        self.eig_tol = eig_tol
        self.provided_init_sol = init_sol
        self.beta = beta
        if tau is None:
            self.tau_weight = tau_weight
            self.init_tau(type=tau_init_type, weight=self.tau_weight)
        else:
            self.tau_weight = None
            self.tau = tau

        self.min_error = np.infty
        self.best_estimate = None

        # Initialize Cadzow denoising algorithm
        self.denoise_verbose = denoise_verbose
        self.nb_cadzow_iter = nb_cadzow_iter
        self.cadzow_backend = cadzow_backend
        self.preweight = 1 / np.sqrt(toeplitz_op.gram)
        self.postweight = np.sqrt(toeplitz_op.gram)
        self.denoising_algorithm = CadzowAlgorithm(nb_iter=self.nb_cadzow_iter, toeplitz_op=self.toeplitz_op,
                                                   rank=self.rank, rho=self.rho, tol=self.eig_tol,
                                                   backend=self.cadzow_backend)

    def initialize(self, y: np.ndarray) -> list:
        """
        Initialize the estimate and store the data. If `nb_init==1` the estimate is initialized to zero otherwise randomly.
        :param y: np.ndarray
        Generalised measurements for the reconstruction.
        :return: list[np.ndarray, np.ndarray]
        Initial estimate and data.

        Note: In practice we have observed that initializing the estimate to zero yields to higher reconstruction accuracy.
        """
        # Initialize the solution of the algorithm
        if self.provided_init_sol is not None:
            init_sol = self.provided_init_sol.astype(np.complex128)
        else:
            if self.nb_init == 1:
                init_sol = np.zeros(shape=(self.linear_op.shape[1],), dtype=np.complex128)
            else:
                print('CPGD randomly initialized!')
                init_sol = self.rng.standard_normal(self.linear_op.shape[1]) + 1j * self.rng.standard_normal(
                    self.linear_op.shape[1])
        return [init_sol, y]  # y is the data

    def init_tau(self, type: str = 'safest', weight: float = 1.5):
        """
        Set the value of tau, the size of the gradient steps. See Theorems 2 and 4 of [1] for more details.
        :param type: str
        Name of the various strategies for setting tau.
        :param weight:
        Weight for tau if `tau_init_type` is not one of {'safest','largest','fastest'}.

        Note: To ensure convergence, we recommend using type='safest'. {'fastest','largest'} can improve convergence speed
        but can also sometimes make the algorithm diverge.
        """

        P = self.toeplitz_op.P
        weighted_gram = 2 * self.linear_op.gram
        if self.beta is not None:
            beta = self.beta
        else:
            try:
                beta = eigs(weighted_gram, k=1, which='LM', return_eigenvectors=False, tol=self.eig_tol)
                beta *= (1 + self.eig_tol)
            except Exception('Eigs solver did not converge, trying again with small tolerance...'):
                beta = eigs(weighted_gram, k=1, which='LM', return_eigenvectors=False, tol=1e-3)
                beta *= (1 + 1e-3)
        ub = 1 / beta * (1 + 1 / np.sqrt(P + 1))
        lb = 1 / beta * (1 - 1 / np.sqrt(P + 1))
        if type == 'fastest':
            try:
                alpha = eigs(weighted_gram, k=1, which='SM', return_eigenvectors=False, tol=self.eig_tol)
                alpha *= (1 + self.eig_tol)
            except Exception('Eigs solver did not converge. Alpha is set to zero.'):
                alpha = 0
            tau_opt = 2 / (beta + alpha)
            if (tau_opt <= ub) & (tau_opt >= lb):
                self.tau = tau_opt
            else:
                min_lb = np.fmin(np.abs(1 - lb * alpha), np.abs(1 - lb * beta))
                min_ub = np.fmin(np.abs(1 - ub * alpha), np.abs(1 - ub * beta))
                if np.argmin([min_lb, min_ub]) == 0:
                    self.tau = lb
                else:
                    self.tau = ub
        elif type == 'safest':
            self.tau = 1 / beta
        elif type == 'largest':
            self.tau = ub
        else:
            self.tau = weight / beta

    def iterate(self, x: list) -> list:
        """
        Iterations of CPGD.
        :param x: list
        `x[0]` is the estimate so far, `x[1]` the data.
        :return: list
        `x[0]` is the updated estimate, `x[1]` the data.
        """
        derivative = 2 * self.linear_op.rmatvec(self.linear_op.matvec(x[0]) - x[1])
        x[0] = x[0] - self.tau * derivative
        x[0] = self.denoising_algorithm.reconstruct(x[0], verbose=self.denoise_verbose, verbose_frequency=1)
        return x

    def stop_criterion(self, x: list) -> dict:
        """
        Determines when to stop the algorithm based on relative improvement.
        :param x: list
        `x[0]` is the estimate so far, `x[1]` the data.
        :return: dict
        If key `stop` of `stop_dict` is True the reconstruction stops at the next iteration.
        """
        stop_dict = dict()
        if np.linalg.norm(self.x_old[0]) == 0:
            relative_improvement = np.infty
        else:
            relative_improvement = np.linalg.norm((x[0] - self.x_old[0])) / np.linalg.norm(self.x_old[0])
        stop_dict['relative_improvement'] = relative_improvement
        stop_dict["stop"] = (relative_improvement < self.tol)
        return stop_dict

    def postprocess(self, x: list) -> np.ndarray:
        """
        Compute the data mismatch and update the best estimate so far.
        :param x: list
        `x[0]` is the estimate so far, `x[1]` the data.
        :return: np.ndarray
        Estimate with minimal data mismatch.
        """
        data_mismatch = np.linalg.norm(x[1] - self.linear_op.matvec(x[0]))

        # Update min_error and best solution if required
        if data_mismatch < self.min_error:
            if self.best_estimate is not None:
                print('Better solution found!')
            self.best_estimate = x[0]
            self.min_error = data_mismatch
        return self.best_estimate

    def display_log(self, **kwargs):
        """
        Print log to terminal.
        """
        stop_dict = kwargs["stop_dict"]
        if stop_dict['stop'] == 1:
            print(
                "{} algorithm -- Iteration {} over {} -- Elapsed time {} s -- Converged".format(self.name,
                                                                                                kwargs["iteration"],
                                                                                                self.nb_iter,
                                                                                                kwargs["elapsed_time"]))
        else:
            print(
                '{} algorithm -- Iteration {} over {} -- Elapsed time {} s -- Relative improvement {} %'.format(
                    self.name,
                    kwargs["iteration"],
                    self.nb_iter,
                    kwargs["elapsed_time"],
                    100 * stop_dict['relative_improvement']))
