# ############################################################################
# genfri.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# Based on the implementation of Hanjie Pan available at https://github.com/hanjiepan/FRI_pkg.
# ############################################################################
"""
Class for the GenFRI algorithm. Description of the algorithm available at:
[1] Pan, H., Blu, T., & Vetterli, M. (2016). Towards generalized FRI sampling with an application to source resolution
in radioastronomy. IEEE Transactions on Signal Processing, 65(4), 821-835.
"""

import numpy as np
import warnings
from scipy import linalg
from pyoneer.algorithms.base_reconstruction_algorithm import BaseReconstructionAlgorithm
from pyoneer.operators.linear_operator import ToeplitzificationOperator, LinearOperatorFromMatrix


class GenFRIAlgorithm(BaseReconstructionAlgorithm):
    """
    Class for the GenFRI algorithm, with parent class `BaseReconstructionAlgorithm`.
    :attribute linear_op: LinearOperatorFromMatrix
    Forward operator G modelling the measurement process.
    :attribute toeplitz_op: ToeplitzificationOperator
    Toeplitzification operator.
    :attribute tol: float
    Tolerance for stopping criterion. Currently unused.
    :attribute rcond: float
    Cutoff for small singular values in np.linalg.pinv.
    :attribute min_error: float
    Minimal data mismatch so far. Used to select best reconstruction among multiple runs with different random
    initalizations.
    :attribute best_estimate: {None, np.ndarray}
    Estimate with minimal data mismatch so far. Prior to any iteration set to None.

    Description of the algorithm available at:
    [1] Pan, H., Blu, T., & Vetterli, M. (2016). Towards generalized FRI sampling with an application to source resolution
    in radioastronomy. IEEE Transactions on Signal Processing, 65(4), 821-835.
    """

    def __init__(self, linear_op: LinearOperatorFromMatrix, toeplitz_op: ToeplitzificationOperator, nb_iter: int = 50,
                 nb_init: int = 15, tol=1e-6, random_state: int = 1, rcond: float = 1e-5):
        """
        Initialize an object of the class.
        :param linear_op: LinearOperatorFromMatrix
        Forward operator G modelling the measurement process.
        :param toeplitz_op: ToeplitzificationOperator
        Toeplitzification operator.
        :param nb_iter: int
        Number of iterations.
        :param nb_init: int
        Number of random initializations.
        :param tol: float
        Tolerance for stopping criterion. Currently unused.
        :param random_state: int
        Seed for the random generator (for reproducibility of the results).
        :param rcond: float
        Cutoff for small singular values in np.linalg.pinv.
        """
        super(GenFRIAlgorithm, self).__init__(nb_iter=nb_iter, name='GenFRI', nb_init=nb_init,
                                              random_state=random_state)
        if not isinstance(linear_op, LinearOperatorFromMatrix):
            raise ValueError("Argument linear_op must be an instance of LinearOperatorFromMatrix class.")
        self.linear_op = linear_op
        if not isinstance(toeplitz_op, ToeplitzificationOperator):
            raise ValueError("Argument toeplitz_op must be an instance of ToeplitzificationOperator class.")
        self.toeplitz_op = toeplitz_op
        self.tol = tol
        self.rcond = rcond

        # Initialize best estimate and minimal error (used across the multiple init)
        self.min_error = np.infty
        self.best_estimate = None

    def initialize(self, x: np.ndarray) -> dict:
        """
        Initialize various inner variables.
        :param x: np.ndarray
        Generalised measurements used for the recovery. Corresponds to the vector a in [1].
        :return: dict
        Dictionary containing inner variables used at each iteration of the reconstruction.
        """
        out = dict()
        # b0 in [1]
        out[0] = np.zeros(shape=(self.linear_op.shape[1],), dtype=np.complex)
        # G'a in [1]
        out[1] = self.linear_op.rmatvec(x)
        # (G'G)^{-1}G'a in [1]
        out[2] = self.linear_op.pinv(x, rcond=self.rcond)
        # T(\beta) in [1]
        out[3] = self.toeplitz_op.matvec(out[2])
        # right hand side of (4) in [1]
        out[4] = np.concatenate((np.zeros(2 * self.linear_op.shape[1] + 1), [1.]))
        # right hand side of (5) in [1]
        out[5] = np.concatenate((out[1], np.zeros(self.linear_op.shape[1] - self.toeplitz_op.P)))
        # Initialize the annihilating filter coefficient and store initial value for
        # feasible set constraint C
        c = self.rng.standard_normal(self.toeplitz_op.P + 1) + 1j * self.rng.standard_normal(self.toeplitz_op.P + 1)
        out[6] = c
        out[7] = c.copy()  # c0 in [1]
        # Build R(c) in [1]
        out[8] = self.toeplitz_op.rightdual(c)
        # First row of big matrix (4) in [1]
        out[9] = np.concatenate(
            (np.zeros(shape=(self.toeplitz_op.P + 1, self.toeplitz_op.P + 1)), out[3].transpose().conj(),
             np.zeros(shape=(self.toeplitz_op.P + 1, self.linear_op.shape[1])), out[7][:, None]), axis=1)
        # Last row of big matrix (4) in [1]
        out[10] = np.concatenate(
            (out[7].conj()[None, :], np.zeros(shape=(1, 2 * self.linear_op.shape[1] - self.toeplitz_op.P + 1))), axis=1)
        # Data a in [1]
        out[11] = x
        return out

    def iterate(self, x: dict) -> dict:
        """
        Iterations of GenFRI as described in [1].
        :param x: dict
        Inner variables as outputted by the `initialize()` method.
        :return: dict
        Updated inner variables.

        Note: Ill-conditionning warnings returned by np.linalg.solve are filtered so use with caution!
        """
        # Update matrix (4) in [1]
        line2_loop = np.concatenate((x[3], np.zeros(
            shape=(self.linear_op.shape[1] - self.toeplitz_op.P, self.linear_op.shape[1] - self.toeplitz_op.P)),
                                     -x[8], np.zeros(shape=(self.linear_op.shape[1] - self.toeplitz_op.P, 1))), axis=1)
        line3_loop = np.concatenate((np.zeros(shape=(self.linear_op.shape[1], self.toeplitz_op.P + 1)),
                                     -x[8].transpose().conj(), self.linear_op.gram,
                                     np.zeros(shape=(self.linear_op.shape[1], 1))), axis=1)
        mtx_loop = np.concatenate((x[9], line2_loop, line3_loop, x[10]), axis=0)
        # Solve linear system (4) in [1] to update annihilating filter coefficients
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = linalg.solve(a=mtx_loop, b=x[4], check_finite=False, assume_a='her')
        # Update filter coefficients cn and discard auxilary variables
        x[6] = solution[:self.toeplitz_op.P + 1]
        # Update R(cn)
        x[8] = self.toeplitz_op.rightdual(x[6])
        # Build matrix (5) in [1]
        line1_brecon = np.concatenate((self.linear_op.gram, x[8].transpose().conj()), axis=1)
        line2_brecon = np.concatenate((x[8],
                                       np.zeros(shape=(self.linear_op.shape[1] - self.toeplitz_op.P,
                                                       self.linear_op.shape[1] - self.toeplitz_op.P))), axis=1)
        mtx_brecon = np.concatenate((line1_brecon, line2_brecon), axis=0)
        # Solve the system to get the estimate bn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x[0] = linalg.solve(a=mtx_brecon, b=x[5])[:self.linear_op.shape[1]]
        return x

    def stop_criterion(self, x: dict) -> dict:
        """
        Determines when to stop the algorithm based on relative improvement.
        :param x: dict
        Inner variables updated at each iteration by `iterate()`.
        :return: dict
        If key `stop` is True the reconstruction stops at the next iteration.

        Note: In certain cases, the relative improvement achieved by GenFRI from one iteration to the other can be too
        small, leading to a premature stop. For this reason, we have temporarily deactivated this stopping criterion and
        the algorithm is always run for the maximal number of iterations. Another stopping criterion will be investiagted
        in the future.
        """
        stop_dict = dict()
        if np.linalg.norm(self.x_old[0]) == 0:
            relative_improvement = np.infty
        else:
            relative_improvement = np.linalg.norm(x[0] - self.x_old[0]) / np.linalg.norm(self.x_old[0])
        stop_dict['relative_improvement'] = relative_improvement
        stop_dict["stop"] = False  # (relative_improvement < self.tol)
        return stop_dict

    def postprocess(self, x: dict) -> np.ndarray:
        """
        Compute the data mismatch and update the best estimate so far.
        :param x: dict
        Inner variables at the end of the reconstruction process.
        :return: np.ndarray
        Estimate with minimal data mismatch.
        """
        data_mismatch = np.linalg.norm(x[11] - self.linear_op.matvec(x[0]))

        # Update min_error and best solution if required
        if data_mismatch < self.min_error:
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
