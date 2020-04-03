# ############################################################################
# cadzow_denoising.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Class for Cadzow denoising.
"""

import numpy as np
from pyoneer.algorithms.base_reconstruction_algorithm import BaseReconstructionAlgorithm
from pyoneer.operators.linear_operator import ToeplitzificationOperator, build_toeplitz_operator, \
    choose_toeplitz_class, low_rank_approximation


class CadzowAlgorithm(BaseReconstructionAlgorithm):
    r"""
    Class for Cadzow denoising, with parent class `BaseReconstructionAlgorithm`.
    :attribute toeplitz_op: ToeplitzificationOperator
    Toeplitzification operator.
    :attribute toeplitz_class: str {'standard', 'matrix_free'}
    How to implement matrix/vector products with toeplitz matrices: 'standard' uses naive matrix/vector products,
    'matrix_free' uses convolutions.
    :attribute method: str {'direct', 'fft'}
    Method used by scipy.signal.convolve for the convolution if `toeplitz_class` is `matrix_free`.
    :attribute rank: int
    Rank of the toeplitz matrix formed by the denoised Fourier series coefficients.
    :attribute rho: float
    Maximal $\Gamma$-norm of the denoised Fourier series coefficients.
    :attribute tol: float
    Tolerance for the convergence of the low rank approximation algorithms. Only used if `backend` is 'scipy.sparse'.
    :attribute backend: str {numpy,scipy,scipy.sparse}
    Which backend to use for the low rank approximation.
    """

    def __init__(self, nb_iter: int, toeplitz_op: ToeplitzificationOperator, rank: int, rho: np.float = np.infty,
                 tol: np.float = 1e-6, backend='scipy'):
        """
        Initializes an object of the class.
        :param nb_iter: int
        Number of iterations.
        :param toeplitz_op: ToeplitzificationOperator
        Toeplitzification operator.
        :param rank: int
        Rank of the toeplitz matrix formed by the denoised Fourier series coefficients.
        :param rho: float
        Maximal $\Gamma$-norm of the denoised Fourier series coefficients.
        :param tol: float
        Tolerance for the convergence of the low rank approximation algorithms. Only used if `backend` is 'scipy.sparse'.
        :param backend: str {numpy,scipy,scipy.sparse}
        Which backend to use for the low rank approximation.
        """
        super(CadzowAlgorithm, self).__init__(nb_iter=nb_iter, name='Cadzow')
        if not isinstance(toeplitz_op, ToeplitzificationOperator):
            raise ValueError("Toeplitz_op must be an instance of ToeplitzificationOperator class.")
        self.toeplitz_op = toeplitz_op
        self.toeplitz_class, self.method = choose_toeplitz_class(P=self.toeplitz_op.P, M=self.toeplitz_op.M,
                                                                 measure=True)
        # Parameters of Cadzow denoising
        self.rank = rank
        self.rho = rho
        self.tol = tol
        self.backend = backend

    def denoise(self, x: np.ndarray, verbose: bool, verbose_frequency: int) -> np.ndarray:
        """
        Run the denoising task.
        :param x: np.ndarray
        Noisy Fourier series coefficients.
        :param verbose: bool
        If True logs monitoring the progression of the reconstruction process will be printed in the terminal.
        :param verbose_frequency: int
        How often the logs are printed (unused if `verbose` is set to False).
        :return: np.ndarray
        The denoised Fourier series coefficients.
        """
        return self.reconstruct(x, verbose=verbose, verbose_frequency=verbose_frequency)

    def display_log(self, **kwargs):
        """
        Print log to terminal.
        """
        print("{} algorithm -- Iteration {} over {} -- Elapsed time {:3.3f} s".format(self.name, kwargs["iteration"],
                                                                                      self.nb_iter,
                                                                                      kwargs["elapsed_time"]))

    def stop_criterion(self, x: np.ndarray) -> dict:
        """
        Determines when to stop the algorithm. For Cadzow denoising the algorithm is always run for the maximal number
        of iterations.
        :param x: np.ndarray
        Iterate.
        :return: dict
        If key `stop` is True the reconstruction stops at the next iteration.
        """
        stop_dict = dict()
        stop_dict['stop'] = False
        return stop_dict

    def iterate(self, x: np.ndarray) -> np.ndarray:
        """
        Iterations of Cadzow denoising.
        :param x: np.ndarray
        Iterate so far.
        :return: np.ndarray
        Updated iterate.

        Note: For efficiency reasons, the low rank approximation is implemented differently depending on the cases.
        """
        x = self.proj_gamma_ball(x)
        conj_sym_coeffs = np.array_equal(np.flip(x), np.conj(x))
        x = build_toeplitz_operator(P=self.toeplitz_op.P, M=self.toeplitz_op.M, x=x, toeplitz_class=self.toeplitz_class,
                                    method=self.method)
        if (x.shape[0] == x.shape[1]) and conj_sym_coeffs:
            print('Cadzow in Hermitian mode.')
            x = low_rank_approximation(x, rank=self.rank, tol=self.tol, hermitian=True, backend=self.backend)
        else:
            x = low_rank_approximation(x, rank=self.rank, tol=self.tol, hermitian=False, backend=self.backend)
        x = self.toeplitz_op.pinv(x)
        return x

    def initialize(self, x: np.ndarray) -> np.ndarray:
        """
        Unused for Cadzow denoising.
        :param x: np.ndarray
        :return: np.ndarray
        """
        return x

    def postprocess(self, x: np.ndarray) -> np.ndarray:
        """
        Unused for Cadzow denoising.
        :param x: np.ndarray
        :return: np.ndarray
        """
        return x

    def proj_gamma_ball(self, x: np.ndarray) -> np.ndarray:
        r"""
        Project `x` on $\Gamma$-ball of radius `rho`.
        :param x: np.ndarray
        Input vector.
        :return: np.ndarray
        Projected vector.
        """
        gamma_norm = np.linalg.norm(np.sqrt(self.toeplitz_op.gram) * x)
        if gamma_norm <= self.rho:
            return x
        else:
            return self.rho * x / gamma_norm
