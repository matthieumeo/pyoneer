# ############################################################################
# linear_operator.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Classes and routines for linear operators used in generalised FRI problems.
"""

import numpy as np
import time as t
from abc import abstractmethod
import scipy.sparse.linalg as spsparse
import numpy.linalg as nplin
import scipy.linalg as splin
from scipy.signal import convolve, choose_conv_method
from numbers import Number
from typing import Union


class AbstractLinearOperator(spsparse.LinearOperator):
    """
    Base class for linear operators, inherited from scipy.sparse.linalg.LinearOperator.
    """

    def __init__(self, dtype: type, shape: tuple):
        super(AbstractLinearOperator, self).__init__(shape=shape, dtype=dtype)

    @abstractmethod
    def pinv(self, x: np.ndarray):
        pass

    def proj(self, x: np.ndarray):
        """
        Orthogonal projection onto the range of the linear operator.
        :param x: np.ndarray
        Vector to be projected.
        :return: np.ndarray
        Projected vector.
        """
        return self.matvec(self.pinv(x))

    def proj_conjugate(self, x: np.ndarray, sigma: float):
        if not isinstance(sigma, Number):
            raise ValueError("Parameter sigma must be numeric.")
        return x - sigma * self.proj(x / sigma)


class LinearOperatorFromMatrix(AbstractLinearOperator):
    """
    Class for linear operators defined from matrices.
    :attribute mat: np.ndarray
    Matrix representation of the linear operator.
    :attribute adjoint: np.ndarray
    Conjugate transpose of `mat`.
    :attribute gram: np.ndarray
    Gram matrix  adjoint @ mat
    :attribute norm, lipschitz_cst: float
    Spectral norm of operator.
    """

    def __init__(self, mat: np.ndarray):
        """
        Initiliaze object of class.
        :param mat: np.ndarray[L,N]
        Matrix representation of the linear operator.
        """
        # Check mat
        try:
            mat = np.asarray(mat)
        except ValueError:
            print("Input matrix must be a numpy array.")
        # Init from super class
        super(LinearOperatorFromMatrix, self).__init__(shape=mat.shape, dtype=mat.dtype)

        # Matrix corresponding to the linear operator
        self.mat = mat

        # Adjoint
        self.adjoint = mat.conj().transpose()

        # Corresponding Gram matrix
        self.gram = self.adjoint @ mat

        # Spectral norm, Lipschitz constant
        self.norm = self.lipschitz_cst = np.sqrt(
            spsparse.eigs(self.gram, k=1, which='LM', return_eigenvectors=False, maxiter=int(5e4)))

    def _matvec(self, x: np.ndarray):
        """
        Matrix/vector product.
        :param x: np.ndarray[N,]
        Vector.
        :return: np.ndarray[L,]
        Vector resulting from matrix/vector product.
        """
        M, N = self.shape
        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError('dimension mismatch')
        return self.mat @ x

    def _rmatvec(self, x: np.ndarray):
        """
        Adjoint matrix/vector product.
        :param x: np.ndarray[L,]
        Vector.
        :return: np.ndarray[N,]
        Vector resulting from the adjoint matrix/vector product.
        """
        M, N = self.shape
        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError('dimension mismatch')
        return self.adjoint @ x

    def pinv(self, x: np.ndarray, rcond: float = 1e-9):
        """
        Evaluate the pseudo-inverse of the linear operator for a vector x.
        :param x: np.ndarray[L,]
        Vector.
        :param rcond:
        Cutoff for eigenvalues in `np.linalg.pinv`.
        :return: np.ndarray[N,]
        """
        M, N = self.shape
        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError('dimension mismatch')
        inv_mat = np.linalg.pinv(self.mat, rcond=rcond)
        return inv_mat @ x


class Id(LinearOperatorFromMatrix):
    """
    Class for identity operator inherited from `LinearOperatorFromMatrix`.
    """

    def __init__(self, n: int):
        super(Id, self).__init__(mat=np.eye(n))


class ToeplitzificationOperator(AbstractLinearOperator):
    """
    Class for Toeplitzification operator, inherited from `AbstractLinearOperator`.
    :attribute P: int
    Parameter P in [Section II.A,1].
    :attribute M: int
    Parameter M in [Section II.A,1].
    :attribute N: int
    Parameter N=2*M+1 in [Section II.A,1].
    :attribute norm: float
    Spectral norm of linear operator.
    :attribute gram: np.ndarray
    Diagonal Gram matrix stored as 1D array.

    Reference: Section II.A of
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """

    def __init__(self, P: int, M: int, dtype: type = np.complex128):
        """
        Initiliase Toeplitzification operator with parameter P acting on vectors of size N=2*M+1.
        :param P: int,
        :param M: int.
        :param dtype: type
        Type of the entries of the linear operator.
        """
        # Check P
        try:
            P = int(P)
        except ValueError:
            print("P must be a number.")
        # Check M
        try:
            M = int(M)
        except ValueError:
            print("M must be a number.")
        self.P = P
        self.M = M
        self.N = 2 * M + 1
        self.__offsets = -(np.arange(1, self.N + 1) - 1 - self.P)

        # Init from super class
        shape = ((self.N - self.P) * (self.P + 1), self.N)
        super(ToeplitzificationOperator, self).__init__(shape=shape, dtype=dtype)

        self.norm = np.sqrt(self.P + 1)
        self.gram = toep_gram(self.P, self.N)

    def _matvec(self, x: np.ndarray):
        """
        Compute Tp(x).
        :param x: np.ndarray[N,]
        :return: np.ndarray[N-P,P+1]
        Toeplitz matrix generated by the entries of x.
        """
        if x.size != self.N:
            return print(f'The vector x must have (odd) size {self.N}.')
        else:
            index_i = -self.M + self.P + np.arange(1, self.N - self.P + 1) - 1
            index_j = np.arange(1, self.P + 2) - 1
            index_ij = index_i[:, None] - index_j[None, :]
            Tp_x = x[index_ij + self.M]
            return Tp_x

    def matvec(self, x: np.ndarray):
        """
        Alias of _matvec.
        """
        return self._matvec(x)

    def _rmatvec(self, x: np.ndarray):
        """
        Compute Tp'(x): maps a matrix onto a vector by summing the anti-diagonals.
        :param x: np.ndarray[N-P,P+1]
        Matrix.
        :return: np.ndarray[N,]
        Vector resulting from anti-diagonal summations.
        """
        if x.size != self.shape[0]:
            return print(f'M must have shape {self.shape[0]}x{self.shape[1]}.')
        else:
            out = np.zeros(shape=(self.N,), dtype=x.dtype)
            for (i, m) in enumerate(self.__offsets):
                out[i] = np.sum(np.diagonal(x, offset=m))
            return out

    def rmatvec(self, x: np.ndarray):
        """
        Alias of _rmatvec.
        """
        return self._rmatvec(x)

    def pinv(self, x: np.ndarray):
        """
        Apply pseudo inverse of Tp to input matrix M. We have Tp.pinv(Tp(x))=x.
        :param M: np.ndarray[N-P,P+1]
        :return: np.ndarray[N,]
        """
        return self.rmatvec(x) / self.gram

    def rightdual(self, h: np.ndarray):
        """
        Right dual R of Toeplitzification operator: T(x)h=R(h)x.
        :param h: np.ndarray[P+1,]
        Generator vector.
        :return: np.ndarray[N-P,N]
        Toeplitz matrix.

        Reference: See Definition 1 of
        [2] Pan, H., Blu, T., & Vetterli, M. (2016). Towards generalized FRI sampling with an application to source resolution
        in radioastronomy. IEEE Transactions on Signal Processing, 65(4), 821-835.
        """
        col = np.concatenate(([h[-1]], np.zeros(self.N - self.P - 1)))
        row = np.concatenate((h[::-1], np.zeros(self.N - self.P - 1)))
        return splin.toeplitz(col, row)


class ToeplitzMatrixFree(AbstractLinearOperator):
    """
    Class for matrix-free Toeplitz operators, inherited from `AbstractLinearOperator`.
    :attribute P: int
    Parameter P in [Section II.A,1].
    :attribute M: int
    Parameter M in [Section II.A,1].
    :attribute N: int
    Parameter N=2*M+1 in [Section II.A,1].
    :attribute x: np.ndarray[N,]
    Generator vector.
    :attribute method: str {'auto','direct','fft'}
    Method used by scipy.signal.convolve for performing discrete convolutions.
    :attribute measure: bool
    Whether or not `method` is chosen from precomputed setups or via direct time measures.

    Reference: Supplementary material Appendix A of
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """

    def __init__(self, P: int, M: int, x, measure: bool = False, method: str = 'auto', choose_method: bool = False):
        """
        Initialize object of the class.
        :param P: int
        Parameter P in [Section II.A,1].
        :param M: int
        Parameter M in [Section II.A,1].
        :param x: np.ndarray[2*M+1,]
        Generator vector.
        :param measure: bool
        Whether or not `method` is chosen from precomputed setups or via direct time measures.
        :param method: str {'auto','direct','fft'}
        Method used by scipy.signal.convolve for performing discrete convolutions.
        :param choose_method: bool
        If True choose the optimal convolution method using scipy.signal.choose_conv_method.
        """
        self.P = P
        self.M = M
        self.N = 2 * M + 1
        self.x = x
        self.measure = measure
        super(ToeplitzMatrixFree, self).__init__(shape=(self.N - self.P, self.P + 1), dtype=x.dtype)
        if choose_method is True:
            self.method = self.choose_method()
        else:
            self.method = method

    @property
    def mat(self) -> np.ndarray:
        """
        Return the Toeplitz matrix associated to the Toeplitz operator.
        :return: np.ndarray[N-P,P+1]
        Toeplitz matrix.
        """
        Tp = ToeplitzificationOperator(P=self.P, M=self.M, dtype=self.x.dtype)
        return Tp.matvec(self.x)

    def choose_method(self):
        """
        Choose the optimal convolution method using scipy.signal.choose_conv_method.
        :return: str {'direct','fft'}
        Optimal convolution method.
        """
        h = np.random.rand(self.P + 1) + 1j * np.random.rand(self.P + 1)
        if self.measure is True:
            method, _ = choose_conv_method(self.x, h, mode='valid', measure=self.measure)
        else:
            method = choose_conv_method(self.x, h, mode='valid', measure=self.measure)
        return method

    def matvec(self, h: np.ndarray) -> np.ndarray:
        """
        Alias to _matvec.
        """
        return self._matvec(h)

    def rmatvec(self, u: np.ndarray) -> np.ndarray:
        """
        Alias to _rmatvec.
        """
        return self._matvec(u)

    def _matvec(self, h: np.ndarray) -> np.ndarray:
        """
        Compute Tp(x)h as the valid part of the convolution between x and h (see [Appendix A, 1]).
        :param h: np.ndarray[P+1,]
        :return: np.ndarray[N-P,]
        """
        return convolve(self.x, h, mode='valid', method=self.method)

    def _rmatvec(self, u: np.ndarray) -> np.ndarray:
        """
        Compute Tp(x)'u as the valid part of the cross-correlation between x and h (see [Appendix A, 1]).
        :param h: np.ndarray[N-P,]
        :return: np.ndarray[P+1,]
        """
        return convolve(self.x.conj()[::-1], u, mode='valid', method=self.method)


class FRISampling(LinearOperatorFromMatrix):
    """
    Class for the non-uniform low-pass sampling operator, used in [Section V, 1].
    Inherited from `AbstractLinearOperator`.
    :attribute frequencies: np.ndarray[N,]
    Fourier frequencies.
    :attribute time_samples: np.ndarray[L,]
    Sampling times.
    :attribute period: float
    Period of Dirac stream.

    Reference: Section V of
    [1] Simeoni, M., Besson, A., Hurley, P. & Vetterli, M. (2020). Cadzow Plug-and-Play Gradient Descent for Generalised FRI.
    Under review.
    """

    def __init__(self, frequencies: np.ndarray, time_samples: np.ndarray, period: float):
        """
        Initialize an object from the class.
        :param frequencies: np.ndarray[N,]
        Fourier frequencies.
        :param time_samples: np.ndarray[L,]
        Sampling times.
        :param period: float
        Period of Dirac stream.
        """
        # Check dtypes of the different inputs
        try:
            frequencies = np.asarray(frequencies)
        except ValueError:
            print("Invalid value of frequencies. Must be a np.array.")
        self.frequencies = frequencies

        try:
            time_samples = np.asarray(time_samples)
        except ValueError:
            print("Invalid value of time samples. Must be a np.array.")
        self.time_samples = time_samples

        if not isinstance(period, Number):
            raise ValueError("Invalid value of period. Must be a float.")
        self.period = float(period)

        # Build FRI sampling matrix and corresponding operator
        super(FRISampling, self).__init__(mat=self.build_sampling_matrix())

    def build_sampling_matrix(self):
        """
        Forward operator for traditional FRI setups (ideal low-pass filtering followed by regular or irregular time
        sampling).
        :param frequencies: np.ndarray[2*M+1,]
        :param time_samples: np.ndarray[L,]
        :param period: float,
        :return: np.ndarray[L,2*M+1]
        """
        G = self.period * np.exp(1j * 2 * np.pi * self.frequencies[None, :] * self.time_samples[:, None] / self.period)
        return G


class GaussianRandomSampling(LinearOperatorFromMatrix):
    """
    Class for Gaussian random matrices.
    Inherited from `LinearOperatorFromMatrix`.
    :attribute nrows: int
    Number of rows.
    :attribute ncols: int
    Number of columns.
    :attribute rank: int
    Rank of matrix
    """

    def __init__(self, nrows: int, ncols: int, rank: int):
        """
        Initialize an object from the class.
        :param nrows: int
        Number of rows.
        :param ncols: int
        Number of columns.
        :param rank: int
        Rank of matrix
        """
        try:
            nrows = int(nrows)
        except ValueError:
            print("Invalid value of nrows. Must be an int.")
        self.nrows = nrows
        try:
            ncols = int(ncols)
        except ValueError:
            print("Invalid value of ncols. Must be an int.")
        self.ncols = ncols
        if rank is None:
            rank = np.minimum(nrows, ncols)
        try:
            rank = int(rank)
        except ValueError:
            print("Invalid value of rank. Must be an int.")
        self.rank = rank
        super(GaussianRandomSampling, self).__init__(mat=self.build_sampling_matrix())

    def build_sampling_matrix(self) -> np.ndarray:
        """
        Forms the matrix.
        :return: np.ndarray[nrows, ncols]
        Gaussian random matrix of rank specified by the attribute `rank`.
        """
        mean = 0.0
        stdev = 1 / np.sqrt(self.nrows)
        prng = np.random.RandomState(seed=3)
        G = prng.normal(loc=mean, scale=stdev, size=(self.nrows, self.ncols))

        # Low-rank approximation to reduce the rank
        U, s, Vh = np.linalg.svd(G, full_matrices=False)
        if self.rank < np.minimum(self.nrows, self.ncols):
            s[self.rank:] = 0
        S = np.diag(s)
        return np.matmul(U, np.matmul(S, Vh))


def toep_gram(P: int, N: int) -> np.ndarray:
    """
    Return diagonal entries of Tp'Tp.
    :param P: float,
    :param N: float,
    :return: np.ndarray (N,)
    Diagonal entries of the Gram matrix.
    """
    weights = np.ones(shape=(N,)) * (P + 1)
    weights[:P] = np.arange(1, P + 1)
    weights[N - P:] = np.flip(np.arange(1, P + 1), axis=0)
    return weights


def build_toeplitz_operator(P: int, M: int, x: np.ndarray, toeplitz_class: str = 'standard', method=None) -> Union[
    LinearOperatorFromMatrix, ToeplitzMatrixFree]:
    """
    Build a Toeplitz operator in standard or matrix-free form.
    :param P: int
    :param M: int
    :param x: np.ndarray[2*M+1,]
    Generator.
    :param toeplitz_class: str {'standard','matrix_free'}
    If 'standard' returns object of class `LinearOperatorFromMatrix` otherwise an object of class `ToeplitzMatrixFree`.
    :param method: {None,'auto','direct','fft'}
    Method used by scipy.signal.convolve for performing discrete convolutions. Only used if `toeplitz_class` is
    'matrix-free'.
    :return: {LinearOperatorFromMatrix,ToeplitzMatrixFree}
    Toeplitz operator object of specified class.
    """
    if toeplitz_class in ['standard', 'matrix_free']:
        chosen_toeplitz_class = toeplitz_class
    else:
        chosen_toeplitz_class = None
        print("Method must be one of: ['standard','matrix_free']")

    if chosen_toeplitz_class == 'standard':
        Tp = ToeplitzificationOperator(P=P, M=M, dtype=x.dtype)
        return LinearOperatorFromMatrix(mat=Tp.matvec(x))
    else:
        return ToeplitzMatrixFree(P, M, x, measure=False, method=method)


def choose_toeplitz_class(P: int, M: int, measure: bool = False, avg_size: int = 10):
    """
    Choose optimal Toeplitz class and convolution method by comparing execution times.
    :param P: int
    :param M: int
    :param measure: bool
    If True computations are timed otherwise the setup is matched to the closest existing pre-computed scenario.
    :param avg_size: int
    Number of repeated runs for timing the computation.
    :return: tuple {('standard',None),('matrix_free',str)}
    Optimal Toeplitz class and convolution method (if applicable).
    """
    x = np.random.rand(2 * M + 1) + 1j * np.random.rand(2 * M + 1)
    Tp = ToeplitzificationOperator(P=P, M=M)
    if P == M:
        return 'standard', None
    Tp_x = Tp.matvec(x)
    Tp_x_free = ToeplitzMatrixFree(P, M, x, measure=measure, choose_method=True)
    avg_time_toeplitz_convolution = 0
    avg_time_scipy_convolution = 0
    for i in range(avg_size):
        h = np.random.rand(P + 1) + 1j * np.random.rand(P + 1)
        t1 = t.time()
        y1 = Tp_x @ h
        avg_time_toeplitz_convolution += t.time() - t1
        t2 = t.time()
        y2 = Tp_x_free.matvec(h)
        avg_time_scipy_convolution += t.time() - t2
    avg_time_toeplitz_convolution /= avg_size
    avg_time_scipy_convolution /= avg_size
    if avg_time_toeplitz_convolution <= avg_time_scipy_convolution:
        return 'standard', None
    else:
        return 'matrix_free', Tp_x_free.method


def low_rank_approximation(LinearOperator: Union[ToeplitzMatrixFree, LinearOperatorFromMatrix], rank: int,
                           hermitian: bool = False, tol: float = 0., backend='scipy') -> np.ndarray:
    """
    Perform low-rank approximation of a linear operator.
    :param LinearOperator: {ToeplitzMatrixFree, LinearOperatorFromMatrix}
    Linear operator.
    :param rank: int
    Rank.
    :param hermitian: bool
    If True, `LinearOperator` is assumed Hermitian symmetric.
    :param tol: float
    Tolerance for convergence. Only used if `backend` is `scipy.sparse`.
    :param backend: str {numpy, scipy, scipy.sparse}
    Backend used for computing the low-rank approximation.  If `rank>min(LinearOperator.shape) - 1` only backends
    {numpy, scipy} are supported.
    :return: np.ndarray with shape `LinearOperator.shape`.
    Low-rank approximation of LinearOperator.

    Note: Backend `numpy` supports broadcasting rules while `scipy` and `scipy.sparse` do not.
    Backends `numpy` and `scipy` are wasteful since they do not support matrix-free computations and perform the entire
    eigenvalue/singular value decomposition. They are however very robust and always converge in practice. `scipy.sparse`
    on the other hand is compatible with matrix-free computations and only computes the portion of eigenvalue/singular value
    decomposition necessary for the low-rank projection. This is particularly interesting for very large matrices that
    cannot be stored in memory. It is however more prone to convergence issues.
    """
    if rank >= np.min(LinearOperator.shape):
        return LinearOperator.mat
    elif backend == 'numpy':
        if hermitian is True:
            w, v = nplin.eigh(LinearOperator.mat)
            sort_index = np.flip(np.argsort(np.abs(w)))
            w = w[sort_index]
            v = v[:, sort_index]
            return (v[:, :rank] * w[None, :rank]) @ v[:, :rank].transpose().conj()
        if LinearOperator.shape[0] == LinearOperator.shape[1]:
            w, v = nplin.eig(LinearOperator.mat)
            sort_index = np.flip(np.argsort(np.abs(w)))
            w = w[sort_index]
            v = v[:, sort_index]
            return (v[:, :rank] * w[None, :rank]) @ v[:, :rank].transpose().conj()
        else:
            u, s, vh = nplin.svd(LinearOperator.mat, full_matrices=False)
            return (u[:, :rank] * s[None, :rank]) @ vh[:rank, :]
    elif (rank >= np.min(LinearOperator.shape) - 1) or (backend == 'scipy'):
        if hermitian is True:
            w, v = splin.eigh(LinearOperator.mat, check_finite=False)
            sort_index = np.flip(np.argsort(np.abs(w)))
            w = w[sort_index]
            v = v[:, sort_index]
            return (v[:, :rank] * w[None, :rank]) @ v[:, :rank].transpose().conj()
        elif LinearOperator.shape[0] == LinearOperator.shape[1]:
            w, v = splin.eig(LinearOperator.mat, check_finite=False)
            sort_index = np.flip(np.argsort(np.abs(w)))
            w = w[sort_index]
            v = v[:, sort_index]
            return (v[:, :rank] * w[None, :rank]) @ v[:, :rank].transpose().conj()
        else:
            u, s, vh = splin.svd(LinearOperator.mat, full_matrices=False, check_finite=False)
            return (u[:, :rank] * s[None, :rank]) @ vh[:rank, :]
    elif backend == 'scipy.sparse':
        if hermitian is True:
            w, v = spsparse.eigsh(LinearOperator, k=rank, which='LM', tol=tol)
            return (v * w[None, :]) @ v.transpose().conj()
        elif LinearOperator.shape[0] == LinearOperator.shape[1]:
            w, v = spsparse.eigs(LinearOperator, k=rank, which='LM', tol=tol)
            return (v * w[None, :]) @ v.transpose().conj()
        else:
            u, s, vh = spsparse.svds(LinearOperator, k=rank, tol=tol, which='LM')
            return (u * s[None, :]) @ vh
