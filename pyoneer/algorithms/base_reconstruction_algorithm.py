# ############################################################################
# base_reconstruction_algorithm.py
# =======
# Authors : Adrien Besson [adribesson@gmail.com] and Matthieu Simeoni [matthieu.simeoni@gmail.com]
# ############################################################################
"""
Base class for the various reconstruction algorithms.
"""

from abc import abstractmethod
import time
import numpy as np


class BaseReconstructionAlgorithm:
    """
    Base class for the various reconstruction algorithms.
    :attribute nb_iter: int
    Number of iterations of the reconstruction algorithm.
    :attribute nb_init: int
    Number of random initializations.
    :attribute name: str
    Name of the reconstructiona algorithm.
    :attribute random_state: int
    Seed for the random generator (for reproducibility of the results).
    :attribute x_old: dict
    Stores relevant variables from previous iteration for monitoring relative improvement.
    :attribute rng: numpy.random.RandomState
    Seeded random number generator.
    :attribute total_iterations: int
    Total number of iterations so far.
    :attribute total_time: float
    Total execution time so far.
    """
    def __init__(self, nb_iter: int, name: str, nb_init: int = 1, random_state: int = 1):
        """
        Initializes an object of the class.
        :param nb_iter: int
        Number of iterations of the reconstruction algorithm.
        :param name: str
        Name of the reconstructiona algorithm.
        :param nb_init: int
        Number of random initializations.
        :param random_state: int
        Seed for the random generator (for reproducibility of the results).
        """
        self.nb_iter = nb_iter
        self.nb_init = nb_init
        self.name = name
        self.random_state = random_state
        self.x_old = None
        self.rng = np.random.RandomState(seed=self.random_state)

    def reconstruct(self, x: np.ndarray, verbose: bool, verbose_frequency: int = 1) -> np.ndarray:
        """
        Reconstructs the annihilable Fourier series coefficients from generalised measurements.
        :param x: np.ndarray
        Generalised measurements used for the reconstruction.
        :param verbose: bool
        If True logs monitoring the progression of the reconstruction process will be printed in the terminal.
        :param verbose_frequency: int
        How often the logs are printed (unused if `verbose` is set to False).
        :return: np.ndarray
        The reconstructed Fourier series coefficients.
        """
        t_start = time.time()
        elapsed_time = 0.
        self.total_iterations = 0
        for init in range(self.nb_init):
            algo_variables = self.initialize(x)
            for it in range(self.nb_iter):
                self.total_iterations += 1
                self.x_old = algo_variables.copy()
                algo_variables = self.iterate(algo_variables)
                stop_dict = self.stop_criterion(algo_variables)
                if (verbose is True) and (it % verbose_frequency == 0):
                    self.display_log(iteration_init=init + 1, iteration=it + 1, stop_dict=stop_dict,
                                     elapsed_time=elapsed_time)
                if stop_dict['stop']:
                    break
                elapsed_time = time.time() - t_start
            out = self.postprocess(algo_variables)
        self.total_time = time.time() - t_start
        return out

    @abstractmethod
    def initialize(self, x):
        pass

    @abstractmethod
    def display_log(self, **kwargs):
        pass

    @abstractmethod
    def initial_solution(self, x):
        pass

    @abstractmethod
    def stop_criterion(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    @abstractmethod
    def iterate(self, x):
        pass
