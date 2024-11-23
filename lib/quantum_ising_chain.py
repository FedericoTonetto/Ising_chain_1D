""" 
This module contains the class for the 1D quantum Ising chain.
"""
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
from joblib import Parallel, delayed

class QuantumIsingChain1D:
    """
        This class models a quantum Ising chain with a number conserving Hamiltonian providing methods for computing relevant physical quantities. 

        Attributes:
            N (int): Number of sites.
            J (numpy.ndarray[float, N,N]): Interaction strength between site $i$ and site $j$.
            h (float): Magnetic field strength.
            H (numpy.ndarray[float, N,N]): The Hamiltonian of the system $N \times N$ such that $\hat{H} = C^+HC$.
    """
    def __init__(self, N: int, J: float, h: float, H: np.ndarray):
        """
        Args:
            N (int): Number of sites.
            J (float): Coupling strenght.
            h (float): Magnetic field strength.
            H (numpy.ndarray[float, N,N]): The Hamiltonian of the system $N \times N$ such that $\hat{H} = C^+HC$.
        """
        self.N = N
        self.J = J
        self.h = h
        self.H = H
        assert(H.conj().T == H).all(), "Hamiltonian must be hermitian."

    @property
    def get_eigenvalues(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray[float, N]: The eigenvalues of the Hamiltonian.
        """
        return np.linalg.eigvalsh(self.H)
    @property
    def get_eigenvectors(self) -> np.ndarray:
        """
        Returns:
            numpy.ndarray[float, N,N]: The eigenvectors of the Hamiltonian.
        """
        return np.linalg.eigh(self.H)

    def compute_evolution_operator(self, t: float) -> np.ndarray:
        """
        Args:
            t (float): Time.
        
        Returns:
            numpy.ndarray[float, N,N]: The evolution operator at time t using expm of scipy.
        """
        return expm(-1j * self.H * t)
    
    def compute_time_evolution_covariance_matrix(self, t_init: float, t_final: float, t_step: float, initial_gamma_0: np.ndarray, disable_tqdm: bool = False):
        """
        Compute the time evolution of the covariance matrix numerically using the formula $\Gamma(t) = e^{iHt} \Gamma (0) e^{-iHt}$.

        Args:
            t_init (float): Initial time.
            t_final (float): Final time.
            t_step (float): Time step.
            initial_gamma_0 (numpy.ndarray): Covariance matrix a time t_init.
            disable_tqdm (bool): Disable progress bar.

        Returns:
            gamma_t_series (numpy.ndarray): np.array of $\Gamma(t)$ for each time step (3D array).
            Jt_values (numpy.ndarray): Array of time points.
        """
        Jt_values = self.J * np.arange(t_init, t_final + t_step, t_step)

        # Function to compute evolution for a single time point
        def evolve_single_time(Jt):
            U_t = self.compute_evolution_operator(Jt)
            return U_t.conj().T @ initial_gamma_0 @ U_t

        # Use joblib for parallel computation
        gamma_t_matrices = Parallel(n_jobs=-1)(
            delayed(evolve_single_time)(Jt) for Jt in tqdm(Jt_values, desc="Computing Numerical Evolution", disable=disable_tqdm)
        )

        gamma_t_series = np.array(gamma_t_matrices)

        return gamma_t_series, Jt_values