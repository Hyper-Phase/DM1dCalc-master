import numpy as np
import scipy
from scipy.fft import fft2, ifft2
from DM1dCalc.core_convert import energy2wavelengh, energy2sigma
import torch
import torch.fft
import torch.nn as nn
from tqdm import tqdm

def get_optimal_device():
    """
    Get the optimal device for computation.
    Returns 'cuda' if CUDA is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon GPU
    else:
        return 'cpu'

def propagation(waves, distance, sampling, energy):
    '''
    Propagate the wave function.
    :param waves: The wave function (torch tensor).
    :param distance: The propagation distance, in Angstrom.
    :param sampling: The sampling rate.
    :param energy: The energy of the electron beam, in eV.
    :return: The propagated wave function.
    '''
    # Convert to torch tensor if not already
    if not isinstance(waves, torch.Tensor):
        waves = torch.tensor(waves, dtype=torch.complex64)
    
    # Calculate wavelength
    wavelength = energy2wavelengh(energy)
    
    # Get dimensions and create frequency grids
    m, n = waves.shape
    kx = torch.fft.fftfreq(m, sampling, device=waves.device)
    ky = torch.fft.fftfreq(n, sampling, device=waves.device)
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    k2 = Kx ** 2 + Ky ** 2
    
    # Propagation kernel
    kernel = torch.exp(-1j * k2 * np.pi * wavelength * distance)
    
    # Apply propagation
    waves_fft = torch.fft.fft2(waves)
    waves_propagated = torch.fft.ifft2(waves_fft * kernel)
    
    return waves_propagated

def propagate_1d(waves, distance, sampling, energy):
    """
    Propagate the wave function in one dimension.
    
    :param waves: The wave function (torch tensor).
    :param distance: The propagation distance, in Angstrom.
    :param sampling: The sampling rate.
    :param energy: The energy of the electron beam, in eV.
    :return: The propagated wave function.
    """
    # Convert to torch tensor if not already
    if not isinstance(waves, torch.Tensor):
        waves = torch.tensor(waves, dtype=torch.complex64)
    
    # Calculate wavelength
    wavelength = energy2wavelengh(energy)
    
    # Get dimensions and create frequency grid
    m = waves.shape[0]
    kx = torch.fft.fftfreq(m, sampling, device=waves.device)
    k2 = kx ** 2
    
    # Propagation kernel
    kernel = torch.exp(-1j * k2 * np.pi * wavelength * distance)
    
    # Apply propagation
    waves_fft = torch.fft.fft(waves)
    waves_propagated = torch.fft.ifft(waves_fft * kernel)
    
    return waves_propagated

def propagate_1d_density_matrix(density_matrix, distance, sampling, energy):
    """
    Propagate the density matrix using the multislice method.
    
    :param density_matrix: The density matrix (torch tensor).
    :param distance: The propagation distance, in Angstrom.
    :param sampling: The sampling rate.
    :param energy: The energy of the electron beam, in eV.
    :return: The propagated density matrix.
    """
    # Convert to torch tensor if not already
    if not isinstance(density_matrix, torch.Tensor):
        density_matrix = torch.tensor(density_matrix, dtype=torch.complex64)
    
    # Calculate wavelength
    wavelength = energy2wavelengh(energy)
    # Get dimensions and create frequency grids
    m, n = density_matrix.shape
    k = torch.fft.fftfreq(m, sampling, device=density_matrix.device)
    kp = torch.fft.fftfreq(n, sampling, device=density_matrix.device)
    K, Kp = torch.meshgrid(k, kp, indexing='ij')
    # Propagation kernel
    kernel = torch.exp(-1j * (K - Kp) * np.pi * wavelength * distance)
    
    # Apply propagation
    density_matrix_fft = torch.fft.fft2(density_matrix)
    density_matrix = torch.fft.ifft2(density_matrix_fft * kernel)
    return density_matrix

def multislice(potential, energy, sampling, slice_thickness=None, device='cpu'):
    """
    Calculate the exit wave function using multislice method with PyTorch for acceleration.
    
    :param potential: The potential array (can be numpy array or torch tensor)
    :param energy: The energy of the electron beam, in eV
    :param sampling: The sampling rate
    :param slice_thickness: The thickness of each slice (optional)
    :param device: Device to run computations on ('cpu', 'cuda', etc.)
    :return: The exit wave function as a numpy array
    """
    # Convert to torch tensors and move to device
    if not isinstance(potential, torch.Tensor):
        potential = torch.tensor(potential, dtype=torch.complex64, device=device)
    else:
        potential = potential.to(device=device, dtype=torch.complex64)
    
    wavelength = energy2wavelengh(energy)  # Convert energy to wavelength

    # Initialize exit wave
    ew = torch.ones((potential.shape[1], potential.shape[2]), 
                    dtype=torch.complex64, device=device)
    
    m, n = ew.shape
    kx = torch.fft.fftfreq(m, sampling, device=device)
    ky = torch.fft.fftfreq(n, sampling, device=device)
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    k2 = Kx ** 2 + Ky ** 2
    
    num_slice = potential.shape[0]
    sigma = energy2sigma(energy)  # Convert energy to sigma in m^2
    
    if slice_thickness is None:
        thickness = potential.shape[0] * sampling 
        slice_thickness = thickness/num_slice
    
    # Pre-compute the propagation kernel
    kernel = torch.exp(-1j * k2 * np.pi * wavelength * slice_thickness)
    
    for i in tqdm(range(num_slice)):
        # Phase shift due to potential
        phase_shift = torch.exp(1j * sigma * potential[i, :, :])
        ew = ew * phase_shift
        
        # Propagation step using FFT
        ew_fft = torch.fft.fft2(ew)
        ew = torch.fft.ifft2(ew_fft * kernel)
    
    # Convert back to numpy array - correct method
    ew = ew.detach().cpu().numpy()
    return ew

def multislice_1d(potential_1d, energy, sampling, slice_thickness=None, device='cpu'):
    """
    Calculate the exit wave function for a 1D potential using multislice method.
    
    :param potential_1d: The 1D potential array (numpy array or torch tensor)
    :param energy: The energy of the electron beam, in eV
    :param sampling: The sampling rate
    :param slice_thickness: The thickness of each slice (optional)
    :param device: Device to run computations on ('cpu', 'cuda', etc.)
    :return: The exit wave function as a numpy array
    """
    # Conduct multislice calculation on one dimension
    if not isinstance(potential_1d, torch.Tensor):
        potential_1d = torch.tensor(potential_1d, dtype=torch.complex64, device=device)
    else:
        potential_1d = potential_1d.to(device=device, dtype=torch.complex64)
    wavelength = energy2wavelengh(energy)  # Convert energy to wavelength
    # Initialize exit wave
    ew = torch.ones((potential_1d.shape[1],), dtype=torch.complex64, device=device)
    m = ew.shape[0]
    kx = torch.fft.fftfreq(m, sampling, device=device)
    k2 = kx ** 2
    num_slice = potential_1d.shape[0]
    sigma = energy2sigma(energy)  # Convert energy to sigma in m^2
    if slice_thickness is None:
        thickness = potential_1d.shape[0] * sampling 
        slice_thickness = thickness/num_slice
    # Pre-compute the propagation kernel
    kernel = torch.exp(-1j * k2 * np.pi * wavelength * slice_thickness)
    for i in tqdm(range(num_slice)):
        # Phase shift due to potential
        phase_shift = torch.exp(1j * sigma * potential_1d[i, :])
        ew = ew * phase_shift
        
        # Propagation step using FFT
        ew_fft = torch.fft.fft(ew)
        ew = torch.fft.ifft(ew_fft * kernel)
    # Convert back to numpy array
    ew = ew.detach().cpu().numpy()
    return ew

def density_matrix_multislice(potential_1d, transitions, energy, sampling, slice_thickness=None, device='cpu', transition_mode="potential"):
    """
    Calculate the exit wave function using multislice method with PyTorch for acceleration.
    
    :param potential: The potential array (can be numpy array or torch tensor)
    :param energy: The energy of the electron beam, in eV
    :param sampling: The sampling rate
    :param slice_thickness: The thickness of each slice (optional)
    :param device: Device to run the computation on ('cpu' or 'cuda')
    :return: The exit wave function as a torch tensor
    """
    if not isinstance(potential_1d, torch.Tensor):
        potential_1d = torch.tensor(potential_1d, dtype=torch.float32, device=device)
    if not isinstance(transitions, torch.Tensor):
        transitions = torch.tensor(transitions, dtype=torch.complex64, device=device)
    # Initialize the wave function
    gridsize = potential_1d.shape[1]   
    waves = torch.ones((gridsize,), dtype=torch.complex64, device=device)
    density_matrix = torch.zeros((gridsize, gridsize), dtype=torch.complex64, device=device)
    
    # Propagate through each slice
    if slice_thickness is None:
        slice_thickness = 1.0  # Default thickness if not specified
    
    sigma = energy2sigma(energy)  # Convert energy to sigma in m^2

    for i in range(potential_1d.shape[0]):
        phase_shift = torch.exp(1j * sigma * potential_1d[i, :])
        phase_shift_dm = torch.outer(phase_shift, phase_shift.conj())
        # Apply the phase shift for the elastic waves
        waves = waves * phase_shift
        # Propagate the waves
        waves = propagate_1d(waves, slice_thickness, sampling, energy)
        # Update the density matrix
        for transition in transitions:
            if transition_mode == "potential":     
                waves_transition = transition * waves
                density_matrix = density_matrix + torch.outer(waves_transition, waves_transition.conj())
            else:
                dm_transition = transition * torch.outer(waves, waves.conj())
                density_matrix = density_matrix + dm_transition
        density_matrix = density_matrix * phase_shift_dm
        density_matrix = propagate_1d_density_matrix(density_matrix, slice_thickness, sampling, energy)
    
    return density_matrix

def transitional_multislice(potential, energy, sampling, slice_thickness=None, device='cpu'):
    """
    Calculate the exit wave function using multislice method with PyTorch for acceleration.
    
    :param potential: The potential array (can be numpy array or torch tensor)
    :param energy: The energy of the electron beam, in eV
    :param sampling: The sampling rate
    :param slice_thickness: The thickness of each slice (optional)
    :param device: Device to run the computation on ('cpu' or 'cuda')
    :return: The exit wave function as a torch tensor
    """

    if not isinstance(potential, torch.Tensor):
        potential = torch.tensor(potential, dtype=torch.float32, device=device)
    
    # Initialize the wave function
    waves = torch.ones_like(potential, dtype=torch.complex64, device=device)
    
    # Propagate through each slice
    if slice_thickness is None:
        slice_thickness = 1.0  # Default thickness if not specified
    
    for i in range(potential.shape[0]):
        waves = propagation(waves, slice_thickness, sampling, energy)
        waves *= torch.exp(-1j * potential[i] * slice_thickness / energy2wavelengh(energy))
    
    return waves