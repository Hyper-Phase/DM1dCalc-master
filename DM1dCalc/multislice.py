import numpy as np
import scipy
from scipy.fft import fft2, ifft2
from scipy.constants import h, hbar, e, m_e, c
import torch
import torch.fft
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
    energy_J = energy * e
    rest_energy = m_e * c**2
    total_energy = energy_J + rest_energy
    momentum = np.sqrt(total_energy**2 - rest_energy**2) / c
    wavelength = (h / momentum) * 1e10
    
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
    
    energy_J = energy * e  # Convert eV to Joules
    rest_energy = m_e * c**2  # Rest energy in Joules
    total_energy = energy_J + rest_energy  # Total energy
    # Relativistic momentum: p = sqrt((E_total)^2 - (mc^2)^2) / c
    momentum = np.sqrt(total_energy**2 - rest_energy**2) / c
    wavelength = (h / momentum) * 1e10  # Convert m to Angstroms
    
    # Initialize exit wave
    ew = torch.ones((potential.shape[1], potential.shape[2]), 
                    dtype=torch.complex64, device=device)
    
    m, n = ew.shape
    kx = torch.fft.fftfreq(m, sampling, device=device)
    ky = torch.fft.fftfreq(n, sampling, device=device)
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    k2 = Kx ** 2 + Ky ** 2
    
    num_slice = potential.shape[0]
    me = m_e * (1 + e*energy/(m_e * c**2))
    sigma = 2*np.pi*me*(wavelength/1e20)*e/h**2
    
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
