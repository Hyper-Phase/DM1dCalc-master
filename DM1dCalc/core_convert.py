import numpy as np
from scipy.constants import h, hbar, e, m_e, c
import torch


def energy2wavelengh(energy):
    """
    Convert energy in eV to wavelength in Angstrom.
    """
    energy_J = energy * e
    rest_energy = m_e * c**2
    total_energy = energy_J + rest_energy
    momentum = np.sqrt(total_energy**2 - rest_energy**2) / c
    wavelength = (h / momentum) * 1e10  # Convert to Angstrom
    return wavelength

def energy2sigma(energy):
    """Convert energy in eV to sigma in m^2.
    """
    wavelength = energy2wavelengh(energy)
    me = m_e * (1 + e*energy/(m_e * c**2))
    sigma = 2*np.pi*me*(wavelength/1e20)*e/h**2
    return sigma

def fourier_shift_array_1d(
    y, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """Apply Fourier shift theorem for sub-pixel shift to a 1 dimensional array."""
    if units == "pixels":
        n = 1
    else:
        n = y
    d_ = torch.abs(torch.zeros(1, dtype=dtype)).dtype
    if hasattr(posn, "__len__"):

        return torch.exp(
            -2
            * np.pi
            * 1j
            * n
            * torch.fft.fftfreq(y, dtype=d_).to(device).view(1, y)
            * posn.view(len(posn), 1)
        )
    else:
        return torch.exp(
            -2 * np.pi * 1j * n * torch.fft.fftfreq(y, dtype=d_).to(device) * posn
        )

def fourier_shift_array(
    size, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """
    Create Fourier shift theorem array to (pixel) position given by list posn.

    Parameters
    ----------
    size : array_like
        size of the array (Y,X)
    posn : array_like
        can be a K x 2 array to give a K x Y x X shift arrays
    posn
    """
    # Get number of dimensions
    p_ = torch.as_tensor(posn)
    nn = len(p_.shape)

    # Get size of array
    y, x = size

    if nn == 1:
        # Make y ramp exp(-2pi i ky y)
        yramp = fourier_shift_array_1d(
            y, p_[0].item(), units=units, dtype=dtype, device=device
        )

        # Make y ramp exp(-2pi i kx x)
        xramp = fourier_shift_array_1d(
            x, p_[1].item(), units=units, dtype=dtype, device=device
        )

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return yramp.view(y, 1) * xramp.view(1, x)
    else:
        K = p_.shape[0]
        # Make y ramp exp(-2pi i ky y)
        yramp, xramp = [
            fourier_shift_array_1d(xx, pos, units=units, dtype=dtype, device=device)
            for xx, pos in zip([y, x], p_.T)
        ]

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return yramp.view(K, y, 1) * xramp.view(K, 1, x)

def fourier_shift_torch(
    array,
    posn,
    dtype=torch.float32,
    device=torch.device("cpu"),
    qspace_in=False,
    qspace_out=False,
    units="pixels",
):
    """
    Apply Fourier shift theorem for sub-pixel shifts to array.

    Parameters
    -----------
    array : torch.tensor (...,Y,X,2)
        Complex array to be Fourier shifted
    posn : torch.tensor (K x 2) or (2,)
        Shift(s) to be applied
    """

    if not qspace_in:
        array = torch.fft.fftn(array, dim=(-2, -1))

    array = array * fourier_shift_array(
        array.shape[-2:],
        posn,
        dtype=array.dtype,
        device=array.device,
        units=units,
    )

    if qspace_out:
        return array

    return torch.fft.ifftn(array, dim=[-2, -1])

def fourier_shift_torch_1d(
    array,
    posn,
    dim=-1,
    dtype=torch.float32,
    device=torch.device("cpu"),
    qspace_in=False,
    qspace_out=False,
    units="pixels",
):
    """
    Apply Fourier shift theorem for sub-pixel shifts to array in one dimension.

    Parameters
    -----------
    array : torch.tensor (...,N)
        Complex array to be Fourier shifted along specified dimension
    posn : torch.tensor (K,) or scalar
        Shift(s) to be applied along the specified dimension
    dim : int
        Dimension along which to apply the shift (default: -1, last dimension)
    dtype : torch.dtype
        Data type for computations
    device : torch.device
        Device for computations
    qspace_in : bool
        Whether input array is already in Fourier space
    qspace_out : bool
        Whether to return array in Fourier space
    units : str
        Units for position ("pixels" or actual units)
    """
    
    # Convert posn to tensor if it's a scalar
    posn = torch.as_tensor(posn, device=device)
    
    # Get the size of the dimension we're shifting along
    size = array.shape[dim]
    
    if not qspace_in:
        array = torch.fft.fft(array, dim=dim)
    
    # Create the 1D shift array
    shift_array = fourier_shift_array_1d(
        size, posn, units=units, dtype=array.dtype, device=array.device
    )
    
    # Handle broadcasting for the shift array
    # Create a shape that matches array dimensions with 1s everywhere except the shift dimension
    shift_shape = [1] * array.ndim
    
    # Handle negative dimension indexing
    if dim < 0:
        dim = array.ndim + dim
    
    # Set the appropriate dimension size
    if posn.ndim == 0:  # scalar case
        shift_shape[dim] = size
        shift_array = shift_array.view(shift_shape)
    else:  # vector case (K,)
        # For vector case, we need to add a batch dimension
        shift_shape = [posn.shape[0]] + [1] * array.ndim
        shift_shape[dim + 1] = size  # +1 because we added batch dimension
        shift_array = shift_array.view(shift_shape)
    
    # Apply the shift
    array = array * shift_array
    
    if qspace_out:
        return array
    
    return torch.fft.ifft(array, dim=dim)