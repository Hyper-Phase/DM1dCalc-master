import numpy as np
import torch
import scipy
from DM1dCalc.core_convert import energy2sigma, energy2wavelengh

def dipole_transition_potential_1d(energy, loss, sampling, shape, posn=None, units="pixels"):
    """
    Calculate the dipole transition potential for a given sampling and size.
    """
    x = np.linspace(-shape*sampling / 2, shape*sampling / 2, shape)
    k = np.fft.fftfreq(shape, sampling)
    kgrid = np.fft.fftfreq(shape)
    # Calculate wavelength and wavevector
    k0 = 1 / energy2wavelengh(energy)
    kn = 1 / energy2wavelengh(energy - loss)
    kz = k0 - kn
    # Calculate potential in Fourier space
    Wx = k / (k**2 + kz**2)
    Wz = kz / (k**2 + kz**2)
    # Shift to certain position if provided
    if posn is not None:
        shifted_x = np.zeros_like(Wx, dtype=complex)
        shifted_z = np.zeros_like(Wz, dtype=complex)
        for p in posn:
            if units == "pixels":
                k_shift = kgrid
            else:
                k_shift = k * sampling
            shifted_x = shifted_x + Wx * np.exp(-2j * np.pi * k_shift * p)
            shifted_z = shifted_z + Wz * np.exp(-2j * np.pi * k_shift * p)
        # Calculate the potentials
        pot_x = scipy.fft.ifft(shifted_x)
        pot_z = scipy.fft.ifft(shifted_z)
    else:
        # Calculate the potentials directly
        pot_x = scipy.fft.ifft(Wx)
        pot_z = scipy.fft.ifft(Wz)
    return [pot_x, pot_z]

def dipole_transition_potential(energy, loss, sampling, shape, posn=None, units="pixels"):
    """
    Calculate the dipole transition potential for a given sampling and size.
    """
    m, n = shape
    x = np.linspace(-m*sampling / 2, m*sampling / 2, m)
    y = np.linspace(-n*sampling / 2, n*sampling / 2, n)
    kx = np.fft.fftfreq(m, sampling)
    ky = np.fft.fftfreq(n, sampling)
    kgridx = np.fft.fftfreq(m)
    kgridy = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(kx, ky)
    kgridx, kgridy = np.meshgrid(kgridx, kgridy)
    k2 = kx**2 + ky**2
    # Calculate wavelength and wavevector
    k0 = 1 / energy2wavelengh(energy)
    kn = 1 / energy2wavelengh(energy - loss)
    kz = k0 - kn
    # Calculate potential in Fourier space
    Wx = kx / (k2 + kz**2)
    Wy = ky / (k2 + kz**2)
    Wz = kz / (k2 + kz**2)
    # Shift to certain position if provided
    if posn is not None:
        shifted_x = np.zeros_like(Wx, dtype=complex)
        shifted_y = np.zeros_like(Wy, dtype=complex)
        shifted_z = np.zeros_like(Wz, dtype=complex)
        for p in posn:
            if units == "pixels":
                k_shiftx = kgridx
                k_shifty = kgridy
            else:
                k_shiftx = kx
                k_shifty = ky
            shifted_x = shifted_x + Wx * np.exp(-2j * np.pi * k_shiftx * p[0] 
                                                  - 2j * np.pi * k_shifty * p[1])
            shifted_y = shifted_y + Wy * np.exp(-2j * np.pi * k_shiftx * p[0]
                                                  - 2j * np.pi * k_shifty * p[1])
            shifted_z = shifted_z + Wz * np.exp(-2j * np.pi * k_shiftx * p[0]
                                                  - 2j * np.pi * k_shifty * p[1])
    else:
        shifted_x = Wx
        shifted_y = Wy
        shifted_z = Wz
    # Calculate the potentials
    pot_x = scipy.fft.ifft2(shifted_x)
    pot_y = scipy.fft.ifft2(shifted_y)
    pot_z = scipy.fft.ifft2(shifted_z)
    return [pot_x, pot_y, pot_z]