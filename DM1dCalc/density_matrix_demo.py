#!/usr/bin/env python3
"""
Demonstration script for the DensityMatrixMultislice PyTorch module.

This script shows how to use the optimized PyTorch module for density matrix
multislice calculations and compares its performance with the original function.
"""

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from multislice import (
    DensityMatrixMultislice, 
    create_density_matrix_module, 
    benchmark_density_matrix_methods,
    density_matrix_multislice,
    get_optimal_device
)

def create_test_data(num_slices=50, gridsize=256, complexity='medium'):
    """
    Create test data for demonstration.
    
    Args:
        num_slices: Number of slices in the potential
        gridsize: Size of the grid
        complexity: 'simple', 'medium', or 'complex' test case
        
    Returns:
        potential_1d, transitions, energy, sampling, slice_thickness
    """
    if complexity == 'simple':
        # Simple Gaussian potential
        x = np.linspace(-5, 5, gridsize)
        potential_slice = np.exp(-x**2)
        potential_1d = np.tile(potential_slice, (num_slices, 1))
        transitions = [np.ones(gridsize, dtype=np.complex64)]
        
    elif complexity == 'medium':
        # Multiple Gaussian peaks with random noise
        x = np.linspace(-10, 10, gridsize)
        potential_1d = np.zeros((num_slices, gridsize))
        for i in range(num_slices):
            # Multiple Gaussian peaks
            potential_slice = (np.exp(-(x-2)**2/2) + 
                             0.5*np.exp(-(x+2)**2/1) + 
                             0.3*np.random.random(gridsize))
            potential_1d[i] = potential_slice
        
        # Complex transitions
        transitions = []
        for _ in range(3):
            transition = np.random.random(gridsize) + 1j * np.random.random(gridsize)
            transitions.append(transition.astype(np.complex64))
            
    else:  # complex
        # Realistic crystal-like potential
        x = np.linspace(0, 20, gridsize)  # 20 Angstrom span
        potential_1d = np.zeros((num_slices, gridsize))
        
        for i in range(num_slices):
            # Periodic atomic-like potential
            atomic_positions = [5, 10, 15]  # Angstrom
            potential_slice = np.zeros(gridsize)
            for pos in atomic_positions:
                potential_slice += 10 * np.exp(-((x - pos)**2) / 0.5)
            
            # Add some disorder
            potential_slice += 0.5 * np.random.random(gridsize)
            potential_1d[i] = potential_slice
        
        # Realistic transitions with different energy losses
        transitions = []
        for energy_loss in [10, 20, 30]:  # eV
            # Create transition with phase dependent on energy loss
            phase = 2 * np.pi * energy_loss * x / 1000  # Simplified dispersion
            transition = np.exp(1j * phase) * np.exp(-x/10)  # Damping with distance
            transitions.append(transition.astype(np.complex64))
    
    # Physical parameters
    energy = 200e3  # 200 keV electrons
    sampling = 0.05  # 0.05 Angstrom/pixel
    slice_thickness = 0.5  # 0.5 Angstrom per slice
    
    return potential_1d, transitions, energy, sampling, slice_thickness

def run_performance_comparison():
    """Run a comprehensive performance comparison."""
    print("Performance Comparison: Original vs Optimized Module")
    print("=" * 60)
    
    test_cases = [
        {'name': 'Small Grid', 'num_slices': 20, 'gridsize': 64, 'complexity': 'simple'},
        {'name': 'Medium Grid', 'num_slices': 50, 'gridsize': 128, 'complexity': 'medium'},
        {'name': 'Large Grid', 'num_slices': 100, 'gridsize': 256, 'complexity': 'medium'},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\nTesting {case['name']} ({case['gridsize']}x{case['num_slices']})...")
        
        # Create test data
        potential_1d, transitions, energy, sampling, slice_thickness = create_test_data(
            case['num_slices'], case['gridsize'], case['complexity']
        )
        
        try:
            # Run benchmark
            benchmark_result = benchmark_density_matrix_methods(
                potential_1d, transitions, energy, sampling, slice_thickness, num_runs=3
            )
            
            results.append({**case, **benchmark_result})
            
            print(f"  Original: {benchmark_result['original_mean_time']:.3f}s ± {benchmark_result['original_std_time']:.3f}s")
            print(f"  Optimized: {benchmark_result['module_mean_time']:.3f}s ± {benchmark_result['module_std_time']:.3f}s")
            print(f"  Speedup: {benchmark_result['speedup_factor']:.2f}x")
            print(f"  Max diff: {benchmark_result['max_difference']:.2e}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results

def demonstrate_usage_patterns():
    """Demonstrate different usage patterns of the module."""
    print("\nUsage Pattern Demonstrations")
    print("=" * 40)
    
    # Create test data
    potential_1d, transitions, energy, sampling, slice_thickness = create_test_data(
        num_slices=30, gridsize=128, complexity='medium'
    )
    
    # Pattern 1: Simple usage with factory function
    print("\n1. Simple usage with factory function:")
    dm_module = create_density_matrix_module(energy, sampling, slice_thickness)
    result1 = dm_module.compute_optimized(potential_1d, transitions)
    print(f"   Result shape: {result1.shape}, dtype: {result1.dtype}")
    
    # Pattern 2: Reusing the module for multiple calculations
    print("\n2. Reusing module for multiple calculations:")
    start_time = time.time()
    results = []
    for i in range(5):
        # Modify potential slightly for each calculation
        modified_potential = potential_1d * (1 + 0.1 * i)
        result = dm_module.compute_optimized(modified_potential, transitions, return_numpy=False)
        results.append(result)
    total_time = time.time() - start_time
    print(f"   Computed 5 variations in {total_time:.3f}s")
    print(f"   Average per calculation: {total_time/5:.3f}s")
    
    # Pattern 3: GPU usage (if available)
    if torch.cuda.is_available():
        print("\n3. GPU acceleration:")
        dm_module_gpu = create_density_matrix_module(energy, sampling, slice_thickness, device='cuda')
        start_time = time.time()
        result_gpu = dm_module_gpu.compute_optimized(potential_1d, transitions)
        gpu_time = time.time() - start_time
        print(f"   GPU calculation time: {gpu_time:.3f}s")
        
        # Compare with CPU
        dm_module_cpu = create_density_matrix_module(energy, sampling, slice_thickness, device='cpu')
        start_time = time.time()
        result_cpu = dm_module_cpu.compute_optimized(potential_1d, transitions)
        cpu_time = time.time() - start_time
        print(f"   CPU calculation time: {cpu_time:.3f}s")
        print(f"   GPU speedup: {cpu_time/gpu_time:.2f}x")
        print(f"   Max difference: {np.abs(result_gpu - result_cpu).max():.2e}")
    else:
        print("\n3. GPU not available, skipping GPU demonstration")

def visualize_results():
    """Create visualizations of the density matrix results."""
    print("\nCreating visualization...")
    
    # Create test data
    potential_1d, transitions, energy, sampling, slice_thickness = create_test_data(
        num_slices=20, gridsize=64, complexity='medium'
    )
    
    # Calculate density matrix
    dm_module = create_density_matrix_module(energy, sampling, slice_thickness)
    density_matrix = dm_module.compute_optimized(potential_1d, transitions)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Potential (first and last slice)
    x = np.arange(potential_1d.shape[1]) * sampling
    axes[0, 0].plot(x, potential_1d[0], label='First slice')
    axes[0, 0].plot(x, potential_1d[-1], label='Last slice')
    axes[0, 0].set_xlabel('Position (Å)')
    axes[0, 0].set_ylabel('Potential')
    axes[0, 0].set_title('Potential Profile')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Density matrix magnitude
    im1 = axes[0, 1].imshow(np.abs(density_matrix), cmap='viridis')
    axes[0, 1].set_title('|Density Matrix|')
    axes[0, 1].set_xlabel('Column Index')
    axes[0, 1].set_ylabel('Row Index')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Plot 3: Density matrix phase
    im2 = axes[1, 0].imshow(np.angle(density_matrix), cmap='hsv')
    axes[1, 0].set_title('Density Matrix Phase')
    axes[1, 0].set_xlabel('Column Index')
    axes[1, 0].set_ylabel('Row Index')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot 4: Diagonal of density matrix (population)
    axes[1, 1].plot(x, np.abs(np.diag(density_matrix)))
    axes[1, 1].set_xlabel('Position (Å)')
    axes[1, 1].set_ylabel('Population')
    axes[1, 1].set_title('Diagonal Elements (Population)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('density_matrix_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'density_matrix_results.png'")
    plt.show()

def main():
    """Main demonstration function."""
    print("DensityMatrixMultislice PyTorch Module Demonstration")
    print("=" * 60)
    print(f"Available device: {get_optimal_device()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Run performance comparison
        performance_results = run_performance_comparison()
        
        # Demonstrate usage patterns
        demonstrate_usage_patterns()
        
        # Create visualizations
        try:
            visualize_results()
        except ImportError:
            print("\nMatplotlib not available, skipping visualization")
        except Exception as e:
            print(f"\nVisualization failed: {e}")
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        
        # Summary
        if performance_results:
            avg_speedup = np.mean([r['speedup_factor'] for r in performance_results])
            print(f"Average speedup across test cases: {avg_speedup:.2f}x")
        
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
