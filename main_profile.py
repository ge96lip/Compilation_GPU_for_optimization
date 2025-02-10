import numpy as np
import time
import matplotlib.pyplot as plt
import cythonfn

def run_performance_test(grid_sizes, num_iterations=1000):
    """
    Run the Gauss-Seidel solver for different grid sizes and measure execution time.

    Parameters:
    - grid_sizes: List of grid sizes to test.
    - num_iterations: Number of iterations for Gauss-Seidel method.

    Returns:
    - Dictionary mapping grid size to execution time.
    """
    performance = {}

    for N in grid_sizes:
        print(f"Running Gauss-Seidel for grid size {N}x{N}...")
        f = np.random.rand(N, N)  # Initialize grid with random numbers
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = 0, 0, 0, 0  # Set boundary conditions

        start_time = time.time()
        cythonfn.gauss_seidel(f, num_iterations)
        elapsed_time = time.time() - start_time

        performance[N] = elapsed_time
        print(f"Grid size {N}x{N} took {elapsed_time:.4f} seconds")

    return performance


def run_profile():
    # Define grid sizes
    grid_sizes = [10, 20, 50, 100] # could include 200

    # Run performance test
    performance_data = run_performance_test(grid_sizes)

    # Plot performance results
    plt.figure(figsize=(8, 6))
    plt.plot(performance_data.keys(), performance_data.values(), marker='o', linestyle='-')
    plt.xlabel("Grid Size (N x N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Gauss-Seidel Solver Performance")
    plt.grid()
    # plt.show()

if __name__ == "__main__" :
    run_profile()