import numpy as np
import time
import matplotlib.pyplot as plt
@profile 
def gauss_seidel(f, num_iterations=1000):
    """
    Perform Gauss-Seidel iterations to solve the 2D Poisson equation.

    Parameters:
    - f: 2D NumPy array representing the grid.
    - num_iterations: Number of iterations to perform.

    Returns:
    - Updated 2D array after Gauss-Seidel iterations.
    """
    newf = f.copy()
    n, m = newf.shape

    for _ in range(num_iterations):
        for i in range(1, n-1):
            for j in range(1, m-1):
                newf[i, j] = 0.25 * (newf[i, j+1] + newf[i, j-1] +
                                     newf[i+1, j] + newf[i-1, j])
    return newf

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
        gauss_seidel(f, num_iterations)
        elapsed_time = time.time() - start_time

        performance[N] = elapsed_time
        print(f"Grid size {N}x{N} took {elapsed_time:.4f} seconds")

    return performance

def run_profiler():

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
    run_profiler()

