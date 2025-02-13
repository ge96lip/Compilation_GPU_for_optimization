import numpy as np
import matplotlib.pyplot as plt
import cythonfn2
import time

def mandelbrot(c, max_iter=100):
    """Computes the number of iterations before divergence."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(width, height, x_min, x_max, y_min, y_max, max_iter=100):
    """Generates the Mandelbrot set image."""
    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)
    image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            c = complex(x_vals[j], y_vals[i])
            image[i, j] = cythonfn2.mandelbrot(c, max_iter)

    return image


def run_profile():

  # Parameters
  width, height = 1000, 800
  x_min, x_max, y_min, y_max = -2, 1, -1, 1

  # Generate fractal
  start_time = time.time()
  image = mandelbrot_set(width, height, x_min, x_max, y_min, y_max)
  elapsed_time = time.time() - start_time
  print("Elapsed time: ", elapsed_time)
  # Display
  plt.imshow(image, cmap='inferno', extent=[x_min, x_max, y_min, y_max])
  plt.colorbar()
  plt.title("Mandelbrot Set")
  plt.show()