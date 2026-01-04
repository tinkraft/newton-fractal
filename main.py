import numpy as np
from newtonfractal import plot_newton_fractal
from fractalbox import box_counting

f = lambda z: z**3 - 1
fprime = lambda z: 3 * z**2

points = plot_newton_fractal(f, fprime, n=1000)

# Define scales
scales = np.logspace(-3, 0, num=50)

result = box_counting(points, scales, method="original")

print("Fractal Dimension:", result["fd"])
print("R²:", result["r_squared"])