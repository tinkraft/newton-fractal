import numpy as np

from newtonfractal import plot_newton_fractal
from fractalbox import box_counting

f = lambda z: (z**3 + (4/5)*z**(2) + (4/5)*z - (1/5))
fprime = lambda z: 3 * z**(2) + (8/5)*z + (4/5)

points = plot_newton_fractal(f, fprime, n=1000)



# Define scales
scales = np.logspace(5, 0, num=50)

result = box_counting(points, scales, method="original")

print("Fractal Dimension:", result["fd"])
print("R²:", result["r_squared"])