import numpy as np
import matplotlib.pyplot as plt
from newtonfractal import plot_newton_fractal

f = lambda z: z**3 - 1
fprime = lambda z: 3 * z**2

points = plot_newton_fractal(f, fprime, n=1000)


# 2. Set up the grid (The complex plane)
res = 1000  # Number of pixels (1000x1000 = 1 million data points)
x = np.linspace(-1, 1, res)
y = np.linspace(-1, 1, res)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

roots = [1 + 0j, -0.5 + 0.866j, -0.5 - 0.866j]
basins = np.zeros(Z.shape, dtype=int)

# Assign a number (0, 1, 2) to each pixel based on which root it reached
for idx, r in enumerate(roots):
    basins[np.abs(Z - r) < 0.1] = idx + 1

# 5. Calculate Area Statistics
total_pixels = res * res
counts = [np.sum(basins == 1), np.sum(basins == 2), np.sum(basins == 3)]
percentages = [(c / total_pixels) * 100 for c in counts]

sum()

# Print the data for your IA
print(f"Basin 1 (Yellow): {percentages[0]:.2f}%")
print(f"Basin 2 (Teal):   {percentages[1]:.2f}%")
print(f"Basin 3 (Purple): {percentages[2]:.2f}%")

# 6. Check for Symmetry
symmetry_error = max(percentages) - min(percentages)
print(f"Area Symmetry Error: {symmetry_error:.4f}%")

# 7. Visualize
plt.imshow(basins, extent=[-1, 1, -1, 1])
plt.title("Newton Basins for $z^3 - 1$")
plt.show()

