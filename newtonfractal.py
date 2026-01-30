import numpy as np
import matplotlib.pyplot as plt
from numpy import roots

from fractalbox import box_counting

TOL = 1.0e-8


def newton(z0, f, fprime, MAX_IT=1000):
    """The Newton-Raphson method applied to f(z).

    Returns the root found, starting with an initial guess, z0, or False
    if no convergence to tolerance TOL was reached within MAX_IT iterations.

    """

    z = z0
    for i in range(MAX_IT):
        dz = f(z) / fprime(z)
        if abs(dz) < TOL:
            return z
        z -= dz
    return False


def plot_newton_fractal(f, fprime, n=200, domain=(-2, 2, -2, 2)):
    """Plot a Newton Fractal by finding the roots of f(z).

    The domain used for the fractal image is the region of the complex plane
    (xmin, xmax, ymin, ymax) where z = x + iy, discretized into n values along
    each axis.

    """

    roots = []
    rootCount = []
    m = np.zeros((n, n))

    def get_root_index(roots, r):
        """Get the index of r in the list roots.

        If r is not in roots, append it to the list.

        """

        try:
            return np.where(np.isclose(roots, r, atol=TOL))[0][0]
        except IndexError:
            roots.append(r)
            rootCount.append(0)
            return len(roots) - 1

    xmin, xmax, ymin, ymax = domain
    for ix, x in enumerate(np.linspace(xmin, xmax, n)):
        for iy, y in enumerate(np.linspace(ymin, ymax, n)):
            z0 = x + y * 1j
            r = newton(z0, f, fprime)
            if r is not False:
                ir = get_root_index(roots, r)
                m[iy, ix] = ir
                rootCount[ir] += 1
                # coords = np.argwhere(m)
                # if len(coords) > 0:
                #     avg_row = np.mean(coords[:, 0])
                #     avg_col = np.mean(coords[:, 0])
                #     centroid_x = r[avg_row]
                #     centroid_y = r[avg_col]

    for target in range(0, len(roots)):
        coords = np.argwhere(m == target)
        centroid_row, centroid_col = coords.mean(axis=0)
        centroid_x = xmin + (centroid_col / (n - 1)) * (xmax - xmin)
        centroid_y = ymin + (centroid_row / (n - 1)) * (ymax - ymin)
        print(f"Centroid of R={target}: ({centroid_x}, {centroid_y})")

    plt.imshow(m, origin="lower")
    plt.axis("off")
    plt.show()

    print(rootCount)
    return m
