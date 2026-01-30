# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Newton fractal visualization and fractal dimension analysis toolkit. It generates Newton-Raphson fractals for complex polynomials and computes their fractal dimensions using various methods.

## Running the Code

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main script (generates fractal and computes fractal dimension)
python main.py

# Run alternative visualization script
python try.py
```

Dependencies: numpy, matplotlib, scipy, pandas (installed in .venv)

## Architecture

**newtonfractal.py** - Core Newton-Raphson fractal generation:
- `newton(z0, f, fprime)` - Finds roots using Newton-Raphson iteration
- `plot_newton_fractal(f, fprime, n, domain)` - Generates the fractal image by iterating over a grid of complex starting points and coloring by which root each converges to

**fractalbox.py** - Fractal dimension computation using multiple methods:
- `box_counting()` - Standard box counting with original/oversample/exact methods
- `box_counting_generalized()` - Generalized (Renyi) fractal dimensions
- `temporal_sampling()` - Fractal dimension via temporal sampling
- `corr_sum()` / `corr_sum_takens()` - Correlation sum methods
- `find_elbow_scale()` - Automatic scale selection for box counting

**utils.py** - Helper functions:
- Range finding for scale selection (`find_ranges_pct`, `find_ranges_ls`)
- Orthogonal distance regression (`perform_odr`)
- Reflex point detection (`get_reflex`)
- Test shape generators (circle, line, Koch snowflake)

**main.py** - Entry point that defines a polynomial, generates its Newton fractal, and computes the fractal dimension

## Key Concepts

- The `f` and `fprime` parameters are the polynomial and its derivative as Python lambdas
- Domain is specified as `(xmin, xmax, ymin, ymax)` in the complex plane
- Fractal dimension is computed via linear regression on log-log box count data
- Tolerance `TOL = 1.0e-8` controls Newton iteration convergence
