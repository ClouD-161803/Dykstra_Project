# Dykstra Project

Implementation of **Dykstra's algorithm** for projecting a point onto the intersection of multiple convex sets (half-spaces) in Hilbert space. This project includes implementations of the standard algorithm, a hybrid MAP-Dykstra variant, and a modified version with stalling detection and prevention.

## Overview

Dykstra's algorithm is a cyclic projection method used to find the projection of a point onto a convex feasible region defined by the intersection of multiple half-spaces. This is a fundamental problem in optimization, control theory, and signal processing.

### Key Features

- **Standard Dykstra's Algorithm** - Classic cyclic projection implementation
- **MAP-Dykstra Hybrid** - Combines Method of Alternating Projections with Dykstra for improved performance
- **Stalling Detection** - Detects and prevents algorithmic stalling (when the algorithm gets stuck cycling)
- **Interactive Visualization** - 2D visualization of projection paths, half-spaces, and convergence
- **Error Tracking** - Monitors convergence and distance to optimal solution
- **Rounded Box Support** - Edge-rounding utilities for non-axis-aligned convex regions
- **Generalized Implementation** - Works for any number of dimensions and half-spaces

### Quick Start

Navigate to the working version and run the main example:

```bash
cd Python/Working\ Version
python main.py
```

This will project a point onto the intersection of a box and a line, displaying:

- The original point being projected
- The final projection result
- The iteration history and convergence behavior
- Active and inactive half-spaces
- A visual representation of the 2D feasible region

### Project Locations

- **LaTeX directory** - Up-to-date mathematical report and derivations: `/Latex/Current Version`
- **Python directory** - Production code: `/Python/Working Version`

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Example 1: Basic Projection (Default)

The main script projects a point to the intersection of a box and a line:

```python
# In Python/Working Version/main.py
z = np.array([-4., 1.4])  # Point to project
# Box constraints: -1 <= x <= 1, -1 <= y <= 1
# Line constraint: x/2 + y = 1
```

Run with:

```bash
python main.py
```

### Example 2: Different Test Cases

Uncomment the desired test case in `main.py` to test various scenarios:

- **Simple top left (with stalling)** - Tests stalling detection capability
- **Simple top left (no stalling)** - Basic example without stalling issues
- **Point in intersection** - Tests handling of points already in the feasible region
- **Very far points** - Tests convergence from distant starting points in all directions
- **Various corners** - Tests algorithm from all corners/directions

Each case has a corresponding starting point `z`, plot range, and configuration.

### Example 3: Switching Between Solvers

Three solver variants are available in `main.py`:

```python
# Standard Dykstra's Algorithm
solver = DykstraProjectionSolver(
    z, A, c, max_iter,
    track_error=True,
    plot_errors=False,
    plot_active_halfspaces=True,
    delete_spaces=True
)

# Hybrid MAP-Dykstra (often faster)
solver = DykstraMapHybridSolver(
    z, A, c, max_iter,
    track_error=True,
    plot_errors=False,
    plot_active_halfspaces=True,
    delete_spaces=True
)

# Dykstra with Stalling Detection
solver = DykstraStallDetectionSolver(
    z, A, c, max_iter,
    track_error=True,
    plot_errors=False,
    plot_active_halfspaces=True,
    delete_spaces=True
)

result = solver.solve()
projection_point = result.projection
```

### Example 4: Rounded Box Constraints

For non-rectangular convex regions, use edge rounding:

```python
from edge_rounder import rounded_box_constraints

center = (0, 0)
width = 2
height = 2
corner_count = 3  # Number of corners to round
N_box, c_box = rounded_box_constraints(center, width, height, corner_count)
```

### Example 5: Custom Half-Spaces

Define arbitrary convex constraints:

```python
import numpy as np

# Define half-spaces: N*x <= c
# Example: Create 4 constraints forming a box
N = np.array([
    [1., 0.],   # x <= 1
    [-1., 0.],  # x >= -1
    [0., 1.],   # y <= 1
    [0., -1.]   # y >= -1
])
c = np.array([1., 1., 1., 1.])

# Add more constraints (e.g., a line)
N_line = np.array([[1/2, 1], [-1/2, -1]])
c_line = np.array([1, -1])

# Combine all constraints
A = np.vstack([N, N_line])
c_full = np.hstack([c, c_line])

# Create and solve
solver = DykstraProjectionSolver(z, A, c_full, max_iter=50)
result = solver.solve()
```

---

## Core Modules

### `convex_projection_solver.py`

Main solver implementations:

- `ConvexProjectionSolver` - Base abstract class
- `DykstraProjectionSolver` - Standard Dykstra's cyclic projection algorithm
- `DykstraMapHybridSolver` - Hybrid Method of Alternating Projections with Dykstra
- `DykstraStallDetectionSolver` - Stalling-aware variant with early termination

### `visualiser.py` and `VerticalVisualiser`

Visualization system for projection results:

- **Visualiser** - Horizontal layout for results
- **VerticalVisualiser** - Vertical layout for results
- Displays half-space constraints and their boundaries
- Shows complete projection paths through iterations
- Highlights active vs. inactive constraints
- Renders error evolution and convergence metrics
- Includes quiver plots for gradient information

### `gradient.py`

Gradient and optimisation utilities:

- Quadratic programming solver

### `edge_rounder.py`

Utilities for rounding box corners:

- `rounded_box_constraints()` - Create smoothed box regions

### `projection_result.py`

Data structure storing solver outputs:

- Final projection point
- Complete iteration history
- Convergence metrics and error tracking
- Active half-space indices at each iteration
- Distance to optimal solution

---

## Directory Structure

```md
|   README.md                          (Project documentation)
|   requirements.txt                   (Python dependencies)
|
+---Application
|   \---Final Drafts                   (Project deliverables and applications)
|           
+---Books & Articles                   (Research papers and references)
|   +---Baushke                        (Baushke research materials)
|   +---claudio_projection             (Personal research and notes)
|   \---Notes                          (Reference lecture notes)
|           C21_MPC_Lecture_Notes.pdf
|           C21_MPC_Problems.pdf
|           dykstra.pdf
|           Dykstra's Projection Algorithm.pdf
|           LQR + Riccati.pdf
|           MPC derivation of condensed form.pdf
|           
+---Latex
|   +---Current Version                (Main paper and report)
|   |   |   paper.tex                  (Main research paper)
|   |   |   master_bib_abbrev.bib      (Bibliography)
|   |   |   mymath.sty                 (Custom LaTeX math macros)
|   |   |
|   |   +---Figures                    (Generated plots and visualizations)
|   |   |
|   |   +---Packages
|   |   |   \---packages.tex           (LaTeX package configurations)
|   |   |
|   |   \---Sections                   (Main paper sections)
|   |           1-Introduction.tex
|   |           2-Dykstra Background.tex
|   |           3-Main Results.tex
|   |           4-Conclusion.tex
|   |           Fast Forwarding Stalling Period.tex
|   |           The Polyhedral Case.tex
|   |           
|   +---Idris Version                  (Functional programming implementation)
|   \---Initial Version                (Early version iterations)
|               
\---Python
    +---Previous Versions              (Development history)
    |   |   (Versions 1-9 showing algorithm evolution)
    |   |
    |   +---Version 1                  (Initial implementation)
    |   +---Version 2                  (Added plotting)
    |   +---Version 3                  (Path visualization)
    |   +---Version 4                  (Added gradients)
    |   +---Version 5-9                (Iterative improvements)
    |           
    \---Working Version               (✓ CURRENT PRODUCTION CODE)
            main.py                    (🚀 Entry point - run this to test)
            convex_projection_solver.py (Core solver implementations)
            visualiser.py              (2D visualization engine)
            projection_result.py       (Result data structure)
            gradient.py                (Gradient/optimization utilities)
            edge_rounder.py            (Box rounding utilities)
            dykstra.py                 (Algorithm helper functions)
```

---

## Algorithm Details

### Standard Dykstra's Algorithm

The algorithm iteratively projects onto each constraint in a cyclic manner:

1. For each half-space constraint in sequence
2. Project the current point onto that constraint boundary
3. Store the projection offset
4. Repeat until convergence or iteration limit

### Stalling Problem

Dykstra's algorithm can "stall" when it gets trapped cycling between adjacent constraints without converging. This project includes detection and prevention mechanisms.

### Modified Algorithm

The stalling-aware variant detects when cycling occurs and employs strategies to escape, improving convergence in difficult cases.

---

## Performance Considerations

- **Dimension Independence** - Works for any dimension (though visualization limited to 2D)
- **Scalability** - Performance depends on number of constraints and dimensionality
- **Convergence** - Guaranteed to converge for convex constraint sets
- **Stalling** - Modified version handles degenerate cases better

---

## References

Research papers and academic references are stored in `/Books & Articles`. Key topics covered:

- Convergence analysis of Dykstra's algorithm
- Stalling phenomena in cyclic projection methods
- Method of Alternating Projections (MAP)
- Model Predictive Control applications
- Proximal splitting methods

---

## Author

Claudio Vestini - University of Oxford

## Funding

Project funded by Keble Research Grant KSRG118
"
