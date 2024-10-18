"""
This module provides functions for visualizing the intersection of
half-spaces in 1D and 2D.

Functions:
- plot_2d_space(N, c, X, Y, label, cmap, ax):
    Plots a 2D region defined by the intersection of half-spaces.
- plot_1d_space(N, c, label, cmap, ax):
    Plots a 1D region (line) defined by the intersection of half-spaces.
- plot_half_spaces(Nc_pairs, num_of_iterations, ax):
    Plots the intersection of multiple sets of half-spaces.
- plot_path(path, ax, errors_for_plotting=None, plot_errors=False):
    Plots the path followed by Dykstra's algorithm during projection,
    with optional error plotting.
- plot_active_spaces(active_spaces, num_of_iterations):
    Plots the activity of half-spaces over iterations.

Notes:
- Change the global variables for plot domain as needed.
- plt.show() is omitted for adding extra points on the same plot.
    Ensure to call plt.show() after calling plot_half_planes().

Additional Features:
- Ability to plot on specific axes.
- Error and active half-space visualization.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_2d_space(N: np.ndarray, c: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  label: str, cmap: str, ax) -> None:
    """
    Plots a 2D region defined by the intersection of half spaces.

    Args:
        N: Matrix of normal vectors.
        c: Vector of constant offsets.
        X: 2D array of x coordinates.
        Y: 2D array of y coordinates.
        label: Label for the plot.
        cmap: Colormap for the plot.
        ax: Axes handle for plotting.

    Returns:
        None
    """

    # Initialize Z to 1 (assuming all points are inside initially)
    Z = np.ones_like(X)

    # This loop checks whether a pair of points X,Y lies within the set of constraints
    # Iterate over each half-plane (one per row of N)
    for i in range(N.shape[0]):
        # Calculate the dot product and compare with the offset
        # - X.ravel() and Y.ravel() flatten the 2D arrays X and Y into 1D arrays;
        # - np.vstack([...]) stacks the flattened X and Y arrays vertically,
        #   creating a 2D array where each row represents a point (x, y) from the grid.
        # - .T transposes the resulting array, so each column now represents a point.
        dot_product = np.dot(np.vstack([X.ravel(), Y.ravel()]).T, N[i])
        # Z will contain 1s for points inside the intersection of all
        # half-planes and 0s for points outside
        Z = np.where(dot_product.reshape(X.shape) > c[i], 0, Z)  # Reshape before comparison

    # Colours
    colourmap = cm.get_cmap(cmap)
    # Map colourmap to a single colour for boundaries
    colour = colourmap(0.69) # some arbitrary constant (totally random)

    # Plot the filled contours
    ax.contourf(X, Y, Z, cmap=colourmap, alpha=0.5)

    # Create a dummy plot with the label for the legend
    ax.plot([], [], color=colour, alpha=0.5, label=label)


def plot_1d_space(N: np.ndarray, c: np.ndarray, label: str, cmap: str, ax,
                  x_range: list)-> None:
    """
    Plots a 1D region (line) defined by the intersection of half spaces.

    Args:
        N: Matrix of normal vectors.
        c: Vector of constant offsets.
        label: Label for the plot.
        cmap: Colormap for the plot.
        ax: Axes handle for plotting.

    Returns:
        None
    """

    # Colours
    colourmap = cm.get_cmap(cmap)
    colour = colourmap(0.69) # again, a total coincidence

    # Check for division by 0
    if N[0, 1] == 0:
        # Vertical line
        ax.axvline(x=c[0] / N[0, 0], linestyle='-', linewidth=2,
                    label='Vertical Line', color=colour)
    elif N[0, 0] == 0:
        # Horizontal line
        ax.axhline(y=c[0] / N[0, 1], linestyle='-', linewidth=2,
                    label='Horizontal Line', color=colour)
    else:
        # General line
        x_line = np.linspace(x_range[0], x_range[1], 100)  # Adjust range as needed
        y_line = (c[0] - N[0, 0] * x_line) / N[0, 1]
        ax.plot(x_line, y_line, linewidth=2, label=label, color=colour)


def plot_half_spaces(Nc_pairs: list, num_of_iterations: int, ax,
                     x_range: list, y_range: list) -> None:
    """
    Plots the intersection of multiple sets of half-spaces defined by Nc_pairs,
    where each pair consists of N (normals) and c (offsets) such that N*x <= c.
    Nc_pairs is of the form [('label'_i, 'cmap_i', N_i, c_i), (...), ...]

    Args:
        Nc_pairs: List of tuples containing ('label', 'cmap', N, c).
        num_of_iterations: Number of iterations.
        ax: Axes handle for plotting.

    Returns:
        None
    """

    try:

        # Create a grid of x and y values
        x = np.linspace(x_range[0], x_range[1], 500)  # Adjust range as needed
        y = np.linspace(y_range[0], y_range[1], 500)
        X, Y = np.meshgrid(x, y)

        for label, cmap, N, c in Nc_pairs:
            # Distinguish cases based on the rank of N
            rank = np.linalg.matrix_rank(N)
            # 1d case (line)
            if rank == 1:
                plot_1d_space(N, c, label, cmap, ax, x_range)
            # 2d case
            elif rank == 2:
                plot_2d_space(N, c, X, Y, label, cmap, ax)
            # Everything else (ADD 3D CASE AT SOME POINT)
            else:
                raise ValueError("Dimension not supported."
                                 "Please provide N and c for 1D or 2D cases.")

        # Set aspect ratio to 'equal'
        ax.set_aspect('equal')

        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"Modified Dykstra's algorithm "
                     f"executed for {num_of_iterations} iterations")
        ax.grid(True)

        # Add the legend
        ax.legend()

    except TypeError as e:
        print(f"TypeError occurred: {e}."
              f"Please ensure Nc_pairs is a list of tuples.")
    except ValueError as e:
        print(f"ValueError occurred: {e}."
              f"Check the format of Nc_pairs or the dimensions of N.")


def plot_path(path: list, ax, errors_for_plotting: np.ndarray=None,
              plot_errors: bool=False) -> None:
    """
    Plots the path followed by Dykstra's algorithm during projection.

    Can also plot the errors at each iteration.

    Args:
        path: A list of points representing the intermediate steps taken by the algorithm.
        ax: Axes handle for plotting.
        errors_for_plotting: Array containing error vectors.
        plot_errors: Boolean to control whether to plot the error vectors.

    Returns:
        None
    """

    # Extract x and y coordinates from the path
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]

    # Plot the path
    ax.plot(x_coords, y_coords, marker='.', linestyle='--',
             color='blue', linewidth=0.5, markersize=1,
             label='Projection Path')

    # Plot the errors - this is quite complex but follows dykstra's structure
    # TODO Need to fix this implementation at some point
    if plot_errors:
        n = len(errors_for_plotting[0]) # number of halfspaces
        m = 0 # running index (not sure how else to do this)
        max_iter = len(errors_for_plotting) - 1 # algorithm iterations
        iteration = 1 # external for loop iterations
        for errors in errors_for_plotting:
            for error in errors:
                index = (m - n) % (iteration * n) + n
                if index < max_iter * n:
                    # debugging prints
                    # print(f"n: {n} | m: {m} | index: {index} | iteration: {iteration}")
                    # print(index, (x_coords[index], y_coords[index]), error)

                    # Plot error vectors as quivers
                    ax.quiver(x_coords[index], y_coords[index], error[0], error[1],
                              angles='xy', scale_units='xy', scale=1, alpha=0.3)
                    m += 1
            iteration += 1

    # Add legend
    ax.legend()


def plot_active_spaces(active_spaces: list, num_of_iterations: int, fig, gs) -> None:
    """
        Plots the activity of half-spaces over iterations.

        Args:
            active_spaces: List of active spaces.
            num_of_iterations: Number of iterations.
            fig: Figure handle for plotting.
            gs: GridSpec handle for plotting.

        Returns:
            None
    """

    # Find total number of halfspaces
    num_of_spaces = len(active_spaces)
    iterations = np.arange(0, num_of_iterations, 1) # for plotting
    ax_vector = np.zeros(num_of_spaces, dtype=object) # initialise vector

    # Update axes
    for i, active_space in enumerate(active_spaces):
        ax_vector[i] = fig.add_subplot(gs[i, 1])

    # Loop over all axes to plot
    for i, (active_space, ax) in enumerate(zip(active_spaces, ax_vector)):

        # Plot each halfspace on a separate axis
        ax.plot(iterations, active_space, color='black',
                 label=f'Halfspace {i}', linestyle='-', marker='o')
        ax.set_ylim(0, 1)

        # Set title for the first plot
        if i == 0:
            ax.set_title('Halfspace Activity')
        
        # Grid and legend
        ax.grid(True)
        ax.legend()