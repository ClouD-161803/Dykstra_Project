"""
This module implements Dykstra's algorithm for projecting a point onto the
intersection of convex sets (specifically, half-spaces).

Functions:
- dykstra_projection(z, N, c, max_iter, track_error=False, min_error=1e-3,
                dimensions=2, plot_errors=False, plot_active_halfspaces=False):
Projects a point 'z' onto the intersection of multiple half-spaces
defined by the matrix N and vector c using dykstra's method.

Additional Features:
- Error tracking: Option to track and plot errors at each iteration.
- Convergence and stalling detection.
- Generalised for any number of dimensions.
- Inactive half-space removal.
"""


import numpy as np
from dykstra_functions import (is_in_half_space,
                               project_onto_half_space,
                               delete_inactive_half_spaces,
                               find_optimal_solution)


def dykstra_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int, track_error: bool=False,
                       min_error: int=1e-3, dimensions: int=2,
                       plot_errors: bool=False,
                       plot_active_halfspaces: bool=False,
                       delete_spaces: bool=False) -> tuple:
    """Projects a point 'z' onto the intersection of convex sets H_i (half spaces).
    The convex set parameters (unit normals and constant offsets) are packaged
    into matrix N and vector c respectively, such that:
    N*x <= c -> [n_i^T]*x <= {c_i} yields a set of linear inequalities of the kind
    <x,n_i> <= c_i for all i = rowcount(N).

    Notes:
    - Uses Dykstra's algorithm.
    - Parameters N and c represent the unit normals and constant offsets of the half spaces.
    - Halts after max_iter iterations.
    - Error tracking includes squared, stalled, and converged errors.
    - Generalised to any number of dimensions.
    - Includes functionality for plotting errors and active/inactive half spaces.
    - Removes inactive half spaces at the start.

    Args:
        z: Initial point.
        N: Matrix of normal vectors.
        c: Vector of constant offsets.
        max_iter: Maximum number of iterations.
        track_error (optional): Whether to track the error at each iteration.
        min_error (optional): Minimum error threshold for convergence.
        dimensions (optional): Number of dimensions.
        plot_errors (optional): Whether to plot errors at each iteration.
        plot_active_halfspaces (optional): Whether to plot active half spaces.
        delete_spaces (optional): Whether to delete inactive halfspaces

    Returns:
        tuple: Final projected point, path taken, error metrics if tracking,
        errors for plotting if selected, and active half spaces if selected."""


    # Eliminate inactive halfspaces (V9)
    if delete_spaces:
        N, c = delete_inactive_half_spaces(z, N, c)

    # Initialise variables
    n = N.shape[0]  # Number of half-spaces
    x = z.copy()  # create a deep copy of the original point
    errors = np.zeros_like(z) # individual error vectors
    e = [errors] * n  # list of a number of error vectors equal to n

    # Vector for storing all errors (V8)
    # if plot_errors:
    errors_for_plotting = [e.copy()] # initialise with all zeros
    # print(f"Errors for plotting {errors_for_plotting}") for debugging

    # Path (V3)
    path = [z.copy()]  # Initialize the path with the original point

    # Active halfspaces vector (V9)
    # if plot_active_halfspaces:
    active_half_spaces = [[np.zeros_like(n) for _ in range(max_iter)]
                          for _ in range(n)]

    # Optimal solution (V4)
    actual_projection = find_optimal_solution(z, N, c, dimensions)
    # Initialise errors vector
    squared_errors = np.zeros(max_iter)
    # Initialise vectors for tracking stalling and convergence
    stalled_errors = np.zeros(max_iter)
    converged_errors = np.zeros(max_iter)

    # Main body of Dykstra's algorithm
    for i in range(max_iter):
        # Iterate over every half plane
        for m, (normal, offset) in enumerate(zip(N, c)):
            # Get m - n index using modulo operator, which ensures
            # we get an index between 0 and n (non-negative)
            index = (m - n) % n  # this is essentially just m-n with zeros for m<n
            x_temp = x.copy() # temporary variable (x_m)

            # Check if current point is in the halfspace (V9)
            if plot_active_halfspaces:
                if not is_in_half_space(x_temp + e[index], normal, offset):
                    # Set item to 1 if halfspace is active, 0 otherwise
                    active_half_spaces[m][i] = 1

            # Update x_m+1
            x = project_onto_half_space(x_temp + e[index], normal, offset)

            # Update e_m
            e[m] =  + e[index] + 1 * (x_temp - x) # change to 0 for MAP

            # Path
            path.append(x.copy())  # Add the updated x to the path

        # Errors
        if plot_errors:
            errors_for_plotting.append(e.copy()) # update error metrix

        # Track the squared error (V4)
        if track_error:
            distance = actual_projection - x
            error = round(np.dot(distance, distance), 10) # num error check
            # Check stalling, modulo used to avoid negative index
            i_minus_one = (i - 1) % max_iter
            # Define conditions for if check
            is_equal1 = squared_errors[i_minus_one] == error
            is_equal2 = stalled_errors[i_minus_one] == error
            # Check if we have converged
            if error < min_error:
                converged_errors[i] = error
                stalled_errors[i] = None
            # Check if we are stalling
            elif is_equal1 or is_equal2:
                stalled_errors[i] = error
                converged_errors[i] = None
            else:
                stalled_errors[i] = None
                converged_errors[i] = None
            # Append error
            squared_errors[i] = error

    if track_error and plot_errors and plot_active_halfspaces:
        # Wrap everything up into a tuple
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        # Return additional vector
        return x, path, error_tuple, errors_for_plotting, active_half_spaces
    elif track_error and plot_active_halfspaces:
        # Wrap everything up into a tuple
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        # Return additional vector
        return x, path, error_tuple, None, active_half_spaces
    elif track_error and plot_errors:
        # Wrap everything up into a tuple
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        # Return additional vector
        return x, path, error_tuple, errors_for_plotting, None
    elif track_error:
        # Wrap everything up into a tuple
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        # Return additional vector
        return x, path, error_tuple, None, None
    else:
        # Return finite-time approximation x to projection task and error e
        return x, path, None, None, None