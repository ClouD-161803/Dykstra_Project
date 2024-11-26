"""
This module implements a modified version Dykstra's algorithm for projecting
a point onto the intersection of convex sets (specifically, half-spaces).
This version can detect stalling and exit it in one iteration.

Functions:
- dykstra_projection(z, N, c, max_iter, track_error=False, min_error=1e-3,
                dimensions=2, plot_errors=False, plot_active_halfspaces=False):
Projects a point 'z' onto the intersection of multiple half-spaces defined
by the matrix N and vector c using a modified version of dykstra that
detects stalling and exits it in one iteration.

Additional Features:
- Error tracking: Option to track and plot errors at each iteration.
- Convergence and stalling detection.
- Generalised for any number of dimensions.
- Inactive half-space removal.
- Active and inactive half-space plotting.
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
    """
    Projects a point 'z' onto the intersection of convex sets (half spaces)
    using a modified version of Dykstra's algorithm.
    This version includes stalling detection and the ability to exit stalling.

    Args:
        z: Initial point.
        N: Matrix of normal vectors.
        c: Vector of constant offsets.
        max_iter: Maximum number of iterations.
        track_error (bool, optional): Track the squared error at each iteration.
        min_error (float, optional): Minimum error threshold for convergence.
        dimensions (int, optional): Number of dimensions.
        plot_errors (bool, optional): Plot errors at each iteration.
        plot_active_halfspaces (bool, optional): Plot active half spaces.
        delete_spaces (optional): Whether to delete inactive halfspaces

    Returns:
        tuple: Final projected point, path taken, error metrics (if tracking),
        errors for plotting (if selected), and active half spaces (if selected).

    Additional Features:
        - Stalling detection: Checks if the algorithm is stalling by comparing
                                historical points.
        - Exit stalling: If stalling is detected, the algorithm attempts to
                            exit stalling by adjusting the point to cross the boundary.
        - Error tracking: Option to track and plot errors at each iteration.
        - Convergence detection: Detects whether the algorithm has converged
                                    based on the minimum error threshold.
        - Generalised for any number of dimensions.
        - Inactive half-space removal.
        - Active and inactive half-space plotting.
    """

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
    # errors_for_plotting = [e.copy()] # initialise with all zeros
    errors_for_plotting = np.array([np.zeros_like(e) for _ in range(max_iter)])
    # print(f"Errors for plotting {errors_for_plotting}") # for debugging

    # Path
    path = [z.copy()]  # Initialize the path with the original point

    # Active halfspaces matrix  (V9)
    # if plot_active_halfspaces:
    active_half_spaces = np.array([[np.zeros_like(n) for _ in range(max_iter)]
                            for _ in range(n)])

    # Stall check (V9)
    stalling = False # initialise boolean
    # Matrix of successive projections
    x_historical = np.array([[np.zeros_like(z) for _ in range(n)] for _ in range(max_iter)])
    # print(f"x_historical: {x_historical}") # for debugging
    # print((np.array(x_historical).shape)) # for debugging
    # print((np.array(errors_for_plotting).shape)) # for debugging

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
                if not is_in_half_space(x + e[index], normal, offset):
                    # Set item to 1 if halfspace is active, 0 otherwise
                    active_half_spaces[m][i] = 1

            # Update x_m+1
            x = project_onto_half_space(x_temp + e[index], normal, offset)

            # Update e_m
            e[m] = e[index] + 1 * (x_temp - x)  # change to 0 for MAP

            # V9
            x_historical[i][m] = x.copy()

            # Check for stalling (V9)
            if i == 0 or m == 0:
                pass
            else:
                # Debugging
                # diff1 = x_current[i][m] - x_current[i][m - 1]
                # diff2 = x_current[i - 1][m] - x_current[i - 1][m - 1]
                # print(diff1, diff2)

                # Check for stalling if x_m = x_m-n
                if np.array_equal(x_historical[i][m], x_historical[i - 1][m]):
                    stalling = True
                else:
                    stalling = False
                # print(stalling) # for debugging

            # Exit stalling (V9)
            if stalling:
                # Constant growth error
                diff = x_historical[i][m] - x_historical[i][m - 1]
                # Need to continue if this half-space cannot become inactive
                tmp = diff / normal
                diff_sign = tmp[np.isfinite(tmp)][0]
                if diff_sign > 0:
                    continue
                k = 0
                # Iterate until boundary is crossed
                while not is_in_half_space(x_temp + k*diff, normal, offset):
                    k += 1
                # Update x
                x = x_temp + k * diff

            # Path
            path.append(x.copy())  # Add the updated x to the path

        # Errors
        if plot_errors:
            errors_for_plotting[i][m] = e[m].copy() # update error matrix
            # print(f"Errors for plotting {errors_for_plotting}") # for debugging

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
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, errors_for_plotting, active_half_spaces
    elif track_error and plot_active_halfspaces:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, None, active_half_spaces
    elif track_error and plot_errors:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, errors_for_plotting, None
    elif track_error:
        error_tuple = (squared_errors, stalled_errors, converged_errors)
        return x, path, error_tuple, None, None
    else:
        return x, path, None, None, None