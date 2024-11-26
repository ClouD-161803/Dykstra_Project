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
    errors_for_plotting = [e.copy()] # initialise with all zeros
    # print(f"Errors for plotting {errors_for_plotting}") for debugging

    # Path
    path = [z.copy()]  # Initialize the path with the original point

    # Active halfspaces matrix  (V9)
    # if plot_active_halfspaces:
    active_half_spaces = [[np.zeros_like(n) for _ in range(max_iter)]
                            for _ in range(n)]

    # Stall check (V9)
    stalling = False # initialise boolean
    # Matrix of successive projections
    x_historical = [[np.zeros_like(z) for _ in range(n)] for _ in range(max_iter)]
    prev_x_no_ffw = None

    # Optimal solution (V4)
    actual_projection = find_optimal_solution(z, N, c, dimensions)
    # Initialise errors vector
    squared_errors = np.zeros(max_iter)
    # Initialise vectors for tracking stalling and convergence
    stalled_errors = np.zeros(max_iter)
    converged_errors = np.zeros(max_iter)

    # Main body of Dykstra's algorithm
    stalling: bool = False
    k_stalling: int = 1  # number of rounds to fast forward when stalling occurs
    m_stalling: int | None = None  # index of half-space from which k_stalling is computed
    for i in range(max_iter):
        # Iterate over every half plane
        for m, (normal, offset) in enumerate(zip(N, c)):
            # Get m - n index using modulo operator, which ensures
            # we get an index between 0 and n (non-negative)
            index = (m - n) % n  # this is essentially just m-n with zeros for m<n
            x_temp = x.copy() # temporary variable (x_m)

            # Find number of rounds to fast forward after 1 round of stalling
            if stalling and m_stalling == m:
                n_fast_forward = int(min(
                    [np.ceil(- np.dot(e[m], normal) / (np.dot(x_historical[i-1][m-1], normal) - offset))
                     if np.dot(x_historical[i-1][m-1], normal) < offset else 1e6
                     for m, (normal, offset) in enumerate(zip(N, c))]
                ))
                n_fast_forward -= 1
                # NOTE: There still seems to be a bug here. Swap the order of line and box, and then it becomes visible
                # that in one situation I have to substract 1 from n_fast_forward, and in the other it is fine as it is.
                # This is most probably related to some bug in how em is updated during the stalling period. To
                # simplify all of this, the stalling check could occur in the range(max_iter) loop, checking if all
                # half-spaces are stalling. Then, the fast forwarding could be applied here.
                print(f"Fast forwarding {n_fast_forward} rounds to exit stalling at iteration {i}. ")
                # Update all errors for the following round
                for m, (normal, offset) in enumerate(zip(N, c)):
                    e[m] = e[m] + n_fast_forward * (x_historical[i-1][m-1] - x_historical[i-1][m])
                    if not is_in_half_space(x + e[index], normal, offset):
                        active_half_spaces[m][i] = 1
                stalling = False
                m_stalling = None

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
            if i > 0:
                # Check for stalling if x_m = x_m-n
                if ((not stalling) and (active_half_spaces[m][i] == 1) and
                        np.array_equal(x_historical[i][m], x_historical[i - 1][m])):
                    stalling = True
                    m_stalling = m
                    print(f"Stalling detected at iteration {i} and half-space {m_stalling}")

            # Path
            path.append(x.copy())  # Add the updated x to the path

        # Errors
        if plot_errors:
            errors_for_plotting.append(e.copy()) # update error matrix

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