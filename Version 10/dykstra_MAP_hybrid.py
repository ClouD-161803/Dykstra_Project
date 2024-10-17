"""
This module implements a hybrid of MAP and Dykstra's algorithm for projecting
a point onto the intersection of convex sets (specifically, half-spaces).

Functions:
- normalise(normal, offset):
    Normalises half space normal and constant offset.
- is_in_half_space(point, unit_normal, constant_offset):
    Checks if a point lies within a single half-space.
- beta_check(point, N, c):
    Selects a value of beta based on whether the point lies within the
    intersection of half-spaces.
- project_onto_half_space(point, normal, offset):
    Projects a given point onto a single half-space.
- dykstra_projection(z, N, c, max_iter, track_error=False, min_error=1e-3,
                            dimensions=2, plot_errors=False):
    Projects a point 'z' onto the intersection of multiple half-spaces
    defined by the matrix N and vector c.

Additional Features:
- Error tracking: Option to track and plot errors at each iteration.
- Convergence and stalling detection.
- Beta parameter for selecting between MAP and Dykstra's method.
- Generalised for any number of dimensions.
"""


import numpy as np
from gradient import quadprog_solve_qp # needed to track error (V4)
from dykstra_functions import (is_in_half_space,
                               project_onto_half_space,
                               delete_inactive_half_spaces,
                               beta_check)


def dykstra_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int, track_error: bool=False,
                       min_error: int=1e-3, dimensions: int=2,
                       plot_errors: bool = False,
                       plot_active_halfspaces: bool = False) -> tuple:
    """
    Projects a point 'z' onto the intersection of convex sets (half spaces)
    via a modified version of dykstra's algorithm: if the current point to be
    projected lies outside the feasible region, we use MAP; otherwise, we
    use dykstra's method.

    Args:
        z: Initial point.
        N: Matrix of normal vectors.
        c: Vector of constant offsets.
        max_iter: Maximum number of iterations.
        track_error (bool, optional): Track the squared error at each iteration.
        min_error (float, optional): Minimum error threshold for convergence.
        dimensions (int, optional): Number of dimensions.
        plot_errors (bool, optional): Plot errors at each iteration.
        plot_active_halfspaces (optional): Whether to plot active half spaces.

    Returns:
        tuple: Final projected point, path taken, error metrics (if tracking),
                and errors for plotting (if selected).
    """

    try:
        # Initialise variables
        # print(f"\nThere are {N.shape[0]} halfspaces or N is {N}") # for debug
        n = N.shape[0]  # Number of half-spaces
        x = z.copy()  # create a deep copy of the original point
        errors = np.zeros_like(z) # individual error vectors
        e_dykstra = [errors] * n  # list of a number of error vectors equal to n
        e_MAP = [errors] * n # same but for MAP (V6)

        # Vector for storing all errors (V8)
        if plot_errors:
            errors_for_plotting = [e_dykstra.copy()] # initialise with all zeros
            # print(f"Errors for plotting {errors_for_plotting}") for debugging
        # Path
        path = [z.copy()]  # initialize the path with the original point

        # Optimal solution
        if track_error:
            # FROM DOCUMENTATION:
            # (see https://scaron.info/blog/quadratic-programming-in-python.html)
            # " The quadratic expression ∥Ax − b∥^2 of the least squares optimisation
            #   is written in standard form with P = 2A^TA and q = −2A^Tb "
            # We are solving min_x ∥x − z∥^2 s.t. Gx <= h so set:
            A = np.eye(dimensions)
            b = z.copy()
            # @ command is recommended
            P = 2 * np.matmul(A.T, A)
            q = -2 * np.matmul(A.T, b)
            G = N
            h = c
            # Find projection
            actual_projection = quadprog_solve_qp(P, q, G, h)
            # Initialise errors vector
            squared_errors = np.zeros(max_iter)
            # Initialise vectors for tracking stalling and convergence
            stalled_errors = np.zeros(max_iter)
            converged_errors = np.zeros(max_iter)

        # Main body of Dykstra's algorithm
        for i in range(max_iter):

            # Choose Beta at the start of every iteration (V6)
            beta = beta_check(x, N, c)
            # Choose between MAP and Dykstra's method
            if beta == 1:
                e = e_dykstra
            else:
                e = e_MAP # these are all zeros and do not change

            # Iterate over every halfspace
            for m, (normal, offset) in enumerate(zip(N, c)):

                # Get m - n index using modulo operator, which ensures
                # we get an index between 0 and n (non-negative)
                index = (m - n) % n  # this is essentially just m-n with zeros for m<n
                x_temp = x.copy() # temporary variable (x_m)

                # Update x_m+1
                x = project_onto_half_space(x_temp + e[index], normal, offset)

                # Update e_m with Dykstra's method
                e_dykstra[m] = e_dykstra[index]+ (x_temp - x)

                # Path
                path.append(x.copy())  # Add the updated x to the path

            # Errors
            if plot_errors:
                errors_for_plotting.append(e_dykstra.copy()) # update error metrix

            # Track the squared error
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

        if track_error and plot_errors:
            # Wrap everything up into a tuple
            error_tuple = (squared_errors, stalled_errors, converged_errors)
            # Return additional vector
            return x, path, error_tuple, errors_for_plotting
        elif track_error:
            # Wrap everything up into a tuple
            error_tuple = (squared_errors, stalled_errors, converged_errors)
            # Return additional vector
            return x, path, error_tuple, None
        else:
            # Return finite-time approximation x to projection task and error e
            return x, path, None, None

    except ValueError as e:
        print(f"ValueError occurred: {e}")
        return None, None, None, None  # Return None in case of errors
    except IndexError as e:
        print(f"IndexError occurred: {e}")
        return None, None, None, None