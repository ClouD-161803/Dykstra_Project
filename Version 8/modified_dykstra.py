"""
This module implements Dykstra's algorithm for projecting a point onto the
intersection of convex sets (specifically, half-spaces).

Functions:

- project_onto_half_space(point, normal, constant_offset):
    Projects a given point onto a single half-space.

- dykstra_projection(z, N, c, max_iter):
    Projects a point 'z' onto the intersection of multiple half-spaces
    defined by the matrix N and vector c.

Added the functionality to keep track of the error, check for stalling and
convergence

Modified Dykstra's method, introducing a new parameter Beta

Modified the structure of the algorithm so that the value of Beta can be
changed at every iteration step (V6)

Structure is now able to output a matrix to plot the errors at each iteration (V7)
"""


import numpy as np
from gradient import quadprog_solve_qp # needed to track error (V4)


def normalise(normal: np.ndarray, offset: np.ndarray) -> tuple:
    """Normalises half space normal and constant offset"""
    # Obtain normal
    norm = np.linalg.norm(normal)
    if norm == 0:  # me and my homies hate division by 0
        raise ValueError("Warning: Zero-norm normal vector encountered.")
    else:
        # Normalise normal
        unit_normal = normal / norm
        # Normalise offset
        constant_offset = offset / norm

    return unit_normal, constant_offset


def is_in_half_space(point: np.ndarray, unit_normal: np.ndarray,
                            constant_offset: np.ndarray) -> bool:
    """Checks if a point lies within a single half space H_i
    NOTE: arguments must be normalised
    Returns True if point is withing the halfspace"""
    # rounded_offset = np.ceil(constant_offset * 10**decimal_places) / 10**decimal_places
    # this rounds a number up to the nearest nth decimal place
    dp = np.dot(point, unit_normal)
    # The rounding prevents numerical errors
    decimal_places = 10
    rounded_offset = np.round(constant_offset, decimal_places)
    rounded_dp = np.round(dp, 10)
    # if rounded_dp <= rounded_offset:
    #     decimal_places = 20
    #     rounded_offset = np.round(constant_offset, decimal_places)

    # Return boolean
    return rounded_dp <= rounded_offset


def beta_check(point: np.ndarray, N: np.ndarray, c: np.ndarray):
    """This functions selects a value of beta based on whether the passed point
    lies within the intersection of the halfspaces. If it does, it returns 1,
    choosing dykstra's method; otherwise, it returns 0, choosing MAP"""

    # Initialise beta
    beta = 1
    not_in_intersection = False  # initialise boolean
    # I encountered numerical error problems
    rounded_point = np.around(point, decimals=10) # round to 10 decimal places
    for _, (normal, offset) in enumerate(zip(N, c)):
        # Normalise
        unit_normal, constant_offset = normalise(normal, offset)
        # Check if we are in the half space to update beta
        if not is_in_half_space(rounded_point, unit_normal, constant_offset):
            not_in_intersection = True
    # Choose Beta = 0 if not in intersection (MAP)
    if not_in_intersection:
        beta = 0
    return beta


def project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                            offset: np.ndarray) -> np.ndarray:
    """Projects a point onto a single half space 'H_i'.
    A half space is defined by H_i := {x | <x,n_i> <= c_i}, with boundary
    B_i := {x | <x,n_i> = c_i}, and the projection of a point z onto H_i is
    given by: P_H_i(z) = z - (<z,n_i> - c_i)*n_i if z is outside H_i, where:
    point = z, unit_normal  = n_i, constant_offset = c_i"""

    # Normalise
    unit_normal, constant_offset = normalise(normal, offset)

    # Check if point is already in the half space
    if is_in_half_space(point, unit_normal, constant_offset):
        return point
    # Project point onto the half space's boundary
    else:
        boundary_projection = (point - (np.dot(point, unit_normal)
                                        - constant_offset) * unit_normal)
        return boundary_projection


def dykstra_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int, track_error: bool=False,
                       min_error: int=1e-3, dimensions: int=2,
                       plot_errors: bool = False) -> tuple:
    """Projects a point 'z' onto the intersection of convex sets H_i (half spaces).
    The convex set parameters (unit normals and constant offsets) are packaged
    into matrix N and vector c respectively, such that:
    N*x <= c -> [n_i^T]*x <= {c_i} yields a set of linear inequalities of the kind
    <x,n_i> <= c_i for all i = rowcount(N).
    Halting is ensured via a finite iteration count max_iter

    Added a boolean track_error to select whether to track the squared error
    to the optimal solution at each iteration. If it is true, the function will
    output an additional vector of squared errors

    Added functionality to check whether the algorithm is stalling via a vector
    of stalled errors, and similarly whether we have converged with a vector of
    converged errors

    Now algorithm selects between MAP and Dykstra's method at each iteration:
    if the previous projection lies withing the intersection of all halfspaces,
    we pick dykstra's method. Otherwise, we pick MAP, to exit stalling

    Generalised the projection algorithm to any number of dimensions (V7)

    Added code for plotting the errors at each iteration (V7)"""

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
        return None, None, None  # Return None in case of errors
    except IndexError as e:
        print(f"IndexError occurred: {e}")
        return None, None, None