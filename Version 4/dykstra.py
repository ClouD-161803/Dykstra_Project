"""
This module implements Dykstra's algorithm for projecting a point onto the
intersection of convex sets (specifically, half-spaces).

Functions:

- project_onto_half_space(point, normal, constant_offset):
    Projects a given point onto a single half-space.

- dykstra_projection(z, N, c, max_iter):
    Projects a point 'z' onto the intersection of multiple half-spaces
    defined by the matrix N and vector c.

Added the functionality to keep track of the error (V4)
"""


import numpy as np
from gradient import quadprog_solve_qp # needed to track error (V4)


def project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                            offset: np.ndarray) -> np.ndarray:
    """Projects a point onto a single half space 'H_i'.
    A half space is defined by H_i := {x | <x,n_i> <= c_i}, with boundary
    B_i := {x | <x,n_i> = c_i}, and the projection of a point z onto H_i is
    given by: P_H_i(z) = z - (<z,n_i> - c_i)*n_i if z is outside H_i, where:
    point = z, unit_normal  = n_i, constant_offset = c_i"""

    # Normalise normal vector
    norm = np.linalg.norm(normal)
    if norm == 0: # me and my homies hate division by 0
        raise ValueError("Warning: Zero-norm normal vector encountered.")
    else:
        unit_normal = normal / norm
        # Normalise offset
        constant_offset = offset / norm

    # Check if point is already in the half space
    if np.dot(point, unit_normal) <= constant_offset:
        return point
    # Project point onto the half space's boundary
    else:
        boundary_projection = (point - (np.dot(point, unit_normal)
                                        - constant_offset) * unit_normal)
        return boundary_projection


def dykstra_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int, track_error: bool=False) -> tuple:
    """Projects a point 'z' onto the intersection of convex sets H_i (half spaces).
    The convex set parameters (unit normals and constant offsets) are packaged
    into matrix N and vector c respectively, such that:
    N*x <= c -> [n_i^T]*x <= {c_i} yields a set of linear inequalities of the kind
    <x,n_i> <= c_i for all i = rowcount(N).
    Halting is ensured via a finite iteration count max_iter

    Added a boolean track_error to select whether to track the squared error
    to the optimal solution at each iteration. If it is true, the function will
    output an additional vector of squared errors (V4)"""
    try:
        # Initialise variables
        n = N.shape[0]  # Number of half-spaces
        x = z.copy()  # create a deep copy of the original point
        errors = np.zeros_like(z) # individual error vectors
        e = [errors] * n  # list of a number of error vectors equal to n

        # Path
        path = [z.copy()]  # Initialize the path with the original point

        # Optimal solution (V4)
        if track_error:
            # FROM DOCUMENTATION:
            # (see https://scaron.info/blog/quadratic-programming-in-python.html)
            # " The quadratic expression ∥Ax − b∥^2 of the least squares optimisation
            #   is written in standard form with P = 2A^TA and q = −2A^Tb "
            # We are solving min_x ∥x − z∥^2 s.t. Gx <= h so set:
            A = np.eye(2)
            b = z.copy()
            # @ command is recommended for 2d matrix multiplication
            P = 2 * A.T @ A
            q = -2 * A.T @ b
            G = N
            h = c
            # Find projection
            actual_projection = quadprog_solve_qp(P, q, G, h)
            # Initialise errors vector
            squared_errors = np.zeros(max_iter)

        # Main body of Dykstra's algorithm
        for i in range(max_iter):
            # Iterate over every half plane
            for m, (normal, offset) in enumerate(zip(N, c)):
                # Get m - n index using modulo operator, which ensures
                # we get an index between 0 and n (non-negative)
                index = (m - n) % n  # this is essentially just m-n with zeros for m<n
                x_temp = x.copy() # temporary variable (x_m)
                # Update x_m+1
                x = project_onto_half_space(x_temp + e[index], normal, offset)
                # Update e_m
                e[m] = x_temp + e[index] - x

                # Path
                path.append(x.copy())  # Add the updated x to the path

            # Track the squared error (V4)
            if track_error:
                distance = actual_projection - x
                squared_errors[i] = np.dot(distance, distance)

        if track_error:
            # Return additional vector (V4)
            return x, e, path, squared_errors
        else:
            # Return finite-time approximation x to projection task and error e
            return x, e, path

    except ValueError as e:
        print(f"ValueError occurred: {e}")
    except IndexError as e:
        print(f"IndexError occurred: {e}")