"""
This module implements the Method for Alternating Projections for projecting
a point onto the intersection of convex sets (specifically, half-spaces).

Functions:

- project_onto_half_space(point, normal, constant_offset):
    Projects a given point onto a single half-space.

- MAP_projection(z, N, c, max_iter):
    Projects a point 'z' onto the intersection of multiple half-spaces
    defined by the matrix N and vector c.
"""


import numpy as np


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


def MAP_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int) -> tuple:
    """Projects a point 'z' onto the intersection of convex sets H_i (half spaces).
    The convex set parameters (unit normals and constant offsets) are packaged
    into matrix N and vector c respectively, such that:
    N*x <= c -> [n_i^T]*x <= {c_i} yields a set of linear inequalities of the kind
    <x,n_i> <= c_i for all i = rowcount(N).
    Halting is ensured via a finite iteration count max_iter"""
    try:
        # Initialise variables
        x = z.copy()  # create a deep copy of the original point

        # Path (V3)
        path = [z.copy()]  # Initialize the path with the original point

        # Main body of Dykstra's algorithm
        for _ in range(max_iter):
            # Iterate over every half plane
            for m, (normal, offset) in enumerate(zip(N, c)):
                # Update x_m+1
                x = project_onto_half_space(x, normal, offset)

                # Path (V3)
                path.append(x.copy())  # Add the updated x to the path

        # Return finite-time approximation x to projection task and error e
        return x, path

    except ValueError as e:
        print(f"ValueError occurred: {e}")
    except IndexError as e:
        print(f"IndexError occurred: {e}")