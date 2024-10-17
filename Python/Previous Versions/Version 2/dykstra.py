import numpy as np

def project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                            constant_offset: np.ndarray) -> np.ndarray:
    """Projects a point onto a single half space 'H_i'.
    A half space is defined by H_i := {x | <x,n_i> <= c_i}, with boundary
    B_i := {x | <x,n_i> = c_i}, and the projection of a point z onto H_i is
    given by: P_H_i(z) = z - (<z,n_i> - c_i)*n_i if z is outside H_i, where:
    point = z, unit_normal  = n_i, constant_offset = c_i"""

    # Normalise normal vector
    norm = np.linalg.norm(normal)
    if norm == 0:
        unit_normal = normal
    else:
        unit_normal = normal / norm

    # Check if point is already in the half space
    if np.dot(point, unit_normal) <= constant_offset:
        return point
    # Project point onto the half space's boundary
    else:
        boundary_projection = (point - (np.dot(point, unit_normal)
                                        - constant_offset) * unit_normal)
        return boundary_projection

def dykstra_projection(z: np.ndarray, N: np.ndarray, c: np.ndarray,
                       max_iter: int) -> tuple:
    """Projects a point 'z' onto the intersection of convex sets H_i (half spaces).
    The convex set parameters (unit normals and constant offsets) are packaged
    into matrix N and vector c respectively, such that:
    N*x <= c -> [n_i^T]*x <= {c_i} yields a set of linear inequalities of the kind
    <x,n_i> <= c_i for all i = rowcount(N).
    Halting is ensured via a finite iteration count max_iter"""

    # Initialise variables
    n = N.shape[0]  # Number of half-spaces
    x = z.copy()  # create a deep copy of the original point
    errors = np.zeros_like(z) # individual error vectors
    e = [errors] * n  # list of a number of error vectors equal to n

    # Main body of Dykstra's algorithm
    for _ in range(max_iter):
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
    # Return finite-time approximation x to projection task and error e
    return x, e[-1] # return projection and last error