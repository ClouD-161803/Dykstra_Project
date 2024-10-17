import numpy as np


def normalise(normal: np.ndarray, offset: np.ndarray) -> tuple:
    """
    Normalises half space normal and constant offset.

    Args:
        normal: Normal vector of the half space.
        offset: Constant offset of the half space.

    Returns:
        tuple: Unit normal vector and normalised offset.
    """
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
    """
    Checks if a point lies within a single half space.

    Args:
        point: Point to check.
        unit_normal: *Unit* normal vector of the half space.
        constant_offset: *Normalised* offset of the half space.

    Returns:
        bool: True if point is within the half space, else False.
    """
    # Find dot product
    dp = np.dot(point, unit_normal)
    # Return boolean
    return dp <= constant_offset


def project_onto_half_space(point: np.ndarray, normal: np.ndarray,
                            offset: np.ndarray) -> np.ndarray:
    """Projects a point onto a single half space 'H_i'.
    A half space is defined by H_i := {x | <x,n_i> <= c_i}, with boundary
    B_i := {x | <x,n_i> = c_i}, and the projection of a point z onto H_i is
    given by: P_H_i(z) = z - (<z,n_i> - c_i)*n_i if z is outside H_i, where:
    point = z, unit_normal  = n_i, constant_offset = c_i

    Args:
        point: Point to project.
        normal: Normal vector of the half space.
        offset: Constant offset of the half space.

    Returns:
        np.ndarray: Projected point.
    """

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


def delete_inactive_half_spaces(z: np.ndarray, N: np.ndarray, c: np.ndarray)\
        -> tuple:
    """
    Deletes inactive half spaces.

    Args:
        z: Point used to check for inactive half spaces.
        N: Matrix of normal vectors.
        c: Vector of constant offsets.

    Returns:
        tuple: Updated matrix of normal vectors and vector of constant offsets.
    """
    # Initialise empty removal mask
    indices_to_remove = np.zeros_like(c, dtype=bool)

    # Update mask
    for m, (normal, offset) in enumerate(zip(N, c)):
        # Normalise
        unit_normal, constant_offset = normalise(normal, offset)
        # Check if half space is inactive
        if is_in_half_space(z, unit_normal, constant_offset):
            indices_to_remove[m] = True

    # Remove inactive halfspaces using mask
    new_N = N[~indices_to_remove]
    new_c = c[~indices_to_remove]

    return new_N, new_c


def beta_check(point: np.ndarray, N: np.ndarray, c: np.ndarray):
    """
    Selects a value of beta based on whether the passed point lies
    within the intersection of half-spaces.

    Args:
        point: Point to check.
        N: Matrix of normal vectors.
        c: Vector of constant offsets.

    Returns:
        int: 1 if the point is within the intersection, else 0.
    """

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