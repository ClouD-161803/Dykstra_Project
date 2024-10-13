import numpy as np


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
    # The rounding prevents numerical errors
    return round(np.dot(point, unit_normal), 10) <= round(constant_offset, 10)



# Choose Beta at the start of every iteration (V6)
# def beta_check(point, N, c):
#     beta = 1
#     not_in_intersection = False  # initialise boolean
#     for _, (normal, offset) in enumerate(zip(N, c)):
#         # Normalise
#         unit_normal, constant_offset = normalise(normal, offset)
#         # Check if we are in the half space to update beta
#         if not is_in_half_space(point, unit_normal, constant_offset):
#             not_in_intersection = True
#     # Choose Beta = 0 if not in intersection (MAP)
#     if not_in_intersection:
#         beta = 0
#     return beta

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


# Test

# Define the box constraints (half-spaces)
N_box = np.array([
    [1, 0],  # Right side of the box: x <= 1
    [-1, 0], # Left side of the box: x >= -1
    [0, 1],  # Top side of the box: y <= 1
    [0, -1]  # Bottom side of the box: y >= -1
])
c_box = np.array([1, 1, 1, 1])

# Define the line constraints
# The line equation is y = 1 - x/2
# Rearranging to get it in the form N*x <= c:
# x/2 + y <= 1 & x/2 + y >= 1
N_line = np.array([[1/2, 1], [-1/2, -1]])
c_line = np.array([1, -1])

# Point very far to the bottom left (stalls)
z = np.array([0.5, 0.76])

N = np.vstack([N_box, N_line])
c = np.hstack([c_box, c_line])

beta = beta_check(z, N, c)
print(beta)