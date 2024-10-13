"""This module contains a function to round the corners of a box.
Explanation:

    1. Straight Sides:
       * Defines constraints for the four straight sides of the box using
        outward-facing normal vectors.
       * Calculates the constant offsets `c_straight` based on the position of
        the center and dimensions of the box.

    2. Rounded Corners:
       * Iterates over each of the four corners of the box.
       * For each corner, it further iterates `corner_segments` times to
       define the linear segments approximating the rounded corner.
       * Calculates the angle `angle` for each segment, evenly spaced within
        a quarter-circle.
       * Computes the normal vector `normal` for the segment based on the
        angle and the corner's quadrant.
       * Determines the radius of the rounded corner as the minimum of half
        the width and half the height of the box.
       * Calculates the offset `offset` for the segment based on the position
        of the center, normal vector, and radius.
       * Appends the `normal` and `offset` to lists
        `N_rounded` and `c_rounded`, respectively.

    3. Combination:
       * Converts the lists `N_rounded` and `c_rounded` to NumPy arrays.
       * Vertically stacks the straight side constraints
        (`N_straight`, `c_straight`) and the rounded corner constraints
        (`N_rounded`, `c_rounded`) to form the final `N` and `c`.

    The resulting `N` and `c` can be used in optimization or projection algorithms
    that require half-space constraints to represent the rounded box.
"""


import numpy as np


def rounded_box_constraints(center, width, height, corner_segments=5):
    """
    Generates half-space constraints (N, c) that define a box with rounded corners.

    This function approximates the rounded corners of a box using multiple
    linear segments (half-spaces).
    The more segments used, the smoother the corners appear.

    Args:
        center: A tuple (x, y) representing the coordinates of the box's center.
        width: The width of the box.
        height: The height of the box.
        corner_segments: The number of linear segments used to approximate
        each rounded corner (default is 5).

    Returns:
        A tuple (N, c) where:

        * N: A NumPy array where each row represents the outward-facing normal
            vector of a half-space constraint.
        * c: A NumPy array where each element represents the constant offset
            of a corresponding half-space constraint.
    """

    half_width = width / 2
    half_height = height / 2

    # Constraints for the straight sides (outward-facing normals)
    N_straight = np.array([
        [-1, 0],  # Left side
        [1, 0],   # Right side
        [0, -1],  # Bottom side
        [0, 1]    # Top side
    ])
    c_straight = np.array([
        -center[0] + half_width,
        center[0] + half_width,
        -center[1] + half_height,
        center[1] + half_height
    ])

    # Constraints for the rounded corners
    N_rounded = []
    c_rounded = []
    for corner in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:  # Four corners
        for i in range(corner_segments):
            angle = i / corner_segments * np.pi / 2  # Adjusted angle calculation
            normal = np.array([corner[0] * np.cos(angle), corner[1] * np.sin(angle)])
            # Radius of the rounded corner is the minimum of half_width and half_height
            radius = min(half_width, half_height)
            offset = np.dot(normal, center) + radius
            N_rounded.append(normal)
            c_rounded.append(offset)

    N_rounded = np.array(N_rounded)
    c_rounded = np.array(c_rounded)

    # Combine all constraints
    N = np.vstack([N_straight, N_rounded])
    c = np.hstack([c_straight, c_rounded])

    return N, c