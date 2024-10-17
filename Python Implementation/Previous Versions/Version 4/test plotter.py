from edge_rounder import rounded_box_constraints
import plotter
import numpy as np
import matplotlib.pyplot as plt


# # Square centered at the origin with side length 2 and rounded edges
# center = (0, 0)
# width = 2
# height = 2
# N_square, c_square = rounded_box_constraints(center, width, height)
#
# # Line passing through (2, 0) and (0, 1)
# N_line = np.array([[1, 2], [-1, -2]])
# c_line = np.array([2, -2])
#
# # Strip
# N_strip = np.array([[1, 2], [-0.9, -2]])
# c_strip = np.array([1.5, -1])

# # Assemble
# Nc_pairs = [('Strip', 'Blues', N_strip, c_strip),
#             ('Diamond', 'Greys', N_square, c_square),
#             ('Line', 'Reds', N_line, c_line)]

# # Assemble
# Nc_pairs = [('Box', 'Greys', N_square, c_square)]
# # print(N_square, c_square)


# # Hexagon
# def hexagon_constraints(center, side_length):
#     """Generates half-space constraints for a hexagon.
#
#     Args:
#         center: (x, y) coordinates of the hexagon's center.
#         side_length: Length of each side of the hexagon.
#
#     Returns:
#         N: Matrix of normal vectors for the half-spaces.
#         c: Vector of constant offsets for the half-spaces.
#     """
#     angles = np.linspace(0, 2 * np.pi, 4)[:-1]  # Angles for the six sides
#
#     N = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
#
#     # Corrected offset calculation
#     c = np.array([np.dot(N[i], center) + side_length / 2 for i in range(len(angles))])
#
#     return N, c
#
# # get hexagon inequalities
# N_hex1, c_hex1 = hexagon_constraints((0,0),1)
#
# # Assemble
# Nc_pairs = [('Hex', 'Greys', N_hex1, c_hex1)]
# # print(N_square, c_square)

# Square centered at (-0.5, 0) origin with side length 2 and rounded edges
center = (-.5, 0)
width = 2
height = 2
N_square1, c_square1 = rounded_box_constraints(center, width, height, 2)

# Square centered at (1, 0) origin with side length 3 and rounded edges
center = (1, 0)
width = 3
height = 3
N_square2, c_square2 = rounded_box_constraints(center, width, height, 100)

# Assemble
Nc_pairs = [('Set1', 'Blues', N_square1, c_square1),
            ('Set2', 'Reds', N_square2, c_square2)]
# print(N_square, c_square)

# Plot
plotter.plot_half_spaces(Nc_pairs)
plt.show()

