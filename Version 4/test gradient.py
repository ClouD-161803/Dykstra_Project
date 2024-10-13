import numpy as np
from gradient import quadprog_solve_qp


# Example
M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)
q = -np.dot(M.T, np.array([3., 2., 3.]))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.]).reshape((3,))

# # Define the box constraints (half-spaces)
# N_box = np.array([
#     [1., 0.],  # Right side of the box: x <= 1
#     [-1., 0.],  # Left side of the box: x >= -1
#     [0., 1.],  # Top side of the box: y <= 1
#     [0., -1.]  # Bottom side of the box: y >= -1
# ])
# c_box = np.array([1., 1., 1., 1.])
#
# # Point very far to the bottom left (stalls)
# z = np.array([-0.75, 1.25])
#
# # # Box centered at the origin with side length 2 and rounded edges
# # center = (0, 0)
# # width = 2
# # height = 2
# # # Change corner_segments to 1 for non-rounded box
# corner_count = 1
# # N_box, c_box = rounded_box_constraints(center, width, height,
# #                                        corner_count)
#
# # Define the line constraints
# # The line equation is y = 1 - x/2
# # Rearranging to get it in the form N*x <= c:
# # x/2 + y <= 1 & x/2 + y >= 1
# N_line = np.array([[1 / 2, 1], [-1 / 2, -1]])
# c_line = np.array([1, -1])

# # We are solving min_x ∥x − z∥^2 s.t. Gx <= h so set:
# A = np.eye(2)
# b = z.copy()
# # @ command is recommended for 2d matrix multiplication
# P = 2 * A.T @ A
# q = -2 * A.T @ b
# G = np.vstack([N_box, N_line])
# h = np.hstack([c_box, c_line])
# # actual_projection = quadprog_solve_qp(P, q, G, h)
# # distance = actual_projection - projection

print(quadprog_solve_qp(P, q, G, h))