import numpy as np
import matplotlib.pyplot as plt

def project_onto_convex_set(point, convex_set_projection_function):
    """Projects a point onto a single convex set given the projection function"""
    return convex_set_projection_function(point)

def dykstra_projection(z, projection_onto_A, projection_onto_B, max_iter):
    """Projects a point 'z' onto the intersection of convex sets A and B.
    Halting is ensured via a finite iteration count"""
    # Initialise variables
    b = z.copy() # need a deep copy
    p = np.zeros_like(z) # zeros_like ensures dimensions agree
    q = np.zeros_like(z)
    # Main body of Dykstra's algorithm
    for _ in range(max_iter):
        # Update main and auxiliary sequences
        a = project_onto_convex_set(b + p, projection_onto_A)
        p = b + p - a
        b = project_onto_convex_set(a + q, projection_onto_B)
        q = a + q - b
    # Return finite-time approximation to projection task
    return a

# Example: Projection onto the intersection of a box and a line
def project_onto_box(point, center, side_length):
    """Projects a point onto a square box given its center and sidelength,
    this is obtained using the np.clip() function"""
    half_side = side_length / 2
    lower_bounds = center - half_side
    upper_bounds = center + half_side
    return np.clip(point, lower_bounds, upper_bounds)

def project_onto_line(point, line_point, line_direction):
    """Projects a point onto a line given a point on the line and its direction.
    This is obtained through the scheme p = lp + [a*a^T/(a^T*a)]*b where
    point = b; line_point = lp, line_direction = a (see linear algebra notes)"""
    ap = point - line_point # vector from b to lp
    # Obtain projection and return it
    p = (line_point + np.dot(ap, line_direction) /
            np.dot(line_direction, line_direction) * line_direction)
    return p

# Define variables for the box and the line
box_center = np.array([0, 0])
box_side_length = 2
line_point = np.array([2, 0]) # a point on the line
line_direction = np.array([-2, 1])

# The point to be projected
z = np.array([-1, 1.2])

# Perform Dykstra's projection using lambda functions to wrap inner projections
projected_point = dykstra_projection(z,
    lambda point: project_onto_box(point, box_center, box_side_length),
    lambda point: project_onto_line(point, line_point, line_direction),
    max_iter=100)

# Visualization
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 3, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Plotting the box
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        projected_point_box = project_onto_box(point, box_center, box_side_length)
        Z[i, j] = np.linalg.norm(point - projected_point_box)
plt.contour(X, Y, Z, levels=[0], colors='r')

# Plotting the line
line_x = [line_point[0] - 3*line_direction[0], line_point[0] + 3*line_direction[0]]
line_y = [line_point[1] - 3*line_direction[1], line_point[1] + 3*line_direction[1]]
plt.plot(line_x, line_y, 'b-', label='Line')

# Plotting the original point and the projected point
plt.plot(z[0], z[1], 'ko', label='Original Point z')
plt.plot(projected_point[0], projected_point[1], 'ro', label='Projected Point')

# Show legend and grid
plt.legend()
plt.grid(True)
plt.show()
