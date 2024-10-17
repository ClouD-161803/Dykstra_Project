from stalling_destroyer import delete_inactive_half_spaces
import numpy as np


# Define the box constraints (half-spaces) (make sure these are floats)
N_box = np.array([
    [1., 0.],  # Right side of the box: x <= 1
    [-1., 0.], # Left side of the box: x >= -1
    [0., 1.],  # Top side of the box: y <= 1
    [0., -1.]  # Bottom side of the box: y >= -1
])
c_box = np.array([1., 1., 1., 1.])

# Define the line constraints
# The line equation is y = 1 - x/2
# Rearranging to get it in the form N*x <= c:
# x/2 + y <= 1 & x/2 + y >= 1
N_line = np.array([[1/2, 1], [-1/2, -1]])
c_line = np.array([1, -1])

# Point very far to the bottom left (stalls)
z = np.array([-1.75, 1.75])

N, c = np.vstack([N_box, N_line]), np.hstack([c_box, c_line])
print(N, c)

N, c = delete_inactive_half_spaces(z, np.vstack([N_box, N_line]),
                           np.hstack([c_box, c_line]))

print(N, c)