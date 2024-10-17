import plotter
import numpy as np
import matplotlib.pyplot as plt


# Square centered at the origin with side length 2
N_square = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]])
c_square = np.array([1, 1, 1, 1, 1, 1])

# Line passing through (2, 0) and (0, 1)
N_line = np.array([[1, 2], [-1, -2]])
c_line = np.array([2, -2])

# Strip
N_strip = np.array([[1, 2], [-0.9, -2]])
c_strip = np.array([1.5, -1])

# Assemble
Nc_pairs = [('Strip', 'Blues', N_strip, c_strip),
            ('Diamond', 'Greys', N_square, c_square),
            ('Line', 'Reds', N_line, c_line)]

# Plot
plotter.plot_half_spaces(Nc_pairs)
plt.show()