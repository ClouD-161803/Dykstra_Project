""" DID NOT WORK"""


import numpy as np
import matplotlib.pyplot as plt
from dykstra import dykstra_projection
from plotter import plot_half_spaces
from path_plotter import plot_path

def hexagon_constraints(center, side_length):
    """Generates half-space constraints for a hexagon.

    Args:
        center: (x, y) coordinates of the hexagon's center.
        side_length: Length of each side of the hexagon.

    Returns:
        N: Matrix of normal vectors for the half-spaces.
        c: Vector of constant offsets for the half-spaces.
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # Angles for the six sides

    N = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])

    # Corrected offset calculation
    c = np.array([np.dot(N[i], center) + side_length / 2 for i in range(len(angles))])

    return N, c

def test_dykstra_on_hexagons():
    """Tests Dykstra's algorithm on the intersection of two hexagons."""

    # Define the first hexagon
    center1 = (0, 0)
    side_length1 = 1.5
    N_hex1, c_hex1 = hexagon_constraints(center1, side_length1)

    # Define the second hexagon (slightly offset and rotated)
    center2 = (0.5, 0.5)
    side_length2 = 1.2
    N_hex2, c_hex2 = hexagon_constraints(center2, side_length2)

    # Point to project (outside the intersection)
    z = np.array([2, -1])

    # Project using Dykstra's algorithm
    projection, error, path = dykstra_projection(z,
                                                 np.vstack([N_hex1, N_hex2]),
                                                 np.hstack([c_hex1, c_hex2]),
                                                 100)
    print(projection, error)

    # Visualize the results
    Nc_pairs = [
        ("Hexagon 1", "Blues", N_hex1, c_hex1),
        ("Hexagon 2", "Greens", N_hex2, c_hex2)
    ]
    plot_half_spaces(Nc_pairs)

    # Plot the path and points
    plot_path(path)
    plt.scatter(z[0], z[1], color='red', marker='o', label='Original Point')
    plt.scatter(projection[0], projection[1], color='purple', marker='x', label='Projection')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_dykstra_on_hexagons()