import numpy as np
import matplotlib.pyplot as plt
from dykstra import dykstra_projection
from plotter import plot_half_spaces

def test_dykstra_on_box_and_line() -> None:
    """Tests Dykstra's algorithm on the intersection of a box at the origin
    and a line passing through (2, 0) and (0, 1)."""

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

    # Point to project (outside the intersection)
    # z = np.array([-0.75, 1.25])
    z = np.array([-1.5, 1.5])

    # Project using Dykstra's algorithm
    projection, error = dykstra_projection(z, np.vstack([N_box, N_line]),
                                      np.hstack([c_box, c_line]), 100)
    print(projection, error)

    # Visualize the results
    Nc_pairs = [
        ("Box", "Greys", N_box, c_box),
        ("Line", "Reds", N_line, c_line)
    ]
    plot_half_spaces(Nc_pairs)

    # Plot the original point and its projection
    plt.scatter(z[0], z[1], color='red', marker='o', label='Original Point')
    plt.scatter(projection[0], projection[1], color='green', marker='x',
                label='Projection')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_dykstra_on_box_and_line()