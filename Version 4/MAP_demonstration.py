"""This module demonstrates the potential for sub-optimality in the results
of the MAP algorithm. Two sets are considered (rounded boxes), and both
MAP and Dykstra's can be ran"""


import numpy as np
import matplotlib.pyplot as plt
from dykstra import dykstra_projection
from MAP import MAP_projection
from plotter import plot_half_spaces
from path_plotter import plot_path
from edge_rounder import rounded_box_constraints

def test_dykstra_on_box_and_line() -> None:
    """Tests Dykstra's algorithm on the intersection of a box at the origin
    and a line passing through (2, 0) and (0, 1)."""

    # Square centered at (-0.5, 0) origin with side length 2 and rounded edges
    center = (-.5, 0)
    width = 2
    height = 2
    N_set1, c_set1 = rounded_box_constraints(center, width, height)

    # Square centered at (1, 0) origin with side length 3 and rounded edges
    center = (1, 0)
    width = 3
    height = 3
    N_set2, c_set2 = rounded_box_constraints(center, width, height)

    # Point to project (outside the intersection)
    z = np.array([-.2, 1.2])


    # Project using Dykstra's algorithm
    projection, error, path = dykstra_projection(z, np.vstack([N_set1, N_set2]),
                                      np.hstack([c_set1, c_set2]), 2)
    print(projection, error)

    # # Project using MAP algorithm
    # projection, path = MAP_projection(z,
    #                                      np.vstack([N_set1, N_set2]),
    #                                      np.hstack([c_set1, c_set2]),
    #                                      3)
    # print(projection)

    # Visualize the results
    Nc_pairs = [
        ("Set 1", "Blues", N_set1, c_set1),
        ("Set 2", "Reds", N_set2, c_set2)
    ]
    plot_half_spaces(Nc_pairs)

    # Path (V3)
    plot_path(path)  # Plot the path

    # Plot the original point and its projection
    plt.scatter(z[0], z[1], color='green', marker='o', label='Original Point')
    plt.scatter(projection[0], projection[1], color='green', marker='x',
                label='Projection')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_dykstra_on_box_and_line()