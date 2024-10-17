import matplotlib.pyplot as plt
import numpy as np


def plot_path(path: list) -> None:
    """Plots the path followed by Dykstra's algorithm during projection.

    Args:
        path: A list of points representing the intermediate steps
              taken by the algorithm.
    """
    # Extract x and y coordinates from the path
    x_coords = [point[0] for point in path] # we love list comprehensions
    y_coords = [point[1] for point in path]

    # Plot the path
    plt.plot(x_coords, y_coords, marker='.', linestyle='--',
             color='blue', linewidth=0.5, markersize=1,
             label='Projection Path')

    # Add legend
    plt.legend()