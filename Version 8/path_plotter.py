import numpy as np


def plot_path(path: list, ax, errors_for_plotting: np.ndarray=None,
              plot_errors: bool=False) -> None:
    """Plots the path followed by Dykstra's algorithm during projection.
    Can also plot the errors at each iteration (V7)

    Args:
        path: A list of points representing the intermediate steps
              taken by the algorithm.
        ax: Axes handle for plotting
        errors_for_plotting: array containing error vectors
        plot_errors: control whether we plot the error vectors
    """
    # Extract x and y coordinates from the path
    x_coords = [point[0] for point in path]
    y_coords = [point[1] for point in path]

    # Plot the path
    ax.plot(x_coords, y_coords, marker='.', linestyle='--',
             color='blue', linewidth=0.5, markersize=1,
             label='Projection Path')

    # Plot the errors - this is quite complex but follows dykstra's structure
    if plot_errors:
        n = len(errors_for_plotting[0]) # number of halfspaces
        m = 0 # running index (not sure how else to do this)
        max_iter = len(errors_for_plotting) - 1 # algorithm iterations
        iteration = 1 # external for loop iterations
        for errors in errors_for_plotting:
            for error in errors:
                index = (m - n) % (iteration * n) + n
                if index < max_iter * n:
                    print(f"n: {n} | m: {m} | index: {index} | iteration: {iteration}")
                    print(index, (x_coords[index], y_coords[index]), error)
                    ax.quiver(x_coords[index], y_coords[index], error[0], error[1],
                              angles='xy', scale_units='xy', scale=1, alpha=0.3)
                    m += 1
            iteration += 1

    # Add legend
    ax.legend()