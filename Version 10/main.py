import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dykstra import dykstra_projection
from plotter import plot_half_spaces, plot_path, plot_active_spaces
from gradient import quadprog_solve_qp


def test_with_tracking() -> None:
    """Tests Dykstra's algorithm on the intersection of a box at the origin
    and a line passing through (2, 0) and (0, 1)"""

    # Define the box constraints (half-spaces) (make sure these are floats)
    N_box = np.array([
        [1., 0.],  # Right side of the box: x <= 1
        [-1., 0.], # Left side of the box: x >= -1
        [0., 1.],  # Top side of the box: y <= 1
        [0., -1.]  # Bottom side of the box: y >= -1
    ])
    c_box = np.array([1., 1., 1., 1.])

    # Corner count for rounding
    corner_count = 1

    # Define the line constraints
    # The line equation is y = 1 - x/2
    # Rearranging to get it in the form N*x <= c:
    # x/2 + y <= 1 & x/2 + y >= 1
    N_line = np.array([[1/2, 1], [-1/2, -1]])
    c_line = np.array([1, -1])

    # Point to project (outside the intersection)
    # z = np.array([-0.75, 1.25])
    # z = np.array([-1.5, 1.5])
    # Exact solution = [0, 1]

    # Point very far to the bottom left (stalls)
    z = np.array([-1.75, 1.75])

    # 17 iterations to exit stalling for non-rounded box [-4, 4]
    # after 16 iterations non-rounded box gives approximation of [-0.8  1.4]
    # 47 iterations to exit stalling for non-rounded box [-10, 10]

    # Project using Dykstra's algorithm
    max_iter: int = 15 # number of iterations
    plot_quivers = True # for plotting error quivers
    plot_activity = True
    projection, path, error_tuple, errs_to_plot, active_half_spaces = (
        dykstra_projection(z, np.vstack([N_box, N_line]),
                           np.hstack([c_box, c_line]),
                           max_iter, track_error=True, plot_errors=plot_quivers,
                           plot_active_halfspaces=plot_activity)
    )

    # Compare final result to actual projection (V4)

    # FROM DOCUMENTATION:
    # (see https://scaron.info/blog/quadratic-programming-in-python.html)
    # " The quadratic expression ∥Ax − b∥^2 of the least squares optimization
    #   is written in standard form with P = 2A^TA and q = −2A^Tb "

    # We are solving min_x ∥x − z∥^2 s.t. Gx <= h so set:
    A = np.eye(2)
    b = z.copy()
    # @ command is recommended for 2d matrix multiplication
    P = 2 * A.T @ A
    q = -2 * A.T @ b
    G = np.vstack([N_box, N_line])
    h = np.hstack([c_box, c_line])
    actual_projection = quadprog_solve_qp(P, q, G, h)
    distance = actual_projection - projection

    # Compare to dykstra approximation (V4)
    print(
        f"\nThe finite time projection over {max_iter} iteration(s) is: "
        f"{projection};\nThe distance to the optimal solution is: "
        f"{distance}\nThe squared-error is {np.dot(distance, distance)}")

    # Create infrastructure for two plots (V4)
    fig = plt.figure(figsize=(8, 10))  # Create the figure
    # This makes the first plot larger
    gs = gridspec.GridSpec(3, 1)  # 3 rows, 1 column
    ax1 = fig.add_subplot(gs[:2, :])  # First subplot spans the top two rows
    ax2 = fig.add_subplot(gs[2, :])  # Second subplot occupies the bottom row

    # Visualize the results
    Nc_pairs = [
        (f"'Box'\n(rounded by {corner_count} corner(s))", "Greys", N_box, c_box),
        ("Line", "Greys", N_line, c_line)
    ]

    # Modified everything to accept an axis handle (V4)

    # Plot the half spaces
    plot_half_spaces(Nc_pairs, max_iter, ax1)
    # Plot the path (V8)
    plot_path(path, ax1, errs_to_plot, plot_errors=plot_quivers)

    # Plot the original point and its projection
    ax1.scatter(z[0], z[1], color='green', marker='o', label='Original Point')
    ax1.scatter(projection[0], projection[1], color='green', marker='x',
                label='Projection')
    # Plot optimal solution (V4)
    ax1.scatter(actual_projection[0], actual_projection[1],
                color='red', marker='x', label='Optimal Solution')
    # Set legend for first plot
    ax1.legend()

    # Plot the squared errors
    iterations = np.arange(0, max_iter, 1)
    last_error = error_tuple[0][-1] # get the last error for printing
    error_tuple[0][-1] = None # set this to None so that we can see last error
    ax2.plot(iterations, error_tuple[0], color='red',
             label='Errors', linestyle='-', marker='o')
    # Plot the stalling errors
    ax2.plot(iterations, error_tuple[1], color='#D5B60A',
             label='Stalling', linestyle='-', marker='o')
    # Plot the converged errors
    ax2.plot(iterations, error_tuple[2], color='green',
             label='Converged\n(error under 1e-3)', linestyle='-', marker='o')
    # Plot the last error
    ax2.scatter(iterations[-1], last_error,
                color='#8B0000', marker='o',
                label=f'Final error is {format(last_error, ".2e")}')

    # Set labels, title and legend for second plot
    ax2.set_xlabel('Number of Iterations')  # Add x-axis label
    ax2.set_ylabel('Squared Errors')  # Add y-axis label
    ax2.set_title('Convergence of Squared Errors')
    ax2.grid(True)
    ax2.legend()

    # Show the plot
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.show()

    # Plot active halfspaces
    if plot_activity:
        # print(active_half_spaces[0])
        plot_active_spaces(active_half_spaces, max_iter)

    # Debugging info
    # print(errs_to_plot)
    # print(path)
    # print(len(path))
    # print(len(errs_to_plot[0]))

if __name__ == "__main__":
    test_with_tracking()