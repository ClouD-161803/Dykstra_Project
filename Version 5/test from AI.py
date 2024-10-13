import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from modified_dykstra import dykstra_projection
from plotter import plot_half_spaces
from path_plotter import plot_path
from edge_rounder import rounded_box_constraints
from gradient import quadprog_solve_qp

def test_dykstra_on_rounded_boxes():
    """Tests Dykstra's algorithm on the intersection of two rounded boxes."""

    # Define the first rounded box
    center1 = (-1, 0)
    width1 = 4
    height1 = 4
    corner_segments1 = 10
    N_box1, c_box1 = rounded_box_constraints(center1, width1, height1, corner_segments1)

    # Define the second rounded box (overlapping with the first)
    center2 = (1, 0)
    width2 = 2
    height2 = 2
    corner_segments2 = 2
    N_box2, c_box2 = rounded_box_constraints(center2, width2, height2, corner_segments2)

    # Point to project (on the intersection)
    z = np.array([100, 100])

    # Project using Dykstra's algorithm
    max_iter = 3000
    beta = 0.01
    converged_error = 1e-3 # minimum error required for convergence
    projection, _, path, error_tuple = dykstra_projection(
        z, np.vstack([N_box1, N_box2]), np.hstack([c_box1, c_box2]), max_iter,
        track_error=True, beta=beta, min_error=converged_error
    )

    # Compare final result to actual projection
    A = np.eye(2)
    b = z.copy()
    P = 2 * A.T @ A
    q = -2 * A.T @ b
    G = np.vstack([N_box1, N_box2])
    h = np.hstack([c_box1, c_box2])
    actual_projection = quadprog_solve_qp(P, q, G, h)
    distance = actual_projection - projection

    print(
        f"\nThe finite time projection over {max_iter} iteration(s) is: "
        f"{projection};\nThe distance to the optimal solution is: "
        f"{distance}\nThe squared-error is {np.dot(distance, distance)}"
    )

    # Create subplots
    fig = plt.figure(figsize=(8, 10))  # Create the figure
    # This makes the first plot larger
    gs = gridspec.GridSpec(3, 1)  # 3 rows, 1 column
    ax1 = fig.add_subplot(gs[:2, :])  # First subplot spans the top two rows
    ax2 = fig.add_subplot(gs[2, :])  # Second subplot occupies the bottom row

    # Visualize the results on the first subplot
    Nc_pairs = [
        ("Box 1", "Blues", N_box1, c_box1),
        ("Box 2", "Greens", N_box2, c_box2)
    ]
    plot_half_spaces(Nc_pairs, max_iter, ax=ax1, beta=beta)

    # Plot the path and points on the first subplot
    plot_path(path, ax=ax1)
    ax1.scatter(z[0], z[1], color='red', marker='o', label='Original Point')
    ax1.scatter(projection[0], projection[1], color='purple', marker='x', label='Projection')
    ax1.scatter(actual_projection[0], actual_projection[1], color='black', marker='x', label='Optimal Solution')
    ax1.legend()
    ax1.set_title(f"Modified Dykstra's algorithm (beta = {beta}) executed for {max_iter} iterations")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True)

    # Plot the errors on the second subplot
    iterations = np.arange(0, max_iter, 1)
    last_error = error_tuple[0][-1]
    error_tuple[0][-1] = None
    ax2.plot(iterations, error_tuple[0], color='red', label='Errors', linestyle='-', marker='o')
    ax2.plot(iterations, error_tuple[1], color='yellow', label='Stalling', linestyle='-', marker='o')
    ax2.plot(iterations, error_tuple[2], color='green', label=f'Converged\n(error under {converged_error})', linestyle='-', marker='o')
    ax2.scatter(iterations[-1], last_error, color='darkred', marker='o', label=f'Final error is {format(last_error, ".2e")}')

    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Squared Errors')
    ax2.set_title('Convergence of Squared Errors')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dykstra_on_rounded_boxes()