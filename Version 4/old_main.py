import numpy as np
import matplotlib.pyplot as plt
from dykstra import dykstra_projection
from plotter import plot_half_spaces
from path_plotter import plot_path
from edge_rounder import rounded_box_constraints
from gradient import quadprog_solve_qp

def test_dykstra_on_box_and_line(max_iter: int = 3) -> None:
    """Tests Dykstra's algorithm on the intersection of a box at the origin
    and a line passing through (2, 0) and (0, 1).
    Added comparison to optimal solution (using quadprog) (V4)"""

    # Comment this in to test on non-rounded box
    # Define the box constraints (half-spaces)
    # N_box = np.array([
    #     [1, 0],  # Right side of the box: x <= 1
    #     [-1, 0], # Left side of the box: x >= -1
    #     [0, 1],  # Top side of the box: y <= 1
    #     [0, -1]  # Bottom side of the box: y >= -1
    # ])
    # c_box = np.array([1, 1, 1, 1])

    # Box centered at the origin with side length 2 and rounded edges
    center = (0, 0)
    width = 2
    height = 2
    N_box, c_box = rounded_box_constraints(center, width, height)

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
    z = np.array([-1.5, 1.5])

    # 17 iterations to exit stalling for non-rounded box [-4, 4]
    # after 16 iterations non-rounded box gives approximation of [-0.8  1.4]
    # 47 iterations to exit stalling for non-rounded box [-10, 10]

    # Project using Dykstra's algorithm
    projection, _, path = (
        dykstra_projection(z, np.vstack([N_box, N_line]),
                           np.hstack([c_box, c_line]),
                           max_iter))

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

    # Visualize the results
    Nc_pairs = [
        ("'Box'", "Greys", N_box, c_box),
        ("Line", "Greys", N_line, c_line)
    ]
    plot_half_spaces(Nc_pairs)

    # Plot the path
    plot_path(path)

    # Plot the original point and its projection
    plt.scatter(z[0], z[1], color='green', marker='o', label='Original Point')
    plt.scatter(projection[0], projection[1], color='green', marker='x',
                label='Projection')
    # Plot optimal solution (V4)
    plt.scatter(actual_projection[0], actual_projection[1],
                color='red', marker='x', label='Optimal Solution')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_dykstra_on_box_and_line()