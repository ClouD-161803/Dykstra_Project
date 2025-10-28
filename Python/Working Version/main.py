import numpy as np
from convex_projection_solver import (DykstraProjectionSolver,
                                       DykstraMapHybridSolver,
                                       DykstraStallDetectionSolver)
from projection_visualizer import ProjectionVisualizer
from gradient import quadprog_solve_qp
from edge_rounder import rounded_box_constraints




def run_with_tracking() -> None:
    """Tests Dykstra's algorithm on the intersection of a box at the origin
    and a line passing through (2, 0) and (0, 1)"""

    # # * Without rounding
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

    # # * With Rounding
    # # Box centered at the origin with side length 2 and rounded edges
    # center = (0, 0)
    # width = 2
    # height = 2
    # N_box, c_box = rounded_box_constraints(center, width, height,
    #                                        corner_count)

    # Define the line constraints
    # The line equation is y = 1 - x/2
    # Rearranging to get it in the form N*x <= c:
    # x/2 + y <= 1 & x/2 + y >= 1
    N_line = np.array([[1/2, 1], [-1/2, -1]])
    c_line = np.array([1, -1])


    # Point to project and x-y range (uncomment wanted example)

    # Simple top left - stalling - y y y
    z = np.array([-4., 1.4])
    x_range = [-2.5, 0.5]
    y_range = [0.5, 2.25]
    delete_half_spaces = True

    # # Simple top left - no stalling - y y y
    # z = np.array([-0.75, 1.3])
    # x_range = [-1.8, 0.5]
    # y_range = [0.5, 2.]
    # delete_half_spaces = True

    # # Intersection - no stalling - y y n
    # z = np.array([0.5, 1.75])
    # x_range = [-2, 2]
    # y_range = [-2, 2]
    # delete_half_spaces = True

    # # Very far to the top left - y n y
    # z = np.array([-10, 5])
    # x_range = [-10, 0.5]
    # y_range = [0.5, 6]
    # delete_half_spaces = True

    # # Very far to bottom left - n y y
    # z = np.array([-5, -5])
    # x_range = [-6, 0.5]
    # y_range = [-6, 4]
    # delete_half_spaces = False

    # # Very far to the top right y y n
    # z = np.array([3.5, 3.5])
    # x_range = [-1, 4]
    # y_range = [-1., 4]
    # delete_half_spaces = True

    # # Very far to the bottom right - y n y
    # z = np.array([10, -5])
    # x_range = [0, 11]
    # y_range = [-6, 1]
    # delete_half_spaces = True


    # Project using solver
    max_iter: int = 50 # number of iterations
    plot_quivers: bool = False # for plotting error quivers
    plot_activity: bool = True # for plotting halfspace activity

    # Project onto box, then line
    A: np.ndarray = np.vstack([N_box, N_line])
    c: np.ndarray = np.hstack([c_box, c_line])

    # Project onto line, then box
    # A: np.ndarray = np.vstack([N_line, N_box])
    # c: np.ndarray = np.hstack([c_line, c_box])
    
    # Select Solver
    # Standard Dykstra's Algorithm
    # solver = DykstraProjectionSolver(
    #     z, A, c, max_iter,
    #     track_error=True,
    #     plot_errors=plot_quivers,
    #     plot_active_halfspaces=plot_activity,
    #     delete_spaces=delete_half_spaces
    # )
    
    # Hybrid MAP-Dykstra Algorithm
    # solver = DykstraMapHybridSolver(
    #     z, A, c, max_iter,
    #     track_error=True,
    #     plot_errors=plot_quivers,
    #     plot_active_halfspaces=plot_activity,
    #     delete_spaces=delete_half_spaces
    # )
    
    # Dykstra with Stalling Detection
    solver = DykstraStallDetectionSolver(
        z, A, c, max_iter,
        track_error=True,
        plot_errors=plot_quivers,
        plot_active_halfspaces=plot_activity,
        delete_spaces=delete_half_spaces
    )
    
    result = solver.solve()

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
    distance = actual_projection - result.projection

    # Compare to dykstra approximation (V4)
    print(f"\n"
        f"\nThe finite time projection over {max_iter} iteration(s) is: "
        f"{result.projection};\nThe distance to the optimal solution is: "
        f"{distance}\nThe squared-error is {np.dot(distance, distance)}\n")

    # Visualize the results
    Nc_pairs = [
        (f"'Box'\n(rounded by {corner_count} corner(s))" if corner_count > 1 else "Box", "Greys", N_box, c_box),
        ("Line", "Greys", N_line, c_line)
    ]

    # Create visualizer and display results
    visualizer = ProjectionVisualizer(result, Nc_pairs, max_iter, x_range, y_range)
    visualizer.visualize(plot_original_point=z, plot_optimal_point=actual_projection)


if __name__ == "__main__":
    run_with_tracking()