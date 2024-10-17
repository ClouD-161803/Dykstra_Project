def test_10d():
    # Main script for performance comparison
    import numpy as np
    import matplotlib.pyplot as plt
    from dykstra import dykstra_projection
    from modified_dykstra import \
        dykstra_projection as modified_dykstra_projection
    from data_generator import generate_data
    # Obtain data sets
    d = 10
    n = 15
    z, N, c = generate_data(dimensions=d, num_half_spaces=n)

    max_iter = 50

    # Project and track errors
    dykstra_projection, _, dykstra_error_tuple, _ = dykstra_projection(
        z, N, c, max_iter, track_error=True, dimensions=d)
    modified_dykstra_projection, _, modified_dykstra_error_tuple, _ = modified_dykstra_projection(
        z, N, c, max_iter, track_error=True, dimensions=d)

    # Plot convergence
    iterations = np.arange(0, max_iter, 1)
    plt.plot(iterations, dykstra_error_tuple[0],
             label='Modified Dykstra', linestyle='-', marker='o')
    plt.plot(iterations, modified_dykstra_error_tuple[0],
             label='Dykstra', linestyle='-', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Squared Error')
    plt.legend()
    plt.title('Convergence Comparison (10 Dimensions)')
    plt.grid(True)
    plt.show()

    # # Print final errors
    # print("Dykstra Final Error:", dykstra_error_tuple[0][-1])
    # print("Modified Dykstra Final Error:", modified_dykstra_error_tuple[0][-1])