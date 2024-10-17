def generate_Nc_data(dimensions: int, num_half_spaces: int) -> tuple:
    """
        Generates random data representing half-spaces in a specified number of dimensions
        with a guaranteed non-empty intersection.

        Args:
            dimensions: The number of dimensions for the half-spaces.
            num_half_spaces: The number of half-spaces to generate.

        Returns:
            tuple: A tuple containing:
                - N (numpy.ndarray): A matrix where each row represents the normal vector of a half-space.
                - c (numpy.ndarray): A vector representing the constant offsets of the half-spaces.

        The generated half-spaces are defined by the inequalities N * x <= c, where x is a point in the
        specified dimensional space. The function ensures that the intersection of these half-spaces
        is non-empty by iteratively adjusting the constraints until a feasible solution is found
        using Dykstra's projection algorithm.
        """
    import numpy as np


    # Generate initial random data
    N = np.random.uniform(-1, 1, (num_half_spaces, dimensions))
    c = np.random.uniform(1, 10, num_half_spaces)

    # Function to check for non-empty intersection (using Dykstra's algorithm)
    def has_non_empty_intersection(N, c):
        from dykstra import dykstra_projection
        z = np.zeros(dimensions)  # Origin as the initial point
        projection, _, _, _ = dykstra_projection(z, N, c, max_iter=100)
        return projection is not None

    # Adjust constraints until a non-empty intersection is found
    while not has_non_empty_intersection(N, c):
        # Slightly perturb the constraints
        N += np.random.normal(0, 0.1, N.shape)
        c += np.random.normal(0, 0.1, c.shape)

    return N, c


def generate_feasible_point_outside(N, c):
    """
    Generates a feasible point outside the intersection of half-spaces defined by N and c.

    Args:
        N (numpy.ndarray): A matrix where each row represents the normal vector of a half-space.
        c (numpy.ndarray): A vector representing the constant offsets of the half-spaces.

    Returns:
        numpy.ndarray: A point outside the intersection of the half-spaces.
    """
    import numpy as np
    from dykstra import dykstra_projection


    # Find a point inside the intersection
    x_inside, _, _, _ = dykstra_projection(np.zeros(N.shape[1]), N, c, max_iter=100)

    # Generate a point outside
    direction = np.random.uniform(-1, 1, N.shape[1])
    scaling_factor = 10  # Adjust this value to control the distance from the intersection
    z = x_inside + scaling_factor * direction

    # Verify feasibility (optional)
    if np.any(np.matmul(N, z) > c):
        return z
    else:
        # If z happens to be inside, try generating another one (rare case)
        return generate_feasible_point_outside(N, c)


def generate_data(dimensions: int, num_half_spaces: int) -> tuple:
    # Assuming you have N and c generated from the previous step
    N, c = generate_Nc_data(dimensions, num_half_spaces)
    z = generate_feasible_point_outside(N, c)
    return z, N, c