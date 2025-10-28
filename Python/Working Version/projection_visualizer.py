"""
This module provides a unified visualization class for projection results.

Classes:
- ProjectionVisualizer:
    Handles all visualization of projection solver results including half-spaces,
    paths, errors, and active half-space tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import cm
from projection_result import ProjectionResult


class ProjectionVisualizer:
    """
    Unified visualization class for projection solver results.
    
    Attributes:
        result: ProjectionResult object containing solver outputs.
        nc_pairs: List of tuples (label, cmap, N, c) defining half-spaces.
        max_iter: Number of iterations performed.
        x_range: X-axis range for plotting.
        y_range: Y-axis range for plotting.
    """
    
    def __init__(self, result: ProjectionResult, nc_pairs: list, 
                 max_iter: int, x_range: list, y_range: list):
        """
        Initialize the visualizer.
        
        Args:
            result: ProjectionResult object from solver.
            nc_pairs: List of (label, cmap, N, c) tuples for half-spaces.
            max_iter: Number of iterations.
            x_range: [min_x, max_x] for plotting domain.
            y_range: [min_y, max_y] for plotting domain.
        """
        self.result = result
        self.nc_pairs = nc_pairs
        self.max_iter = max_iter
        self.x_range = x_range
        self.y_range = y_range
        self.fig: Figure | None = None
        self.ax_main: Axes | None = None
        self.ax_error: Axes | None = None
        self.ax_activity: Axes | None = None

    def plot_2d_space(self, N: np.ndarray, c: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      label: str, cmap: str, ax: Axes) -> None:
        """
        Plot a 2D region defined by the intersection of half-spaces.

        Args:
            N: Matrix of normal vectors.
            c: Vector of constant offsets.
            X: 2D array of x coordinates.
            Y: 2D array of y coordinates.
            label: Label for the plot.
            cmap: Colormap name.
            ax: Axes handle for plotting.
        """
        Z = np.ones_like(X)

        for i in range(N.shape[0]):
            dot_product = np.dot(np.vstack([X.ravel(), Y.ravel()]).T, N[i])
            Z = np.where(dot_product.reshape(X.shape) > c[i], 0, Z)

        colourmap = cm.get_cmap(cmap)
        colour = colourmap(0.69)

        ax.contourf(X, Y, Z, cmap=colourmap, alpha=0.5)
        ax.plot([], [], color=colour, alpha=0.5, label=label)

    def plot_1d_space(self, N: np.ndarray, c: np.ndarray, label: str, 
                      cmap: str, ax: Axes) -> None:
        """
        Plot a 1D region (line) defined by the intersection of half-spaces.

        Args:
            N: Matrix of normal vectors.
            c: Vector of constant offsets.
            label: Label for the plot.
            cmap: Colormap name.
            ax: Axes handle for plotting.
        """
        colourmap = cm.get_cmap(cmap)
        colour = colourmap(0.69)

        if N[0, 1] == 0:
            ax.axvline(x=c[0] / N[0, 0], linestyle='-', linewidth=2,
                        label='Vertical Line', color=colour)
        elif N[0, 0] == 0:
            ax.axhline(y=c[0] / N[0, 1], linestyle='-', linewidth=2,
                        label='Horizontal Line', color=colour)
        else:
            x_line = np.linspace(self.x_range[0], self.x_range[1], 100)
            y_line = (c[0] - N[0, 0] * x_line) / N[0, 1]
            ax.plot(x_line, y_line, linewidth=2, label=label, color=colour)

    def plot_half_spaces(self, ax: Axes) -> None:
        """
        Plot all half-spaces.

        Args:
            ax: Axes handle for plotting.
        """
        try:
            x = np.linspace(self.x_range[0], self.x_range[1], 500)
            y = np.linspace(self.y_range[0], self.y_range[1], 500)
            X, Y = np.meshgrid(x, y)

            for label, cmap, N, c in self.nc_pairs:
                rank = np.linalg.matrix_rank(N)
                if rank == 1:
                    self.plot_1d_space(N, c, label, cmap, ax)
                elif rank == 2:
                    self.plot_2d_space(N, c, X, Y, label, cmap, ax)
                else:
                    raise ValueError("Dimension not supported. "
                                   "Please provide N and c for 1D or 2D cases.")

            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"Dykstra's algorithm executed for {self.max_iter} iterations")
            ax.grid(True)
            ax.legend()

        except TypeError as e:
            print(f"TypeError occurred: {e}. "
                  f"Please ensure nc_pairs is a list of tuples.")
        except ValueError as e:
            print(f"ValueError occurred: {e}. "
                  f"Check the format of nc_pairs or the dimensions of N.")

    def plot_path(self, ax: Axes) -> None:
        """
        Plot the path followed by the algorithm.

        Args:
            ax: Axes handle for plotting.
        """
        path = self.result.path
        active_half_spaces = self.result.active_half_spaces
        
        n_spaces = len(active_half_spaces) if active_half_spaces is not None else 0
        x_coords = [path[0][0]]
        y_coords = [path[0][1]]
        
        if active_half_spaces is not None:
            x_coords.extend([point[0] for i, point in enumerate(path[1:-1]) 
                           if active_half_spaces[i % n_spaces][i // n_spaces] == 1])
            y_coords.extend([point[1] for i, point in enumerate(path[1:-1]) 
                           if active_half_spaces[i % n_spaces][i // n_spaces] == 1])

        ax.plot(x_coords, y_coords, marker='.', linestyle='--',
                color='blue', linewidth=0.5, markersize=1,
                label='Projection Path')
        ax.legend()

    def plot_errors(self, ax: Axes) -> None:
        """
        Plot the squared error convergence.

        Args:
            ax: Axes handle for plotting.
        """
        if not self.result.has_error_tracking():
            print("Error tracking data not available.")
            return

        squared_errors = self.result.squared_errors
        stalled_errors = self.result.stalled_errors
        converged_errors = self.result.converged_errors
        
        if squared_errors is None or stalled_errors is None or converged_errors is None:
            print("Error data is incomplete.")
            return
        
        iterations = np.arange(0, self.max_iter, 1)
        
        # Get last error before modification
        last_error = squared_errors[-1]
        squared_errors_copy = squared_errors.copy()
        squared_errors_copy[-1] = None
        
        ax.plot(iterations, squared_errors_copy, color='red',
                label='Errors', linestyle='-', marker='o')
        ax.plot(iterations, stalled_errors, color='#D5B60A',
                label='Stalling', linestyle='-', marker='o')
        ax.plot(iterations, converged_errors, color='green',
                label='Converged\n(error under 1e-3)', linestyle='-', marker='o')
        ax.scatter(iterations[-1], last_error,
                   color='#8B0000', marker='o',
                   label=f'Final error is {format(last_error, ".2e")}')

        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Squared Errors')
        ax.set_title('Convergence of Squared Errors')
        ax.grid(True)
        ax.legend()

    def plot_active_halfspaces(self, fig: Figure, gs: gridspec.GridSpec) -> None:
        """
        Plot the activity of half-spaces over iterations.

        Args:
            fig: Figure handle.
            gs: GridSpec handle.
        """
        if not self.result.has_active_halfspace_data():
            print("Active half-space data not available.")
            return

        active_spaces = self.result.active_half_spaces
        if active_spaces is None:
            print("Active half-space data is None.")
            return
        
        num_of_spaces = len(active_spaces)
        iterations = np.arange(0, self.max_iter, 1)
        ax_vector = np.zeros(num_of_spaces, dtype=object)

        for i in range(num_of_spaces):
            ax_vector[i] = fig.add_subplot(gs[i, 1])

        for i, (active_space, ax) in enumerate(zip(active_spaces, ax_vector)):
            if ax is None:
                continue
            ax.plot(iterations, active_space, color='black',
                   label=f'Halfspace {i}', linestyle='-', marker='o')
            ax.set_ylim(0, 1)

            if i == 0:
                ax.set_title('Halfspace Activity')

            ax.grid(True)
            ax.legend()

    def visualize(self, plot_original_point: np.ndarray | None = None,
                  plot_optimal_point: np.ndarray | None = None) -> None:
        """
        Create a comprehensive visualization of the projection results.

        Args:
            plot_original_point: Original point z (optional).
            plot_optimal_point: Optimal solution (optional).
        """
        # Create infrastructure for plots
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2)
        self.ax_main = self.fig.add_subplot(gs[:2, 0])
        self.ax_error = self.fig.add_subplot(gs[2, 0])

        if self.ax_main is None or self.ax_error is None:
            print("Failed to create axes.")
            return

        # Plot half-spaces
        self.plot_half_spaces(self.ax_main)

        # Plot path
        self.plot_path(self.ax_main)

        # Plot points
        if plot_original_point is not None:
            self.ax_main.scatter(plot_original_point[0], plot_original_point[1],
                               color='green', marker='o', label='Original Point')

        self.ax_main.scatter(self.result.projection[0], self.result.projection[1],
                           color='green', marker='x', label='Projection')

        if plot_optimal_point is not None:
            self.ax_main.scatter(plot_optimal_point[0], plot_optimal_point[1],
                               color='red', marker='x', label='Optimal Solution')

        self.ax_main.legend()

        # Plot errors if available
        if self.result.has_error_tracking():
            self.plot_errors(self.ax_error)

        # Plot active halfspaces if available
        if self.result.has_active_halfspace_data():
            self.plot_active_halfspaces(self.fig, gs)

        # Adjust layout
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()
