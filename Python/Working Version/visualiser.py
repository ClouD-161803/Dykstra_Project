"""
This module provides a unified visualisation class for projection results.

Classes:
- Visualiser:
    Handles all visualisation of projection solver results including half-spaces,
    paths, errors, and active half-space tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import cm
from projection_result import ProjectionResult


class Visualiser:
    """
    Unified visualisation class for projection solver results.
    
    Attributes:
        result: ProjectionResult object containing solver outputs.
        nc_pairs: List of tuples (label, cmap, N, c) defining half-spaces.
        max_iter: Number of iterations performed.
        x_range: X-axis range for plotting.
        y_range: Y-axis range for plotting.
    """
    
    def __init__(self, result: ProjectionResult, nc_pairs: list, 
                 max_iter: int, x_range: list[float], y_range: list[float]) -> None:
        """
        Initialise the visualiser.
        
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
            cmap: Colourmap name.
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
            cmap: Colourmap name.
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
        Plot the path followed by the algorithm with optional quiver plotting.

        Args:
            ax: Axes handle for plotting.
        """
        path = self.result.path
        errors_for_plotting = self.result.errors_for_plotting
        
        if path is None:
            print("Path data not available.")
            return
        
        # Ensure path is proper shape
        if path.ndim != 3:
            print("Path has unexpected shape.")
            return
        
        # Flatten path for plotting - this includes all points from all iterations and half-spaces
        flattened_path = path.reshape(-1, path.shape[-1])
        
        # Extract x and y coordinates from the path
        x_coords = [point[0] for point in flattened_path]
        y_coords = [point[1] for point in flattened_path]

        # Plot the path
        ax.plot(x_coords, y_coords, marker='.', linestyle='--',
                color='blue', linewidth=0.5, markersize=1,
                label='Projection Path')

        # Plot the errors (quivers) - only where errors were tracked
        if errors_for_plotting is not None and errors_for_plotting.ndim == 3:
            max_iter, n_spaces = errors_for_plotting.shape[0], errors_for_plotting.shape[1]
            for i in range(max_iter):
                for m in range(n_spaces):
                    error = errors_for_plotting[i, m]
                    # Only plot quiver if error vector is non-zero
                    if not np.allclose(error, 0):
                        # The point for the quiver is the start of the projection step
                        point = path[i, m]
                        ax.quiver(point[0], point[1], error[0], error[1],
                                 angles='xy', scale_units='xy', scale=1, alpha=0.3)

        ax.legend()

    def plot_errors(self, ax: Axes) -> None:
        """
        Plot the squared error convergence.

        Args:
            ax: Axes handle for plotting.
        """
        if (self.result.squared_errors is None or 
            self.result.stalled_errors is None or 
            self.result.converged_errors is None):
            print("Error tracking data not available.")
            return

        squared_errors = self.result.squared_errors.copy()
        stalled_errors = self.result.stalled_errors
        converged_errors = self.result.converged_errors
        
        iterations = np.arange(0, self.max_iter, 1)
        
        ax.plot(iterations, squared_errors, color='red',
                label='Errors', linestyle='-', marker='o')
        ax.plot(iterations, stalled_errors, color='#D5B60A',
                label='Stalling', linestyle='-', marker='o')
        ax.plot(iterations, converged_errors, color='green',
                label='Converged\n(error under 1e-3)', linestyle='-', marker='o')
        ax.scatter(iterations[-1], squared_errors[-1],
                   color='#8B0000', marker='o',
                   label=f'Final error is {format(squared_errors[-1], ".2e")}')

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
        if self.result.active_half_spaces is None:
            print("Active half-space data not available.")
            return

        active_spaces = self.result.active_half_spaces
        num_of_spaces = active_spaces.shape[0]
        iterations = np.arange(0, self.max_iter, 1)

        for i in range(num_of_spaces):
            ax = fig.add_subplot(gs[i, 1])
            if ax is None:
                continue
            
            active_space = active_spaces[i]
            ax.plot(iterations, active_space, color='black',
                   label=f'Halfspace {i}', linestyle='-', marker='o')
            ax.set_ylim(0, 1)

            if i == 0:
                ax.set_title('Halfspace Activity')

            ax.grid(True)
            ax.legend()

    def visualise(self, plot_original_point: np.ndarray | None = None,
                  plot_optimal_point: np.ndarray | None = None) -> None:
        """
        Create a comprehensive visualisation of the projection results.

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
        if self.result.squared_errors is not None:
            self.plot_errors(self.ax_error)

        # Plot active halfspaces if available
        if self.result.active_half_spaces is not None:
            self.plot_active_halfspaces(self.fig, gs)

        # Adjust layout
        plt.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.show()
