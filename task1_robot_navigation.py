"""
Smart Robot Navigation System
============================

This module implements a navigation system for a smart robot operating in a grid-based warehouse.
It demonstrates both informed (A*) and uninformed (BFS) search algorithms to find optimal paths
from a starting position to a goal position while avoiding obstacles.

Features:
- 2D grid-based warehouse representation
- A* search algorithm implementation (informed search)
- Breadth-First Search implementation (uninformed search)
- Interactive visualization of the warehouse, search process, and optimal path
- Performance comparison between algorithms

Author: Kodu AI
Date: April 2025
"""

import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

class Warehouse:
    """
    Represents a 2D grid-based warehouse with a start position, goal position, and obstacles.
    
    Attributes:
        grid (numpy.ndarray): 2D array representing the warehouse grid
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        start (tuple): (row, col) coordinates of the starting position
        goal (tuple): (row, col) coordinates of the goal position
    """
    
    # Cell type definitions
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    VISITED = 4
    PATH = 5
    
    def __init__(self, rows=10, cols=10, obstacle_prob=0.2):
        """
        Initialize a random warehouse grid.
        
        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            obstacle_prob (float): Probability of a cell being an obstacle (0.0 to 1.0)
        """
        self.rows = rows
        self.cols = cols
        
        # Initialize empty grid
        self.grid = np.zeros((rows, cols), dtype=int)
        
        # Randomly place obstacles
        for i in range(rows):
            for j in range(cols):
                if np.random.random() < obstacle_prob:
                    self.grid[i, j] = self.OBSTACLE
        
        # Ensure start and goal are not obstacles and are different
        while True:
            self.start = (np.random.randint(0, rows), np.random.randint(0, cols))
            self.goal = (np.random.randint(0, rows), np.random.randint(0, cols))
            
            if (self.start != self.goal and 
                self.grid[self.start] != self.OBSTACLE and 
                self.grid[self.goal] != self.OBSTACLE):
                break
        
        # Mark start and goal on the grid
        self.grid[self.start] = self.START
        self.grid[self.goal] = self.GOAL
    
    def is_valid_position(self, pos):
        """
        Check if a position is valid (within grid bounds and not an obstacle).
        
        Args:
            pos (tuple): (row, col) position to check
            
        Returns:
            bool: True if position is valid, False otherwise
        """
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row, col] != self.OBSTACLE)
    
    def get_neighbors(self, pos):
        """
        Get all valid neighboring positions (up, down, left, right).
        
        Args:
            pos (tuple): (row, col) position
            
        Returns:
            list: List of valid neighboring (row, col) positions
        """
        row, col = pos
        neighbors = [
            (row-1, col),  # Up
            (row+1, col),  # Down
            (row, col-1),  # Left
            (row, col+1)   # Right
        ]
        
        # Filter out invalid positions
        return [n for n in neighbors if self.is_valid_position(n)]
    
    def reset_search(self):
        """Reset the grid by clearing visited cells and path markers."""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == self.VISITED or self.grid[i, j] == self.PATH:
                    self.grid[i, j] = self.EMPTY
        
        # Restore start and goal positions
        self.grid[self.start] = self.START
        self.grid[self.goal] = self.GOAL


class WarehouseVisualizer:
    """
    Visualizes the warehouse grid, search process, and solution path.
    
    Attributes:
        warehouse (Warehouse): The warehouse to visualize
        fig (matplotlib.figure.Figure): The matplotlib figure
        ax (matplotlib.axes.Axes): The matplotlib axes
        colors (list): List of colors for different cell types
    """
    
    def __init__(self, warehouse):
        """
        Initialize the visualizer.
        
        Args:
            warehouse (Warehouse): The warehouse to visualize
        """
        self.warehouse = warehouse
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Define colors for different cell types
        self.colors = ['white', 'black', 'green', 'red', 'lightblue', 'yellow']
        self.cmap = ListedColormap(self.colors)
        
        # Initial plot
        self.img = self.ax.imshow(self.warehouse.grid, cmap=self.cmap, vmin=0, vmax=5)
        
        # Add grid lines
        self.ax.grid(True, which='both', color='gray', linewidth=0.5)
        self.ax.set_xticks(np.arange(-0.5, self.warehouse.cols, 1))
        self.ax.set_yticks(np.arange(-0.5, self.warehouse.rows, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='white', label='Empty'),
            mpatches.Patch(color='black', label='Obstacle'),
            mpatches.Patch(color='green', label='Start'),
            mpatches.Patch(color='red', label='Goal'),
            mpatches.Patch(color='lightblue', label='Visited'),
            mpatches.Patch(color='yellow', label='Path')
        ]
        self.ax.legend(handles=legend_elements, loc='upper center', 
                       bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    def update_plot(self):
        """Update the plot with the current warehouse grid state."""
        self.img.set_data(self.warehouse.grid)
        self.fig.canvas.draw_idle()
        plt.pause(0.01)
    
    def show_static(self):
        """Display a static visualization of the current grid state."""
        self.img.set_data(self.warehouse.grid)
        self.ax.set_title('Warehouse Grid')
        plt.tight_layout()
        plt.show()
    
    def animate_search(self, visited_cells, path=None, title=None):
        """
        Create an animation of the search process.
        
        Args:
            visited_cells (list): List of (row, col) positions in the order they were visited
            path (list, optional): List of (row, col) positions in the final path
            title (str, optional): Title for the animation
        """
        # Reset grid for animation
        self.warehouse.reset_search()
        self.img.set_data(self.warehouse.grid)
        
        if title:
            self.ax.set_title(title)
        
        frames = []
        grid_copy = self.warehouse.grid.copy()
        
        # Add visited cells one by one
        for pos in visited_cells:
            if pos != self.warehouse.start and pos != self.warehouse.goal:
                grid_copy[pos] = Warehouse.VISITED
            frames.append(grid_copy.copy())
        
        # Add path cells one by one if a path exists
        if path:
            for pos in path:
                if pos != self.warehouse.start and pos != self.warehouse.goal:
                    grid_copy[pos] = Warehouse.PATH
                frames.append(grid_copy.copy())
        
        def update(frame):
            self.img.set_data(frame)
            return [self.img]
        
        ani = FuncAnimation(self.fig, update, frames=frames,
                           interval=100, blit=True)
        plt.tight_layout()
        plt.show()


class SearchAlgorithm:
    """Base class for search algorithms."""
    
    def __init__(self, warehouse):
        """
        Initialize the search algorithm.
        
        Args:
            warehouse (Warehouse): The warehouse to navigate
        """
        self.warehouse = warehouse
        self.visited_cells = []
        self.path = []
    
    def search(self, start, goal):
        """
        Search for a path from start to goal.
        
        Args:
            start (tuple): (row, col) starting position
            goal (tuple): (row, col) goal position
            
        Returns:
            tuple: (path, visited_cells) where path is a list of positions from start to goal,
                  and visited_cells is a list of all positions explored during the search
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_results(self):
        """
        Get the search results.
        
        Returns:
            dict: Dictionary containing search results (path, visited cells, metrics)
        """
        return {
            'path': self.path,
            'visited_cells': self.visited_cells,
            'path_length': len(self.path) if self.path else 0,
            'cells_explored': len(self.visited_cells)
        }


class BreadthFirstSearch(SearchAlgorithm):
    """
    Implements Breadth-First Search (BFS) - an uninformed search algorithm.
    BFS explores all neighbors at the present depth before moving to nodes at the next depth level.
    """
    
    def search(self, start, goal):
        """
        Perform BFS search from start to goal.
        
        Args:
            start (tuple): (row, col) starting position
            goal (tuple): (row, col) goal position
            
        Returns:
            tuple: (path, visited_cells)
        """
        # Initialize
        self.visited_cells = []
        self.path = []
        
        # Queue for BFS
        queue = deque([start])
        
        # To track visited cells and reconstruct path
        visited = set([start])
        parent = {start: None}
        
        # Start search timer
        start_time = time.time()
        
        # BFS loop
        while queue:
            current = queue.popleft()
            self.visited_cells.append(current)
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                while current:
                    self.path.append(current)
                    current = parent[current]
                self.path.reverse()
                break
            
            # Explore neighbors
            for neighbor in self.warehouse.get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current
        
        # End search timer
        self.execution_time = time.time() - start_time
        
        return self.path, self.visited_cells


class AStarSearch(SearchAlgorithm):
    """
    Implements A* Search - an informed search algorithm.
    A* uses a heuristic function to guide the search towards the goal more efficiently.
    """
    
    def heuristic(self, a, b):
        """
        Calculate Manhattan distance heuristic between positions a and b.
        
        Args:
            a (tuple): (row, col) first position
            b (tuple): (row, col) second position
            
        Returns:
            int: Manhattan distance between a and b
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def search(self, start, goal):
        """
        Perform A* search from start to goal.
        
        Args:
            start (tuple): (row, col) starting position
            goal (tuple): (row, col) goal position
            
        Returns:
            tuple: (path, visited_cells)
        """
        # Initialize
        self.visited_cells = []
        self.path = []
        
        # Priority queue for A*
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # To track visited cells and reconstruct path
        closed_set = set()
        parent = {start: None}
        
        # Cost from start to each node
        g_score = {start: 0}
        
        # Estimated total cost from start to goal through each node
        f_score = {start: self.heuristic(start, goal)}
        
        # Start search timer
        start_time = time.time()
        
        # A* loop
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            self.visited_cells.append(current)
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                while current:
                    self.path.append(current)
                    current = parent[current]
                self.path.reverse()
                break
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self.warehouse.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + 1
                
                # Check if we found a better path to neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path to neighbor
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    
                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # End search timer
        self.execution_time = time.time() - start_time
        
        return self.path, self.visited_cells


def compare_algorithms(warehouse, visualizer):
    """
    Compare BFS and A* search algorithms.
    
    Args:
        warehouse (Warehouse): The warehouse to navigate
        visualizer (WarehouseVisualizer): The visualizer
        
    Returns:
        dict: Dictionary with comparison results
    """
    # Create search algorithms
    bfs = BreadthFirstSearch(warehouse)
    astar = AStarSearch(warehouse)
    
    # Run BFS
    print("Running Breadth-First Search (Uninformed Search)...")
    bfs.search(warehouse.start, warehouse.goal)
    bfs_results = bfs.get_results()
    bfs_results['execution_time'] = bfs.execution_time
    
    # Visualize BFS
    visualizer.animate_search(
        bfs_results['visited_cells'], 
        bfs_results['path'], 
        "Breadth-First Search (Uninformed)"
    )
    
    # Reset warehouse
    warehouse.reset_search()
    
    # Run A*
    print("Running A* Search (Informed Search)...")
    astar.search(warehouse.start, warehouse.goal)
    astar_results = astar.get_results()
    astar_results['execution_time'] = astar.execution_time
    
    # Visualize A*
    visualizer.animate_search(
        astar_results['visited_cells'], 
        astar_results['path'], 
        "A* Search (Informed)"
    )
    
    # Print comparison
    print("\n===== Algorithm Comparison =====")
    print(f"BFS (Uninformed):")
    print(f"  - Path Length: {bfs_results['path_length']}")
    print(f"  - Cells Explored: {bfs_results['cells_explored']}")
    print(f"  - Execution Time: {bfs_results['execution_time']:.6f} seconds")
    
    print(f"\nA* (Informed):")
    print(f"  - Path Length: {astar_results['path_length']}")
    print(f"  - Cells Explored: {astar_results['cells_explored']}")
    print(f"  - Execution Time: {astar_results['execution_time']:.6f} seconds")
    
    # Calculate improvement
    path_improvement = ((bfs_results['path_length'] - astar_results['path_length']) / 
                         bfs_results['path_length'] * 100) if bfs_results['path_length'] > 0 else 0
    exploration_improvement = ((bfs_results['cells_explored'] - astar_results['cells_explored']) / 
                               bfs_results['cells_explored'] * 100) if bfs_results['cells_explored'] > 0 else 0
    time_improvement = ((bfs_results['execution_time'] - astar_results['execution_time']) / 
                         bfs_results['execution_time'] * 100) if bfs_results['execution_time'] > 0 else 0
    
    print(f"\nA* Improvement over BFS:")
    print(f"  - Path Length: {path_improvement:.2f}%")
    print(f"  - Cells Explored: {exploration_improvement:.2f}%")
    print(f"  - Execution Time: {time_improvement:.2f}%")
    
    # Plot comparison
    plot_comparison(bfs_results, astar_results)
    
    return {
        'bfs': bfs_results,
        'astar': astar_results
    }


def plot_comparison(bfs_results, astar_results):
    """
    Plot comparison between BFS and A* algorithms.
    
    Args:
        bfs_results (dict): BFS search results
        astar_results (dict): A* search results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Metrics to compare
    metrics = ['path_length', 'cells_explored', 'execution_time']
    metric_labels = ['Path Length', 'Cells Explored', 'Execution Time (s)']
    
    # Data for bar chart
    bfs_data = [bfs_results[m] for m in metrics]
    astar_data = [astar_results[m] for m in metrics]
    
    # Bar chart
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, bfs_data, width, label='BFS (Uninformed)')
    ax1.bar(x + width/2, astar_data, width, label='A* (Informed)')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.legend()
    ax1.set_title('Algorithm Performance Comparison')
    
    # Improvement percentages
    improvements = []
    for i, metric in enumerate(metrics):
        if bfs_data[i] > 0:
            imp = (bfs_data[i] - astar_data[i]) / bfs_data[i] * 100
            improvements.append(imp)
        else:
            improvements.append(0)
    
    # Bar chart for improvements
    ax2.bar(metric_labels, improvements, color='green')
    ax2.set_title('A* Improvement over BFS (%)')
    ax2.set_ylabel('Improvement (%)')
    
    # Add text labels above bars
    for i, v in enumerate(improvements):
        ax2.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the warehouse robot navigation simulation."""
    print("===== Smart Robot Navigation System =====")
    print("Initializing warehouse grid...")
    
    # Create warehouse with random grid
    warehouse = Warehouse(rows=15, cols=15, obstacle_prob=0.2)
    
    # Create visualizer
    visualizer = WarehouseVisualizer(warehouse)
    
    # Show initial warehouse grid
    print(f"Start position: {warehouse.start}")
    print(f"Goal position: {warehouse.goal}")
    visualizer.show_static()
    
    # Compare algorithms
    compare_algorithms(warehouse, visualizer)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()