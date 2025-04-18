"""
Delivery Robot Route Planner
===========================

This module implements a smart route planner for a delivery robot that must visit
multiple locations in a city and return to the warehouse. This is a classic case of
the Travelling Salesman Problem (TSP).

Features:
- Multiple TSP algorithms implementation:
  * Brute Force (exact, for small problems)
  * Dynamic Programming (exact, more efficient than brute force)
  * Genetic Algorithm (metaheuristic, for larger problems)
  * Nearest Neighbor (greedy heuristic, fast but suboptimal)
- Interactive visualization of delivery points and routes
- Performance comparison between algorithms
- Simulation of different city layouts and scenarios

Author: Kodu AI
Date: April 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import random
import math
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
from matplotlib.lines import Line2D


class DeliveryCity:
    """
    Represents a city with delivery points and a warehouse.
    
    Attributes:
        num_points (int): Total number of points (including warehouse)
        locations (numpy.ndarray): Array of (x, y) coordinates
        distances (numpy.ndarray): Distance matrix between all points
        warehouse_idx (int): Index of the warehouse location (always 0)
    """
    
    def __init__(self, num_points=10, city_size=100, seed=None):
        """
        Initialize a city with random delivery point locations.
        
        Args:
            num_points (int): Number of delivery points (excluding warehouse)
            city_size (int): Size of the city square (city_size x city_size)
            seed (int, optional): Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.num_points = num_points + 1  # +1 for warehouse
        self.warehouse_idx = 0
        
        # Generate random locations (warehouse is at index 0)
        self.locations = np.random.randint(0, city_size, size=(self.num_points, 2))
        
        # Make warehouse location distinctive (e.g., center of the city)
        self.locations[0] = [city_size // 2, city_size // 2]
        
        # Compute distance matrix
        self.distances = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            for j in range(self.num_points):
                if i != j:
                    self.distances[i, j] = np.linalg.norm(self.locations[i] - self.locations[j])
    
    def get_distance(self, i, j):
        """
        Get distance between two points.
        
        Args:
            i (int): Index of first point
            j (int): Index of second point
            
        Returns:
            float: Distance between points i and j
        """
        return self.distances[i, j]
    
    def total_distance(self, route):
        """
        Calculate total distance of a route.
        
        Args:
            route (list): List of point indices representing the route
            
        Returns:
            float: Total distance of the route
        """
        total = 0
        for i in range(len(route) - 1):
            total += self.get_distance(route[i], route[i + 1])
        # Add distance back to warehouse
        if route[0] != route[-1]:
            total += self.get_distance(route[-1], route[0])
        return total


class CityVisualizer:
    """
    Visualizes the city, delivery points, and routes.
    
    Attributes:
        city (DeliveryCity): The city to visualize
        fig (matplotlib.figure.Figure): The matplotlib figure
        ax (matplotlib.axes.Axes): The matplotlib axes
    """
    
    def __init__(self, city):
        """
        Initialize the visualizer.
        
        Args:
            city (DeliveryCity): The city to visualize
        """
        self.city = city
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Set limits with some padding
        max_coord = np.max(city.locations) * 1.1
        self.ax.set_xlim(0, max_coord)
        self.ax.set_ylim(0, max_coord)
    
    def plot_points(self):
        """Plot all delivery points and the warehouse."""
        # Plot warehouse (larger and different color)
        warehouse = self.city.locations[0]
        self.ax.scatter(warehouse[0], warehouse[1], s=200, c='red', marker='s', 
                        label='Warehouse', zorder=10)
        
        # Plot delivery points
        delivery_points = self.city.locations[1:]
        self.ax.scatter(delivery_points[:, 0], delivery_points[:, 1], s=100, c='blue', 
                        marker='o', label='Delivery Points')
        
        # Add point labels
        for i, (x, y) in enumerate(self.city.locations):
            label = 'W' if i == 0 else str(i)
            self.ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')
    
    def plot_route(self, route, route_name="Route", color='green', linestyle='-', alpha=1.0):
        """
        Plot a delivery route.
        
        Args:
            route (list): List of point indices representing the route
            route_name (str, optional): Name of the route for the legend
            color (str, optional): Color of the route line
            linestyle (str, optional): Style of the route line
            alpha (float, optional): Alpha (transparency) value for the route line
        """
        # Create a full route (add return to warehouse if not already included)
        full_route = list(route)
        if full_route[0] != full_route[-1]:
            full_route.append(full_route[0])
        
        # Plot route
        route_x = [self.city.locations[i][0] for i in full_route]
        route_y = [self.city.locations[i][1] for i in full_route]
        
        self.ax.plot(route_x, route_y, color=color, linestyle=linestyle, linewidth=2, alpha=alpha,
                     label=f"{route_name} (Dist: {self.city.total_distance(route):.2f})")
        
        # Add direction arrows
        for i in range(len(full_route) - 1):
            start_x, start_y = self.city.locations[full_route[i]]
            end_x, end_y = self.city.locations[full_route[i + 1]]
            
            # Calculate midpoint for arrow
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Calculate direction
            dx = end_x - start_x
            dy = end_y - start_y
            arrow_len = np.sqrt(dx**2 + dy**2)
            
            # Only add arrow if points are not too close
            if arrow_len > 10:
                self.ax.arrow(mid_x - 0.1*dx, mid_y - 0.1*dy, 0.2*dx, 0.2*dy, 
                              head_width=arrow_len/20, head_length=arrow_len/15,
                              fc=color, ec=color, alpha=alpha)
    
    def animate_route(self, route, route_name="Route", interval=300):
        """
        Animate the traversal of a route.
        
        Args:
            route (list): List of point indices representing the route
            route_name (str, optional): Name of the route for the title
            interval (int, optional): Animation interval in milliseconds
        """
        # Create a full route (add return to warehouse if not already included)
        full_route = list(route)
        if full_route[0] != full_route[-1]:
            full_route.append(full_route[0])
        
        # Clear previous animation title if exists
        if hasattr(self, 'anim_title'):
            self.anim_title.remove()
        
        # Set animation title
        self.anim_title = self.ax.set_title(f"Animating {route_name} (Distance: {self.city.total_distance(route):.2f})")
        
        # Line to update in animation
        line, = self.ax.plot([], [], 'g-', linewidth=2)
        robot, = self.ax.plot([], [], 'ro', markersize=10)
        
        # Initialize function for animation
        def init():
            line.set_data([], [])
            robot.set_data([], [])
            return line, robot
        
        # Update function for animation
        def update(frame):
            # Get coordinates up to current frame
            route_x = [self.city.locations[i][0] for i in full_route[:frame+1]]
            route_y = [self.city.locations[i][1] for i in full_route[:frame+1]]
            
            # Update line
            line.set_data(route_x, route_y)
            
            # Update robot position (at current point)
            if frame < len(full_route):
                current_point = full_route[frame]
                robot.set_data(self.city.locations[current_point][0], 
                               self.city.locations[current_point][1])
            
            return line, robot
        
        # Create animation
        ani = FuncAnimation(self.fig, update, frames=len(full_route),
                           init_func=init, blit=True, interval=interval)
        
        # Display animation
        plt.tight_layout()
        plt.show()
    
    def show(self, title=None):
        """
        Display the current visualization.
        
        Args:
            title (str, optional): Title for the plot
        """
        if title:
            self.ax.set_title(title)
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


class TSPSolver:
    """Base class for TSP solvers."""
    
    def __init__(self, city):
        """
        Initialize the TSP solver.
        
        Args:
            city (DeliveryCity): The city with delivery points
        """
        self.city = city
        self.best_route = None
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def solve(self):
        """
        Solve the TSP problem.
        
        Returns:
            tuple: (best_route, best_distance)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_results(self):
        """
        Get the solver results.
        
        Returns:
            dict: Dictionary containing solver results
        """
        return {
            'best_route': self.best_route,
            'best_distance': self.best_distance,
            'execution_time': self.execution_time,
            'method': self.__class__.__name__
        }


class BruteForceTSP(TSPSolver):
    """
    Implements a brute force solution to the TSP.
    Evaluates all possible permutations of delivery points.
    Only feasible for small problems.
    """
    
    def solve(self):
        """
        Solve TSP using brute force (all permutations).
        
        Returns:
            tuple: (best_route, best_distance)
        """
        # Start with warehouse
        warehouse = self.city.warehouse_idx
        
        # Points to visit (excluding warehouse)
        points = list(range(1, self.city.num_points))
        
        # Start time
        start_time = time.time()
        
        # Try all permutations
        for perm in itertools.permutations(points):
            # Make a complete route starting and ending at warehouse
            route = [warehouse] + list(perm)
            
            # Calculate distance
            distance = self.city.total_distance(route)
            
            # Update best route if found
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route
        
        # End time
        self.execution_time = time.time() - start_time
        
        return self.best_route, self.best_distance


class DynamicProgrammingTSP(TSPSolver):
    """
    Implements a dynamic programming solution to the TSP.
    Uses the Held-Karp algorithm which is more efficient than brute force
    but still exact.
    """
    
    def solve(self):
        """
        Solve TSP using dynamic programming (Held-Karp algorithm).
        
        Returns:
            tuple: (best_route, best_distance)
        """
        # Start time
        start_time = time.time()
        
        # Number of points
        n = self.city.num_points
        
        # Initialize memoization table
        # dp[S][i] = min distance of path from 0 to i visiting all vertices in S exactly once
        dp = {}
        
        # Initialize route reconstruction information
        parent = {}
        
        # Warehouse is the start and end point
        warehouse = self.city.warehouse_idx
        
        # Initial state: singleton sets with just one city
        for i in range(1, n):
            dp[frozenset([i]), i] = self.city.get_distance(warehouse, i)
            parent[frozenset([i]), i] = warehouse
        
        # Dynamic programming recurrence
        # For each subset size from 2 to n-1
        for subset_size in range(2, n):
            # For each subset of cities of the given size
            for subset in itertools.combinations(range(1, n), subset_size):
                # Convert to frozenset for dict key
                frozen_subset = frozenset(subset)
                
                # For each last city in the path
                for i in subset:
                    # Try all previous cities
                    subset_without_i = frozen_subset - {i}
                    
                    # Find the best predecessor
                    dp[frozen_subset, i] = float('inf')
                    for j in subset:
                        if j != i:
                            value = dp[subset_without_i, j] + self.city.get_distance(j, i)
                            if value < dp[frozen_subset, i]:
                                dp[frozen_subset, i] = value
                                parent[frozen_subset, i] = j
        
        # Find final best path
        all_cities = frozenset(range(1, n))
        self.best_distance = float('inf')
        last_city = None
        
        # Consider all possible last cities before returning to warehouse
        for i in range(1, n):
            current_distance = dp[all_cities, i] + self.city.get_distance(i, warehouse)
            if current_distance < self.best_distance:
                self.best_distance = current_distance
                last_city = i
        
        # Reconstruct path
        self.best_route = [warehouse]
        
        # Backtracking
        current_set = all_cities
        current_city = last_city
        
        while current_set:
            self.best_route.append(current_city)
            new_city = parent[current_set, current_city]
            current_set = current_set - {current_city}
            current_city = new_city
        
        # End time
        self.execution_time = time.time() - start_time
        
        return self.best_route, self.best_distance


class NearestNeighborTSP(TSPSolver):
    """
    Implements a nearest neighbor heuristic solution to the TSP.
    A greedy algorithm that always visits the closest unvisited delivery point.
    Fast but often suboptimal.
    """
    
    def solve(self):
        """
        Solve TSP using the nearest neighbor heuristic.
        
        Returns:
            tuple: (best_route, best_distance)
        """
        # Start time
        start_time = time.time()
        
        # Start at warehouse
        current = self.city.warehouse_idx
        self.best_route = [current]
        
        # Set of unvisited points (excluding warehouse)
        unvisited = set(range(1, self.city.num_points))
        
        # Nearest neighbor algorithm
        while unvisited:
            # Find nearest unvisited point
            nearest = None
            min_distance = float('inf')
            
            for point in unvisited:
                dist = self.city.get_distance(current, point)
                if dist < min_distance:
                    nearest = point
                    min_distance = dist
            
            # Add nearest to route and remove from unvisited
            self.best_route.append(nearest)
            current = nearest
            unvisited.remove(nearest)
        
        # Calculate total distance
        self.best_distance = self.city.total_distance(self.best_route)
        
        # End time
        self.execution_time = time.time() - start_time
        
        return self.best_route, self.best_distance


class GeneticAlgorithmTSP(TSPSolver):
    """
    Implements a genetic algorithm solution to the TSP.
    Evolves a population of routes to find a near-optimal solution.
    Good for larger problems where exact methods are infeasible.
    """
    
    def __init__(self, city, population_size=50, generations=100, mutation_rate=0.01):
        """
        Initialize the genetic algorithm solver.
        
        Args:
            city (DeliveryCity): The city with delivery points
            population_size (int): Size of the population
            generations (int): Number of generations to evolve
            mutation_rate (float): Probability of mutation for each gene
        """
        super().__init__(city)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # For tracking evolution
        self.evolution_history = []
    
    def _create_individual(self):
        """
        Create a random individual (route).
        
        Returns:
            list: A random route starting at the warehouse
        """
        # Create a route with all points except warehouse
        points = list(range(1, self.city.num_points))
        random.shuffle(points)
        
        # Route starts at warehouse
        return [self.city.warehouse_idx] + points
    
    def _initialize_population(self):
        """
        Initialize a random population.
        
        Returns:
            list: A list of random routes
        """
        return [self._create_individual() for _ in range(self.population_size)]
    
    def _fitness(self, route):
        """
        Calculate fitness of a route (inverse of distance).
        
        Args:
            route (list): A route
            
        Returns:
            float: Fitness value (higher is better)
        """
        distance = self.city.total_distance(route)
        return 1 / distance if distance > 0 else float('inf')
    
    def _selection(self, population):
        """
        Select individuals for reproduction using tournament selection.
        
        Args:
            population (list): Current population
            
        Returns:
            list: Selected individuals
        """
        tournament_size = 5
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            tournament = random.sample(population, tournament_size)
            
            # Select best individual from tournament
            best = max(tournament, key=self._fitness)
            selected.append(best)
        
        return selected
    
    def _crossover(self, parent1, parent2):
        """
        Perform ordered crossover between two parents.
        
        Args:
            parent1 (list): First parent route
            parent2 (list): Second parent route
            
        Returns:
            list: Child route
        """
        # Ensure warehouse is at the start
        warehouse = self.city.warehouse_idx
        
        # Remove warehouse for crossover
        p1 = parent1[1:]
        p2 = parent2[1:]
        
        # Choose crossover points
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create child with segment from parent1
        child = [None] * size
        for i in range(start, end + 1):
            child[i] = p1[i]
        
        # Fill remaining positions with parent2's genes (in order)
        p2_genes = [gene for gene in p2 if gene not in child]
        for i in range(size):
            if child[i] is None:
                child[i] = p2_genes.pop(0)
        
        # Re-add warehouse at the start
        return [warehouse] + child
    
    def _mutation(self, route):
        """
        Apply swap mutation to a route.
        
        Args:
            route (list): A route
            
        Returns:
            list: Mutated route
        """
        # Copy route
        mutated = route.copy()
        
        # Warehouse should stay at the start, only mutate delivery points
        for i in range(1, len(mutated)):
            if random.random() < self.mutation_rate:
                # Select another position to swap with (not warehouse)
                j = random.randint(1, len(mutated) - 1)
                mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def solve(self):
        """
        Solve TSP using genetic algorithm.
        
        Returns:
            tuple: (best_route, best_distance)
        """
        # Start time
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population()
        
        # Track best solution
        best_individual = max(population, key=self._fitness)
        best_fitness = self._fitness(best_individual)
        self.best_route = best_individual
        self.best_distance = self.city.total_distance(best_individual)
        
        # Save initial state
        self.evolution_history.append({
            'generation': 0,
            'best_distance': self.best_distance,
            'avg_distance': np.mean([self.city.total_distance(ind) for ind in population])
        })
        
        # Evolution loop
        for generation in range(1, self.generations + 1):
            # Selection
            selected = self._selection(population)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_individual)
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = random.sample(selected, 2)
                
                # Create child
                child = self._crossover(parent1, parent2)
                
                # Mutate child
                child = self._mutation(child)
                
                # Add to new population
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Update best solution
            current_best = max(population, key=self._fitness)
            current_fitness = self._fitness(current_best)
            
            if current_fitness > best_fitness:
                best_individual = current_best
                best_fitness = current_fitness
                self.best_route = best_individual
                self.best_distance = self.city.total_distance(best_individual)
            
            # Record history
            self.evolution_history.append({
                'generation': generation,
                'best_distance': self.best_distance,
                'avg_distance': np.mean([self.city.total_distance(ind) for ind in population])
            })
        
        # End time
        self.execution_time = time.time() - start_time
        
        return self.best_route, self.best_distance
    
    def plot_evolution(self):
        """Plot the evolution of the genetic algorithm."""
        if not self.evolution_history:
            print("No evolution history available. Solve TSP first.")
            return
        
        # Extract data
        generations = [data['generation'] for data in self.evolution_history]
        best_distances = [data['best_distance'] for data in self.evolution_history]
        avg_distances = [data['avg_distance'] for data in self.evolution_history]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_distances, 'b-', linewidth=2, label='Best Distance')
        plt.plot(generations, avg_distances, 'r--', linewidth=1, label='Average Distance')
        
        plt.title('Genetic Algorithm Evolution')
        plt.xlabel('Generation')
        plt.ylabel('Route Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def compare_algorithms(city, visualizer, max_points_exact=10):
    """
    Compare different TSP algorithms.
    
    Args:
        city (DeliveryCity): The city to solve
        visualizer (CityVisualizer): The visualizer
        max_points_exact (int): Maximum number of points for exact algorithms
        
    Returns:
        dict: Dictionary with comparison results
    """
    results = {}
    
    # Create solvers
    solvers = []
    
    # Only use exact methods for small problems
    if city.num_points <= max_points_exact:
        solvers.append(BruteForceTSP(city))
        solvers.append(DynamicProgrammingTSP(city))
    
    # Always use these methods
    solvers.append(NearestNeighborTSP(city))
    solvers.append(GeneticAlgorithmTSP(city))
    
    # Colors for routes
    colors = ['green', 'purple', 'orange', 'brown']
    
    # Solve with each algorithm
    for i, solver in enumerate(solvers):
        # Print algorithm name
        print(f"\nRunning {solver.__class__.__name__}...")
        
        # Solve TSP
        solver.solve()
        
        # Store results
        results[solver.__class__.__name__] = solver.get_results()
        
        # Print results
        print(f"  - Best distance: {solver.best_distance:.2f}")
        print(f"  - Execution time: {solver.execution_time:.6f} seconds")
        
        # Plot route
        visualizer.plot_route(
            solver.best_route,
            route_name=solver.__class__.__name__,
            color=colors[i % len(colors)]
        )
    
    # Plot all routes
    visualizer.show(title="Comparison of TSP Algorithms")
    
    # Plot GA evolution if available
    for solver in solvers:
        if isinstance(solver, GeneticAlgorithmTSP):
            solver.plot_evolution()
    
    # Animate best route (from genetic algorithm or other best algorithm)
    best_solver = min(solvers, key=lambda s: s.best_distance)
    print(f"\nAnimating best route from {best_solver.__class__.__name__}...")
    visualizer.animate_route(best_solver.best_route, route_name=best_solver.__class__.__name__)
    
    # Plot comparison bar chart
    plot_comparison(results)
    
    return results


def plot_comparison(results):
    """
    Plot comparison between different TSP algorithms.
    
    Args:
        results (dict): Dictionary with results from different algorithms
    """
    # Extract data
    algorithms = list(results.keys())
    distances = [results[algo]['best_distance'] for algo in algorithms]
    times = [results[algo]['execution_time'] for algo in algorithms]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distance comparison
    ax1.bar(algorithms, distances, color='blue', alpha=0.7)
    ax1.set_title('Route Distance Comparison')
    ax1.set_ylabel('Distance')
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Add text labels
    for i, v in enumerate(distances):
        ax1.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Time comparison (logarithmic scale)
    ax2.bar(algorithms, times, color='green', alpha=0.7)
    ax2.set_title('Execution Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_yscale('log')
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Add text labels
    for i, v in enumerate(times):
        ax2.text(i, v * 1.1, f"{v:.6f}s", ha='center')
    
    plt.tight_layout()
    plt.show()


def run_tsp_simulation():
    """Run a simulation of the delivery robot route planning."""
    print("===== Delivery Robot Route Planner =====")
    
    # Get user input for number of delivery points
    while True:
        try:
            num_points = int(input("Enter number of delivery points (3-15 recommended): "))
            if num_points < 3:
                print("Too few points. Please enter at least 3.")
            elif num_points > 15:
                print("Warning: Large number of points may slow down exact algorithms.")
                confirm = input("Continue with this many points? (y/n): ")
                if confirm.lower() == 'y':
                    break
            else:
                break
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nCreating city with {num_points} delivery points...")
    
    # Create city with delivery points
    city = DeliveryCity(num_points=num_points, seed=42)
    
    # Create visualizer
    visualizer = CityVisualizer(city)
    
    # Plot points
    visualizer.plot_points()
    visualizer.show(title="Delivery Points")
    
    # Compare algorithms
    print("\nComparing TSP algorithms...")
    compare_algorithms(city, visualizer, max_points_exact=10)
    
    print("\nSimulation complete!")


def main():
    """Main function to run the delivery robot route planning."""
    run_tsp_simulation()


if __name__ == "__main__":
    main()