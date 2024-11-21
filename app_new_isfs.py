import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.special import gamma
import random
from itertools import product, combinations

# Constants
NUM_DRONES = 20
SPACE_SIZE = 100  # Size of the space in each dimension
ALTITUDES = np.linspace(50, 300, NUM_DRONES)  # Different altitude levels
MIN_VELOCITY = 5  # Minimum velocity in m/s
MAX_VELOCITY = 15  # Maximum velocity in m/s
NO_GO_RADIUS = 20  # No-go zone radius in meters
MAX_ITERATIONS = 100  # Maximum iterations for the SFS algorithm
POPULATION_SIZE = 50  # Population size for SFS
NUM_OBSTACLES = 3  # Number of obstacles

# Function to generate random lat/long start and end points for drones
def generate_random_points(num_points, space_size):
    return np.random.rand(num_points, 2) * space_size

# Function to generate random obstacles
def generate_obstacles(num_obstacles, space_size):
    obstacles = []
    for _ in range(num_obstacles):
        min_corner = np.random.rand(3) * space_size * 0.5
        max_corner = min_corner + np.random.rand(3) * (space_size * 0.5)
        obstacles.append((min_corner, max_corner))
    return obstacles

# Function to check if a point is inside an obstacle
def is_inside_obstacle(point, obstacle):
    min_corner, max_corner = obstacle
    return np.all(point >= min_corner) and np.all(point <= max_corner)

# Function to check if a path intersects any obstacles
def path_intersects_obstacles(waypoints, obstacles):
    for waypoint in waypoints:
        for obstacle in obstacles:
            if is_inside_obstacle(waypoint, obstacle):
                return True
    return False

# Function to compute waypoints for a straight line path, including altitude
def compute_waypoints(start, end, altitude, num_waypoints=10):
    latitudes = np.linspace(start[0], end[0], num_waypoints)
    longitudes = np.linspace(start[1], end[1], num_waypoints)
    altitudes = np.full(num_waypoints, altitude)
    return np.column_stack((latitudes, longitudes, altitudes))

# Fitness function to evaluate a solution
def fitness_function(drone_paths, obstacles):
    total_distance = 0
    total_conflicts = 0

    for path in drone_paths:
        waypoints = path['waypoints']
        total_distance += np.sum(np.sqrt(np.sum(np.diff(waypoints[:, :2], axis=0)**2, axis=1)))
        if path_intersects_obstacles(waypoints, obstacles):
            total_distance += 1e6  # Large penalty for intersecting with obstacles

    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            if distance.euclidean(drone_paths[i]['end'], drone_paths[j]['end']) < NO_GO_RADIUS and \
               abs(drone_paths[i]['altitude'] - drone_paths[j]['altitude']) < NO_GO_RADIUS:
                total_conflicts += 1

    return total_distance + total_conflicts * 1000  # Penalize conflicts heavily

# Lévy flight
def levy_flight(Lambda, size):
    sigma1 = np.power((gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) / 
                      (gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=size)
    v = np.random.normal(0, sigma2, size=size)
    step = u / np.power(np.abs(v), 1 / Lambda)
    return step

# Gaussian walk
def gaussian_walk(mu=0, sigma=1, size=(10, 3)):
    return np.random.normal(mu, sigma, size=size)

# Improved SFS algorithm
def improved_sfs(obstacles):
    population = [generate_initial_solution(obstacles) for _ in range(POPULATION_SIZE)]

    for iteration in range(MAX_ITERATIONS):
        new_population = []

        for solution in population:
            new_solution = levy_flight_solution(solution, obstacles)
            new_solution = local_search(new_solution, obstacles)
            new_population.append(new_solution)

        population = select_best_solutions(population + new_population, obstacles)

    return select_best_solution(population, obstacles)

# Function to generate an initial solution
def generate_initial_solution(obstacles):
    start_points = generate_random_points(NUM_DRONES, SPACE_SIZE)
    end_points = generate_random_points(NUM_DRONES, SPACE_SIZE)
    drone_paths = []

    for i in range(NUM_DRONES):
        waypoints = compute_waypoints(start_points[i], end_points[i], ALTITUDES[i])
        # Apply a heuristic to avoid obstacles by adjusting the waypoints
        waypoints = adjust_waypoints_for_obstacles(waypoints, obstacles)
        drone_paths.append({
            'start': start_points[i],
            'end': end_points[i],
            'waypoints': waypoints,
            'altitude': ALTITUDES[i],
            'velocity': random.uniform(MIN_VELOCITY, MAX_VELOCITY)
        })

    return drone_paths

# Function to adjust waypoints to avoid obstacles
def adjust_waypoints_for_obstacles(waypoints, obstacles):
    for _ in range(100):  # Limit the number of adjustments to prevent infinite loops
        if not path_intersects_obstacles(waypoints, obstacles):
            return waypoints
        waypoints += np.random.normal(0, 5, size=waypoints.shape)  # Adjust waypoints slightly
    return waypoints

# Function to apply Lévy flight to diversify the solution
def levy_flight_solution(solution, obstacles):
    new_solution = []

    for i, path in enumerate(solution):
        step = levy_flight(1.5, size=path['waypoints'].shape)  # Lévy flight with Lambda = 1.5
        new_waypoints = path['waypoints'] + step
        new_waypoints = adjust_waypoints_for_obstacles(new_waypoints, obstacles)
        new_solution.append({
            'start': path['start'],
            'end': path['end'],
            'waypoints': new_waypoints,
            'altitude': path['altitude'],
            'velocity': path['velocity']
        })

    return new_solution

# Function to perform a local search to refine the solution
def local_search(solution, obstacles):
    new_solution = []

    for i, path in enumerate(solution):
        step = gaussian_walk(size=path['waypoints'].shape)  # Gaussian walk
        new_waypoints = path['waypoints'] + step
        new_waypoints = adjust_waypoints_for_obstacles(new_waypoints, obstacles)
        new_solution.append({
            'start': path['start'],
            'end': path['end'],
            'waypoints': new_waypoints,
            'altitude': path['altitude'],
            'velocity': path['velocity']
        })

    return new_solution

# Function to select the best solutions from the population
def select_best_solutions(population, obstacles):
    population.sort(key=lambda sol: fitness_function(sol, obstacles))
    return population[:POPULATION_SIZE]

# Function to select the best solution from the population
def select_best_solution(population, obstacles):
    return min(population, key=lambda sol: fitness_function(sol, obstacles))

# Function to visualize the drone paths and obstacles
def visualize_paths(drone_paths, obstacles):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for path in drone_paths:
        waypoints = path['waypoints']
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], marker='o')
        ax.scatter(path['start'][0], path['start'][1], path['altitude'], color='green')  # Start point
        ax.scatter(path['end'][0], path['end'][1], path['altitude'], color='red')  # End point

    for obstacle in obstacles:
        min_corner, max_corner = obstacle
        r = np.array(list(product(*zip(min_corner, max_corner))))
        for s, e in combinations(r, 2):
            if np.sum(np.abs(s - e)) in [0, max_corner[0] - min_corner[0], max_corner[1] - min_corner[1], max_corner[2] - min_corner[2]]:
                ax.plot3D(*zip(s, e), color="b")

    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude')
    ax.set_title('Drone Paths and Obstacles')
    plt.grid(True)
    plt.show()

# Main simulation function
def simulate_drone_paths():
    obstacles = generate_obstacles(NUM_OBSTACLES, SPACE_SIZE)
    best_solution = improved_sfs(obstacles)
    visualize_paths(best_solution, obstacles)

if __name__ == "__main__":
    simulate_drone_paths()
