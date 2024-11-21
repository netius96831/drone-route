import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import Tk, Label, Toplevel

# Parameters
num_drones = 20
space_size = 1000  # Define the size of the space
min_height = 100  # Minimum height for drones
max_height = 500  # Maximum height for drones
height_offset = 30  # Height offset to deconflict routes
separation_distance = 20  # Minimum separation distance between drones and obstacles
min_velocity = 5  # Minimum velocity for drones (m/s)
max_velocity = 15  # Maximum velocity for drones (m/s)
max_iterations = 1000
max_generations = 100
pop_size = 50

# Define rectangular hexahedron obstacles
def generate_hexahedron(center, size):
    x, y, z = center
    dx, dy, dz = size
    vertices = np.array([
        [x - dx / 2, y - dy / 2, z - dz / 2],
        [x + dx / 2, y - dy / 2, z - dz / 2],
        [x + dx / 2, y + dy / 2, z - dz / 2],
        [x - dx / 2, y + dy / 2, z - dz / 2],
        [x - dx / 2, y - dy / 2, z + dz / 2],
        [x + dx / 2, y - dy / 2, z + dz / 2],
        [x + dx / 2, y + dy / 2, z + dz / 2],
        [x - dx / 2, y + dy / 2, z + dz / 2]
    ])
    return vertices

obstacle_centers = [
    [200, 200, 200],
    [500, 500, 300],
    [800, 200, 400]
]

obstacle_sizes = [
    [200, 200, 200],
    [300, 300, 300],
    [250, 250, 250]
]

hexahedrons = [generate_hexahedron(center, size) for center, size in zip(obstacle_centers, obstacle_sizes)]

# Function to check if a point is inside any obstacle
def is_inside_obstacle(point, obstacles):
    for hex in obstacles:
        if (hex[:, 0].min() <= point[0] <= hex[:, 0].max() and
            hex[:, 1].min() <= point[1] <= hex[:, 1].max() and
            hex[:, 2].min() <= point[2] <= hex[:, 2].max()):
            return True
    return False

# Function to check if a line segment intersects with any obstacle
def is_collision(p1, p2, obstacles):
    for hex in obstacles:
        # Check if either endpoint is inside the obstacle
        if is_inside_obstacle(p1, [hex]) or is_inside_obstacle(p2, [hex]):
            return True

        # Check for intersection with each face of the hexahedron
        faces = [
            [hex[0], hex[1], hex[5], hex[4]],
            [hex[1], hex[2], hex[6], hex[5]],
            [hex[2], hex[3], hex[7], hex[6]],
            [hex[3], hex[0], hex[4], hex[7]],
            [hex[0], hex[1], hex[2], hex[3]],
            [hex[4], hex[5], hex[6, hex[7]]],
        ]
        for face in faces:
            if line_intersects_face(p1, p2, face):
                return True
    return False

# Helper function to check if a line segment intersects with a face
def line_intersects_face(p1, p2, face):
    def ccw(A, B, C):
        return (C[2] - A[2]) * (B[0] - A[0]) > (B[2] - A[2]) * (C[0] - A[0])
    
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    for i in range(4):
        if intersect(p1, p2, face[i], face[(i + 1) % 4]):
            return True
    return False

# Function to check if two line segments intersect
def lines_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[2] - A[2]) * (B[0] - A[0]) > (B[2] - A[2]) * (C[0] - A[0])
    
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    return intersect(p1, p2, q1, q2)

# Generate random start and end points for drones, avoiding obstacles
def generate_random_points(num_points, space_size, min_height, max_height, obstacles):
    points = []
    while len(points) < num_points:
        point = np.random.rand(3) * [space_size, space_size, (max_height - min_height)] + [0, 0, min_height]
        if not is_inside_obstacle(point, obstacles):
            points.append(point)
    return np.array(points)

start_points = generate_random_points(num_drones, space_size, min_height, max_height, hexahedrons)
end_points = generate_random_points(num_drones, space_size, min_height, max_height, hexahedrons)

# Assign random velocities to drones
velocities = np.random.uniform(min_velocity, max_velocity, num_drones)

# DPRRT* Algorithm Implementation

class Node:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent
        self.cost = 0

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def sample_point(bias_goal=False, goal=None):
    if bias_goal and random.random() < 0.2:  # 20% of the time, sample the goal
        return goal
    return (random.uniform(0, space_size), random.uniform(0, space_size), random.uniform(min_height, max_height))

def nearest_node(nodes, point):
    return min(nodes, key=lambda node: dist(node.point, point))

def is_collision_with_paths(p1, p2, paths_taken):
    for path in paths_taken:
        for i in range(len(path) - 1):
            if lines_intersect(p1, p2, path[i], path[i + 1]):
                return True
    return False

def rewire(nodes, new_node, neighbor_radius=15):
    for node in nodes:
        if node != new_node and dist(node.point, new_node.point) < neighbor_radius:
            potential_cost = new_node.cost + dist(new_node.point, node.point)
            if potential_cost < node.cost:
                node.parent = new_node
                node.cost = potential_cost

def dprrt_star(start, goal, obstacles, space_size, paths_taken, max_iter=1000):
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for iter_count in range(max_iter):
        rand_point = sample_point(bias_goal=True, goal=goal)
        nearest = nearest_node(nodes, rand_point)
        direction = np.array(rand_point) - np.array(nearest.point)
        direction = direction / np.linalg.norm(direction) * 10
        new_point = tuple(np.array(nearest.point) + direction)

        if (0 <= new_point[0] <= space_size and 0 <= new_point[1] <= space_size and
            min_height <= new_point[2] <= max_height and not is_collision(nearest.point, new_point, obstacles) and not is_collision_with_paths(nearest.point, new_point, paths_taken)):
            
            new_node = Node(new_point, nearest)
            new_node.cost = nearest.cost + dist(nearest.point, new_point)
            nodes.append(new_node)

            rewire(nodes, new_node)

            if dist(new_point, goal) < 10 and not is_collision(new_point, goal, obstacles) and not is_collision_with_paths(new_point, goal, paths_taken):
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + dist(new_node.point, goal)
                nodes.append(goal_node)
                break
        if iter_count % 100 == 0:
            print(f"Iteration {iter_count}: Current node count = {len(nodes)}")

    path = []
    node = goal_node
    while node is not None:
        path.append(node.point)
        node = node.parent
    if len(path) == 1:
        print(f"Failed to find path from {start} to {goal}")
        return []
    return path[::-1]

# HAPF Algorithm Implementation for local anti-collision planning

def hapf(current_pos, goal_pos, obstacles):
    force = np.zeros(3)
    K_att = 1.0  # Attraction coefficient
    K_rep = 100.0  # Repulsion coefficient
    d0 = 50.0  # Distance threshold for repulsion
    
    # Attractive force towards goal
    force += K_att * (goal_pos - current_pos)
    
    # Repulsive force from obstacles
    for obs in obstacles:
        for i in range(8):
            obs_pos = obs[i]
            dist_to_obs = np.linalg.norm(current_pos - obs_pos)
            if dist_to_obs < d0:
                force += K_rep * (1.0 / dist_to_obs - 1.0 / d0) * (current_pos - obs_pos) / (dist_to_obs**3)
    
    # Compute next position
    next_pos = current_pos + force / np.linalg.norm(force)
    return next_pos

# Function to adjust path based on HAPF
def adjust_path_with_hapf(path, obstacles):
    adjusted_path = [path[0]]
    for i in range(1, len(path) - 1):
        adjusted_path.append(hapf(np.array(path[i]), np.array(path[i + 1]), obstacles))
    adjusted_path.append(path[-1])
    return adjusted_path

# Function to optimize routes with scheduling and speed adjustment strategies
def fitness(paths, velocities, ETDs):
    total_distance = sum([sum([dist(paths[i][j], paths[i][j + 1]) for j in range(len(paths[i]) - 1)]) for i in range(len(paths))])
    total_time = sum([sum([dist(paths[i][j], paths[i][j + 1]) / velocities[i] for j in range(len(paths[i]) - 1)]) for i in range(len(paths))])
    total_schedule = sum(ETDs)
    return total_distance + total_time + total_schedule

def initialize_population(pop_size, num_drones):
    population = []
    for _ in range(pop_size):
        ETDs = np.random.uniform(0, 10, num_drones)
        paths = [generate_random_points(10, space_size, min_height, max_height, hexahedrons).tolist() for _ in range(num_drones)]
        population.append((paths, velocities.copy(), ETDs))
    return population

def crossover(parent1, parent2):
    idx = random.randint(0, len(parent1[0]) - 1)
    new_paths = parent1[0][:idx] + parent2[0][idx:]
    new_velocities = np.concatenate((parent1[1][:idx], parent2[1][idx:]))
    new_ETDs = np.concatenate((parent1[2][:idx], parent2[2][idx:]))
    return (new_paths, new_velocities, new_ETDs)

def mutate(individual):
    idx = random.randint(0, len(individual[0]) - 1)
    individual[0][idx] = generate_random_points(10, space_size, min_height, max_height, hexahedrons).tolist()
    individual[1][idx] = np.random.uniform(min_velocity, max_velocity)
    individual[2][idx] = np.random.uniform(0, 10)
    return individual

def select_population(population, fitnesses, pop_size):
    selected_indices = np.argsort(fitnesses)[:pop_size]
    return [population[i] for i in selected_indices]

def optimize_routes(pop_size=50, max_generations=100):
    population = initialize_population(pop_size, num_drones)
    best_fitness = float('inf')
    best_individual = None

    for generation in range(max_generations):
        fitnesses = [fitness(ind[0], ind[1], ind[2]) for ind in population]
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_fitness:
            best_fitness = fitnesses[best_idx]
            best_individual = population[best_idx]

        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = random.sample(population, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])

        population = select_population(population + new_population, fitnesses + [fitness(ind[0], ind[1], ind[2]) for ind in new_population], pop_size)
        print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

    return best_individual

# Calculate paths using DPRRT* and HAPF
def calculate_paths():
    global paths, velocities, ETDs
    best_individual = optimize_routes()
    paths, velocities, ETDs = best_individual
    print("Path calculation completed.")

# Deconflict routes by adjusting heights and speeds
def deconflict_routes(paths, velocities):
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            for t in range(min(len(paths[i]), len(paths[j]))):
                if np.linalg.norm(np.array(paths[i][t]) - np.array(paths[j][t])) < separation_distance:
                    # Adjust height for drone j
                    for k in range(t, len(paths[j])):
                        paths[j][k] = (paths[j][k][0], paths[j][k][1], paths[j][k][2] + height_offset)
                    # Adjust speed for drone j
                    velocities[j] *= 0.9
    return paths, velocities

# Generate waypoints for each drone
def generate_waypoints(path, num_waypoints=10):
    if len(path) < num_waypoints:
        return path
    indices = np.linspace(0, len(path) - 1, num_waypoints).astype(int)
    return [path[i] for i in indices]

def main_app():
    global waypoints
    # Calculate waypoints for each drone
    waypoints = [generate_waypoints(paths[i]) for i in range(len(paths))]

    # Output waypoints to console
    for i in range(len(waypoints)):
        print(f"Drone {i+1} Waypoints:")
        for wp in waypoints[i]:
            print(wp)
        print("\n")

    # Create a Matplotlib figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(min_height, max_height)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')

    # Plot rectangular hexahedron obstacles
    for hex in hexahedrons:
        ax.add_collection3d(Poly3DCollection([hex[[0, 1, 5, 4]], hex[[1, 2, 6, 5]], hex[[2, 3, 7, 6]], hex[[3, 0, 4, 7]], hex[[0, 1, 2, 3]], hex[[4, 5, 6, 7]]], color='gray', alpha=0.5))

    # Initial scatter points for start and end points
    start_scatters = [ax.scatter(start_points[i, 0], start_points[i, 1], start_points[i, 2], marker='o', color='blue') for i in range(num_drones)]
    end_scatters = [ax.scatter(end_points[i, 0], end_points[i, 1], end_points[i, 2], marker='x', color='red') for i in range(num_drones)]

    # Text annotations for start and end points
    start_texts = [ax.text(start_points[i, 0], start_points[i, 1], start_points[i, 2], f'S {i+1}', fontsize=9, color='blue') for i in range(num_drones)]
    end_texts = [ax.text(end_points[i, 0], end_points[i, 1], end_points[i, 2], f'E {i+1}', fontsize=9, color='red') for i in range(num_drones)]

    # Line objects for paths
    for i in range(len(paths)):
        path = np.array(paths[i])
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], label=f'Drone {i+1}')

    plt.legend(loc='upper right')
    plt.show()

# Create Tkinter GUI window
root = Tk()
root.title("Drone Route Optimization")

# Create a splash screen
splash = Toplevel(root)
splash.title("Loading")
Label(splash, text="Loading...", font=("Helvetica", 16)).pack(padx=20, pady=20)

# Hide the root window initially
root.withdraw()

# Check if the path calculation is done and then run the main app
def check_calculation():
    calculate_paths()
    main_app()

root.after(100, check_calculation)

# Run the Tkinter event loop
root.mainloop()
