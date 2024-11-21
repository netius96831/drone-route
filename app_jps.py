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
height_offset = 25  # Height offset to deconflict routes
separation_distance = 10  # Minimum separation distance between drones and obstacles
min_velocity = 5  # Minimum velocity for drones (m/s)
max_velocity = 15  # Maximum velocity for drones (m/s)
max_iterations = 1000

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
            [hex[4], hex[5], hex[6], hex[7]],
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

# Function to calculate the Euclidean distance between two points
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Voxel JPS Algorithm for initial path planning
def voxel_jps(start, goal, obstacles):
    # Implement a basic voxel JPS algorithm to generate an initial path
    path = [start]
    current = start
    while np.linalg.norm(np.array(current) - np.array(goal)) > 10:
        direction = np.array(goal) - np.array(current)
        step = direction / np.linalg.norm(direction) * 10
        next_point = tuple(np.array(current) + step)
        if is_collision(current, next_point, obstacles):
            # Adjust the direction to avoid obstacles
            for i in range(-10, 11, 1):
                adjusted_next_point = tuple(np.array(current) + step + [i, i, i])
                if not is_collision(current, adjusted_next_point, obstacles):
                    next_point = adjusted_next_point
                    break
        path.append(next_point)
        current = next_point
    path.append(goal)
    return path

# Function to optimize paths
def optimize_path(path):
    # Apply de-diagonalization, reconstruction, and smoothness techniques
    optimized_path = [path[0]]
    for i in range(1, len(path) - 1):
        mid_point = (np.array(path[i - 1]) + np.array(path[i + 1])) / 2
        optimized_path.append(tuple(mid_point))
    optimized_path.append(path[-1])
    return optimized_path

# MDP-based Dynamic Collision Resolution
def mdp_collision_resolution(path, dynamic_threats):
    # Implement MDP-based dynamic collision resolution
    resolved_path = [path[0]]
    for i in range(1, len(path) - 1):
        current_state = path[i]
        best_action = None
        best_value = float('-inf')
        for action in range(-10, 11):
            next_state = tuple(np.array(current_state) + [action, action, action])
            value = -np.linalg.norm(np.array(next_state) - np.array(dynamic_threats[i]))
            if value > best_value:
                best_value = value
                best_action = next_state
        resolved_path.append(best_action)
    resolved_path.append(path[-1])
    return resolved_path

# Function to calculate paths
def calculate_paths():
    global paths, velocities
    paths = []
    for i in range(num_drones):
        start = tuple(start_points[i])
        goal = tuple(end_points[i])
        path = voxel_jps(start, goal, hexahedrons)
        path = optimize_path(path)
        paths.append(path)
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
        print(f"Drone {i + 1} Waypoints:")  # Corrected drone ID
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
    start_texts = [ax.text(start_points[i, 0], start_points[i, 1], start_points[i, 2], f'S {i + 1}', fontsize=9, color='blue') for i in range(num_drones)]  # Corrected drone ID
    end_texts = [ax.text(end_points[i, 0], end_points[i, 1], end_points[i, 2], f'E {i + 1}', fontsize=9, color='red') for i in range(num_drones)]  # Corrected drone ID

    # Line objects for paths
    for i in range(len(paths)):
        path = np.array(paths[i])
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], label=f'Drone {i + 1}')  # Corrected drone ID

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
