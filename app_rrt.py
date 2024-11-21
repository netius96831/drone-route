import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
num_drones = 20
space_size = 1000  # Define the size of the space
min_height = 100  # Minimum height for drones
max_height = 500  # Maximum height for drones
height_offset = 30  # Height offset to deconflict routes
separation_distance = 20  # Minimum separation distance between drones

# Generate random start and end points for drones
start_points = np.random.rand(num_drones, 2) * space_size
end_points = np.random.rand(num_drones, 2) * space_size

# Assign random heights to drones
heights = np.linspace(min_height, max_height, num_drones)

# RRT algorithm implementation
def rrt(start, goal, space_size, max_iter=1000):
    class Node:
        def __init__(self, point, parent=None):
            self.point = point
            self.parent = parent

    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def sample_point(bias_goal=False):
        if bias_goal and random.random() < 0.2:  # 20% of the time, sample the goal
            return goal
        return (random.uniform(0, space_size), random.uniform(0, space_size))

    def nearest_node(nodes, point):
        return min(nodes, key=lambda node: dist(node.point, point))

    def is_collision(p1, p2, obstacles):
        for obs in obstacles:
            if np.linalg.norm(np.array(obs) - np.array(p1)) < separation_distance or np.linalg.norm(np.array(obs) - np.array(p2)) < separation_distance:
                return True
        return False

    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for _ in range(max_iter):
        rand_point = sample_point(bias_goal=True)
        nearest = nearest_node(nodes, rand_point)
        direction = np.array(rand_point) - np.array(nearest.point)
        direction = direction / np.linalg.norm(direction) * 10
        new_point = tuple(np.array(nearest.point) + direction)

        if 0 <= new_point[0] <= space_size and 0 <= new_point[1] <= space_size and not is_collision(nearest.point, new_point, obstacles):
            new_node = Node(new_point, nearest)
            nodes.append(new_node)
            if dist(new_point, goal) < 10:
                goal_node.parent = new_node
                nodes.append(goal_node)
                break

    path = []
    node = goal_node
    while node is not None:
        path.append(node.point)
        node = node.parent
    return path[::-1]

# Calculate paths using RRT with retry mechanism
obstacles = []
paths = []
for i in range(num_drones):
    start = tuple(start_points[i])
    goal = tuple(end_points[i])
    path = rrt(start, goal, space_size)
    attempts = 0
    while not path and attempts < 5:  # Retry up to 5 times if path is empty
        path = rrt(start, goal, space_size)
        attempts += 1
    paths.append(path)

# Deconflict routes by adjusting heights
def deconflict_routes(paths, heights):
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            if any(np.linalg.norm(np.array(p1) - np.array(p2)) < separation_distance for p1 in paths[i] for p2 in paths[j]):
                heights[j] += height_offset
    return heights

heights = deconflict_routes(paths, heights)

# Visualize the routes
plt.figure(figsize=(10, 10))
plt.xlim(0, space_size)
plt.ylim(0, space_size)

for i in range(num_drones):
    path = paths[i]
    if path:
        plt.plot([p[0] for p in path], [p[1] for p in path], label=f'Drone {i+1}')
    plt.scatter(start_points[i, 0], start_points[i, 1], marker='o', color='blue', label=f'Start {i+1}')
    plt.scatter(end_points[i, 0], end_points[i, 1], marker='x', color='red', label=f'End {i+1}')
    plt.text(start_points[i, 0], start_points[i, 1], f'{i+1} ({heights[i]:.1f}m)', fontsize=9, color='blue')
    plt.text(end_points[i, 0], end_points[i, 1], f'{i+1} ({heights[i]:.1f}m)', fontsize=9, color='red')

plt.title('Drone Route Optimization')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()

# Generate waypoints for each drone
def generate_waypoints(path, num_waypoints=10):
    if len(path) < num_waypoints:
        return path
    indices = np.linspace(0, len(path) - 1, num_waypoints).astype(int)
    return [path[i] for i in indices]

waypoints = [generate_waypoints(paths[i]) for i in range(num_drones)]

# Display waypoints and altitudes for each drone
for i in range(num_drones):
    print(f"Drone {i+1} Waypoints and Altitude ({heights[i]:.1f}m):")
    for wp in waypoints[i]:
        print(wp)
