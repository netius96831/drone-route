import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev

# Parameters
num_drones = 20
space_size = 1000  # Define the size of the space
min_height = 100  # Minimum height for drones
max_height = 500  # Maximum height for drones
height_offset = 30  # Height offset to deconflict routes
separation_distance = 20  # Minimum separation distance between drones
min_velocity = 5  # Minimum velocity for drones (m/s)
max_velocity = 15  # Maximum velocity for drones (m/s)
time_step = 1  # Time step for the animation

# Generate random start and end points for drones
start_points = np.random.rand(num_drones, 3) * [space_size, space_size, (max_height - min_height)] + [0, 0, min_height]
end_points = np.random.rand(num_drones, 3) * [space_size, space_size, (max_height - min_height)] + [0, 0, min_height]

# Assign random velocities to drones
velocities = np.random.uniform(min_velocity, max_velocity, num_drones)

# Define curved obstacles as splines
def generate_curve(points):
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0)
    u_fine = np.linspace(0, 1, 100)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine, z_fine)).T

curved_obstacles = [
    generate_curve(np.array([[200, 200, 100], [250, 250, 150], [300, 300, 200], [350, 350, 250]])),
    generate_curve(np.array([[500, 500, 200], [550, 450, 250], [600, 400, 300], [650, 350, 350]])),
    generate_curve(np.array([[700, 300, 300], [750, 350, 350], [800, 400, 400], [850, 450, 450]]))
]

# RRT algorithm implementation
def rrt(start, goal, curved_obstacles, space_size, max_iter=1000):
    class Node:
        def __init__(self, point, parent=None):
            self.point = point
            self.parent = parent

    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def sample_point(bias_goal=False):
        if bias_goal and random.random() < 0.2:  # 20% of the time, sample the goal
            return goal
        return (random.uniform(0, space_size), random.uniform(0, space_size), random.uniform(min_height, max_height))

    def nearest_node(nodes, point):
        return min(nodes, key=lambda node: dist(node.point, point))

    def is_collision_with_curve(point, curve):
        for i in range(len(curve) - 1):
            p1, p2 = curve[i], curve[i + 1]
            d = np.linalg.norm(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
            if d < separation_distance:
                return True
        return False

    def is_collision(p1, p2, curved_obstacles):
        for curve in curved_obstacles:
            if is_collision_with_curve(p1, curve) or is_collision_with_curve(p2, curve):
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

        if (0 <= new_point[0] <= space_size and 0 <= new_point[1] <= space_size and
            min_height <= new_point[2] <= max_height and not is_collision(nearest.point, new_point, curved_obstacles)):
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

# Calculate paths using RRT
paths = []
for i in range(num_drones):
    start = tuple(start_points[i])
    goal = tuple(end_points[i])
    path = rrt(start, goal, curved_obstacles, space_size)
    attempts = 0
    while not path and attempts < 5:  # Retry up to 5 times if path is empty
        path = rrt(start, goal, curved_obstacles, space_size)
        attempts += 1
    paths.append(path)

# Deconflict routes by adjusting heights and speeds
def deconflict_routes(paths, velocities):
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            for t in range(min(len(paths[i]), len(paths[j]))):
                if np.linalg.norm(np.array(paths[i][t]) - np.array(paths[j][t])) < separation_distance:
                    # Adjust height for drone j
                    for k in range(t, len(paths[j])):
                        paths[j][k] = (paths[j][k][0], paths[j][k][1], paths[j][k][2] + height_offset)
                    # Adjust speed for drone j
                    velocities[j] *= 0.9
    return paths, velocities

paths, velocities = deconflict_routes(paths, velocities)

# Generate waypoints for each drone
def generate_waypoints(path, num_waypoints=10):
    if len(path) < num_waypoints:
        return path
    indices = np.linspace(0, len(path) - 1, num_waypoints).astype(int)
    return [path[i] for i in indices]

waypoints = [generate_waypoints(paths[i]) for i in range(num_drones)]

# Display waypoints for each drone
for i in range(num_drones):
    print(f"Drone {i+1} Waypoints:")
    for wp in waypoints[i]:
        print(wp)

# Visualize the routes in 3D with animation
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)
ax.set_zlim(min_height, max_height)

# Plot curved obstacles
for curve in curved_obstacles:
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='black')

# Initial scatter points for start and end points
start_scatters = [ax.scatter(start_points[i, 0], start_points[i, 1], start_points[i, 2], marker='o', color='blue') for i in range(num_drones)]
end_scatters = [ax.scatter(end_points[i, 0], end_points[i, 1], end_points[i, 2], marker='x', color='red') for i in range(num_drones)]

# Text annotations for start and end points
start_texts = [ax.text(start_points[i, 0], start_points[i, 1], start_points[i, 2], f'S {i+1}', fontsize=9, color='blue') for i in range(num_drones)]
end_texts = [ax.text(end_points[i, 0], end_points[i, 1], end_points[i, 2], f'E {i+1}', fontsize=9, color='red') for i in range(num_drones)]

# Line objects for paths
path_lines = [ax.plot([], [], [], label=f'Drone {i+1}')[0] for i in range(num_drones)]

# Point objects for drones
drone_points = [ax.scatter([], [], [], marker='o') for i in range(num_drones)]

def init():
    for line in path_lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for point in drone_points:
        point._offsets3d = ([], [], [])
    return path_lines + drone_points

def update(frame):
    for i in range(num_drones):
        path = paths[i]
        if len(path) > 0:
            path_array = np.array(path)
            path_lines[i].set_data(path_array[:, 0], path_array[:, 1])
            path_lines[i].set_3d_properties(path_array[:, 2])
            # Update drone position based on velocity
            index = min(frame, len(path_array) - 1)
            drone_points[i]._offsets3d = (path_array[index:index+1, 0], path_array[index:index+1, 1], path_array[index:index+1, 2])
    return path_lines + drone_points

# Animation
ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=False, repeat=True)

ax.set_title('Drone Route Optimization')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')
ax.legend()
plt.show()
