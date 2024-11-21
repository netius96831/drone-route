import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
NUM_DRONES = 20
SAFE_DISTANCE = 20  # Minimum horizontal distance between drones in meters
ALTITUDE_INCREMENT = 30  # Height difference to avoid clashes
MAX_ALTITUDE = 500  # Maximum altitude for the drones

# Generate random start and end points for drones (longitude, latitude, altitude)
np.random.seed(42)  # For reproducibility
start_points = np.random.rand(NUM_DRONES, 3) * np.array([360, 180, MAX_ALTITUDE])
end_points = np.random.rand(NUM_DRONES, 3) * np.array([360, 180, MAX_ALTITUDE])

# Assign altitudes to ensure deconfliction in the vertical plane
for i in range(NUM_DRONES):
    start_points[i, 2] = 100 + i * ALTITUDE_INCREMENT
    end_points[i, 2] = 100 + i * ALTITUDE_INCREMENT

# Generate waypoints avoiding obstacles and ensuring deconfliction
def generate_waypoints(start, end, safe_distance):
    waypoints = [start]
    direction = (end - start) / np.linalg.norm(end - start)
    distance = np.linalg.norm(end - start)
    num_steps = int(distance / safe_distance)
    for i in range(1, num_steps):
        waypoint = start + i * safe_distance * direction
        waypoints.append(waypoint)
    waypoints.append(end)
    return np.array(waypoints)

# Generate routes for each drone
routes = []
for i in range(NUM_DRONES):
    waypoints = generate_waypoints(start_points[i], end_points[i], SAFE_DISTANCE)
    routes.append(waypoints)

# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot start and end points
for i in range(NUM_DRONES):
    ax.scatter(start_points[i, 0], start_points[i, 1], start_points[i, 2], c='g', marker='o')  # Start points in green
    ax.scatter(end_points[i, 0], end_points[i, 1], end_points[i, 2], c='r', marker='o')  # End points in red

# Plot routes and waypoints
for i in range(NUM_DRONES):
    route = routes[i]
    ax.plot(route[:, 0], route[:, 1], route[:, 2], label=f'Drone {i+1}')
    for waypoint in route:
        ax.scatter(waypoint[0], waypoint[1], waypoint[2], c='b', marker='x')

# Define rectangular hexahedron obstacles (as an example)
obstacles = [
    # Each obstacle is defined by its corner points (8 points for a rectangular hexahedron)
    np.array([[100, 50, 200], [200, 50, 200], [200, 150, 200], [100, 150, 200],
              [100, 50, 300], [200, 50, 300], [200, 150, 300], [100, 150, 300]]),
    np.array([[250, 100, 300], [350, 100, 300], [350, 200, 300], [250, 200, 300],
              [250, 100, 400], [350, 100, 400], [350, 200, 400], [250, 200, 400]])
]

# Plot rectangular hexahedron obstacles
def plot_hexahedron(ax, hexahedron):
    # Define the vertices that form the 12 faces of the rectangular hexahedron
    vertices = [
        [hexahedron[j] for j in [0, 1, 2, 3]],
        [hexahedron[j] for j in [4, 5, 6, 7]],
        [hexahedron[j] for j in [0, 1, 5, 4]],
        [hexahedron[j] for j in [2, 3, 7, 6]],
        [hexahedron[j] for j in [1, 2, 6, 5]],
        [hexahedron[j] for j in [4, 7, 3, 0]]
    ]
    poly3d = Poly3DCollection(vertices, alpha=0.3, facecolors='r')
    ax.add_collection3d(poly3d)

# Plot each obstacle
for hexahedron in obstacles:
    plot_hexahedron(ax, hexahedron)

# Ensure waypoints avoid obstacles
def avoid_obstacles(route, obstacles, safe_distance):
    for obs in obstacles:
        min_corner = np.min(obs, axis=0)
        max_corner = np.max(obs, axis=0)
        for waypoint in route:
            if all(min_corner - safe_distance <= waypoint) and all(waypoint <= max_corner + safe_distance):
                direction = (waypoint - (min_corner + max_corner) / 2) / np.linalg.norm(waypoint - (min_corner + max_corner) / 2)
                waypoint += direction * safe_distance
    return route

# Adjust routes for obstacle avoidance
for i in range(NUM_DRONES):
    routes[i] = avoid_obstacles(routes[i], obstacles, SAFE_DISTANCE)
    ax.plot(routes[i][:, 0], routes[i][:, 1], routes[i][:, 2], linestyle='dashed')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')
plt.legend()
plt.title('3D Drone Route Optimization and Deconfliction with Rectangular Hexahedron Obstacles')
plt.show()
