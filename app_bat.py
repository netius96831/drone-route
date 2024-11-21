import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product, combinations
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

class Drone:
    def __init__(self, start, end, ceiling, floor, min_velocity, max_velocity):
        self.start = np.array(start)
        self.end = np.array(end)
        self.ceiling = ceiling
        self.floor = floor
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.route = [self.start, self.end]
        self.height = random.uniform(self.floor, self.ceiling)
        
    def generate_waypoints(self, num_waypoints=10):
        waypoints = []
        for i in range(1, num_waypoints + 1):
            waypoint = self.start + (self.end - self.start) * i / (num_waypoints + 1)
            waypoints.append(waypoint)
        self.route = [self.start] + waypoints + [self.end]

class Obstacle:
    def __init__(self, center, size):
        self.vertices = self.generate_hexahedron(center, size)
        self.min_corner = np.min(self.vertices, axis=0)
        self.max_corner = np.max(self.vertices, axis=0)
    
    def generate_hexahedron(self, center, size):
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
    
    def contains_point(self, point):
        return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)
    
    def intersects_route(self, route):
        for point in route:
            if self.contains_point(point):
                return True
        return False

def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) / 
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size=3)
    v = np.random.normal(0, 1, size=3)
    step = u / abs(v) ** (1 / Lambda)
    return step

def adjust_route(drone, obstacles, num_waypoints=10, alpha=0.9, gamma=0.9, Lambda=1.5):
    adjusted_route = [drone.start]
    for i in range(1, len(drone.route) - 1):
        waypoint = drone.route[i]
        adjusted = False
        for obstacle in obstacles:
            while obstacle.contains_point(waypoint):
                step = levy_flight(Lambda) * alpha
                waypoint += step
                adjusted = True
            if adjusted:
                break
        adjusted_route.append(waypoint)
    adjusted_route.append(drone.end)
    drone.route = adjusted_route

def plot_routes(drones, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for drone in drones:
        route = np.array(drone.route)
        ax.plot(route[:, 0], route[:, 1], route[:, 2], label=f'Drone {drones.index(drone) + 1}')
        ax.scatter(route[:, 0], route[:, 1], route[:, 2])
        ax.text(route[0, 0], route[0, 1], route[0, 2], f'Start {drones.index(drone) + 1}')
        ax.text(route[-1, 0], route[-1, 1], route[-1, 2], f'End {drones.index(drone) + 1}')
    
    for obstacle in obstacles:
        vertices = obstacle.vertices
        faces = [[vertices[j] for j in [0, 1, 2, 3]],
                 [vertices[j] for j in [4, 5, 6, 7]], 
                 [vertices[j] for j in [0, 1, 5, 4]], 
                 [vertices[j] for j in [2, 3, 7, 6]], 
                 [vertices[j] for j in [0, 3, 7, 4]], 
                 [vertices[j] for j in [1, 2, 6, 5]]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors='red', linewidths=1, edgecolors='r', alpha=.25))
    
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude')
    plt.legend()
    plt.show()

def is_point_in_obstacle(point, obstacles):
    for obstacle in obstacles:
        if obstacle.contains_point(point):
            return True
    return False

# Example usage
num_drones = 20
drones = []
obstacles = []

# Define rectangular hexahedron obstacles with smaller sizes
obstacle_centers = [
    [500, 500, 100],
    [300, 200, 100],
    [300, 500, 200]
]

obstacle_sizes = [
    [50, 50, 50],
    [170, 170, 270],
    [60, 60, 200]
]

for center, size in zip(obstacle_centers, obstacle_sizes):
    obstacle = Obstacle(center, size)
    obstacles.append(obstacle)

# Create random drones ensuring start and end points are not inside obstacles
for i in range(num_drones):
    while True:
        start = (random.uniform(0, 1000), random.uniform(0, 1000), random.uniform(0, 400))
        if not is_point_in_obstacle(start, obstacles):
            break
    while True:
        end = (random.uniform(0, 1000), random.uniform(0, 1000), random.uniform(0, 400))
        if not is_point_in_obstacle(end, obstacles):
            break
    ceiling = random.uniform(100, 200)
    floor = random.uniform(50, 99)
    min_velocity = random.uniform(10, 20)
    max_velocity = random.uniform(20, 30)
    drone = Drone(start, end, ceiling, floor, min_velocity, max_velocity)
    drone.generate_waypoints()
    drones.append(drone)

# Adjust drone routes to avoid obstacles using improved bat algorithm
for drone in drones:
    adjust_route(drone, obstacles)

# Plot the routes and obstacles
plot_routes(drones, obstacles)
