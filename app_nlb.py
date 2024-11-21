import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from geopy.distance import geodesic
import random

# Constants
NUM_DRONES = 20
SAFETY_RADIUS = 20  # Safety radius in meters
ALTITUDE_OFFSET = 30  # Altitude offset for deconfliction
TIME_STEP = 0.1  # Simulation time step

# Drone Class
class Drone:
    def __init__(self, start_latlon, end_latlon, ceiling, floor, min_vel, max_vel):
        self.start_latlon = np.array(start_latlon)
        self.end_latlon = np.array(end_latlon)
        self.ceiling = ceiling
        self.floor = floor
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.path = [self.start_latlon]
        self.altitude = random.uniform(floor, ceiling)

    def generate_path(self):
        # Create a simple path using start and end lat/lon points
        self.path = [self.start_latlon, self.end_latlon]

    def add_waypoints(self, num_waypoints=10):
        waypoints = []
        for i in range(1, num_waypoints + 1):
            waypoint = self.start_latlon + (self.end_latlon - self.start_latlon) * (i / num_waypoints)
            waypoint = np.append(waypoint, self.altitude)  # Add altitude
            waypoints.append(waypoint)
        self.path = waypoints

# Obstacle Class
class Obstacle:
    def __init__(self, min_latlon, max_latlon, min_altitude, max_altitude):
        self.min_latlon = np.array(min_latlon)
        self.max_latlon = np.array(max_latlon)
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

    def is_collision(self, point):
        latlon, altitude = point[:2], point[2]
        # Check if the lat/lon point is inside the obstacle bounds
        in_latlon = np.all(self.min_latlon <= latlon) and np.all(latlon <= self.max_latlon)
        in_altitude = self.min_altitude <= altitude <= self.max_altitude
        return in_latlon and in_altitude

    def plot_obstacle(self, ax):
        # Plot the obstacle as a rectangular prism in the lat/lon/alt space
        lats = [self.min_latlon[0], self.max_latlon[0]]
        lons = [self.min_latlon[1], self.max_latlon[1]]
        alts = [self.min_altitude, self.max_altitude]
        
        # Create the vertices of the rectangular hexahedron
        for lat in lats:
            for lon in lons:
                for alt in alts:
                    ax.scatter(lat, lon, alt, color='red')

        # Connect the vertices
        for lat in lats:
            ax.plot([lat, lat], [self.min_latlon[1], self.max_latlon[1]], [self.min_altitude, self.min_altitude], color='red')
            ax.plot([lat, lat], [self.min_latlon[1], self.max_latlon[1]], [self.max_altitude, self.max_altitude], color='red')
        
        for lon in lons:
            ax.plot([self.min_latlon[0], self.max_latlon[0]], [lon, lon], [self.min_altitude, self.min_altitude], color='red')
            ax.plot([self.min_latlon[0], self.max_latlon[0]], [lon, lon], [self.max_altitude, self.max_altitude], color='red')
        
        for alt in alts:
            ax.plot([self.min_latlon[0], self.max_latlon[0]], [self.min_latlon[1], self.min_latlon[1]], [alt, alt], color='red')
            ax.plot([self.min_latlon[0], self.max_latlon[0]], [self.max_latlon[1], self.max_latlon[1]], [alt, alt], color='red')

# Simulation Class
class DroneSimulation:
    def __init__(self, num_drones=NUM_DRONES):
        self.drones = []
        self.obstacles = []
        self.generate_drones(num_drones)
        self.generate_obstacles()

    def generate_drones(self, num_drones):
        for i in range(num_drones):
            start_latlon = [random.uniform(35.0, 35.1), random.uniform(-120.0, -119.9)]
            end_latlon = [random.uniform(35.2, 35.3), random.uniform(-119.8, -119.7)]
            ceiling = random.uniform(80, 100)
            floor = random.uniform(20, 40)
            min_vel = random.uniform(5, 10)
            max_vel = random.uniform(10, 20)
            drone = Drone(start_latlon, end_latlon, ceiling, floor, min_vel, max_vel)
            drone.generate_path()
            drone.add_waypoints()
            self.drones.append(drone)

    def generate_obstacles(self):
        for _ in range(5):  # Create 5 larger obstacles
            min_latlon = [random.uniform(35.05, 35.15), random.uniform(-119.95, -119.85)]
            max_latlon = [min_latlon[0] + random.uniform(0.02, 0.05), min_latlon[1] + random.uniform(0.02, 0.05)]
            min_altitude = random.uniform(10, 50)
            max_altitude = min_altitude + random.uniform(40, 60)  # Larger altitude range
            obstacle = Obstacle(min_latlon, max_latlon, min_altitude, max_altitude)
            self.obstacles.append(obstacle)

    def check_collisions(self, drone):
        for waypoint in drone.path:
            for obstacle in self.obstacles:
                if obstacle.is_collision(waypoint):
                    return True
        return False

    def deconflict_paths(self):
        for i, drone in enumerate(self.drones):
            for j, other_drone in enumerate(self.drones):
                if i != j:
                    if geodesic(drone.path[-1][:2], other_drone.path[-1][:2]).meters < SAFETY_RADIUS:
                        result = self.optimize_deconflict(drone, other_drone)
                        drone.altitude = result['altitude']
            if self.check_collisions(drone):
                drone.altitude += ALTITUDE_OFFSET

    def optimize_deconflict(self, drone, other_drone):
        def objective(x):
            return np.sum(np.square(x - drone.altitude))

        def constraint(x):
            return geodesic(drone.end_latlon, other_drone.end_latlon).meters - (SAFETY_RADIUS + x[0] - other_drone.altitude)

        cons = {'type': 'ineq', 'fun': constraint}
        initial_guess = [drone.altitude]
        result = minimize(objective, initial_guess, constraints=cons, method='SLSQP')
        return {'altitude': result.x[0]}

    def plot_paths(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for drone in self.drones:
            path = np.array(drone.path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], label=f"Drone {self.drones.index(drone) + 1}")
            ax.scatter(path[:, 0], path[:, 1], path[:, 2])
            # Add labels for start and end points
            ax.text(path[0, 0], path[0, 1], path[0, 2], f'Start {self.drones.index(drone) + 1}', color='green')
            ax.text(path[-1, 0], path[-1, 1], path[-1, 2], f'End {self.drones.index(drone) + 1}', color='red')

        for obstacle in self.obstacles:
            obstacle.plot_obstacle(ax)
        
        ax.legend()
        ax.set_title("Drone Path Optimization with Deconfliction and Larger Obstacles")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
        ax.set_zlabel("Altitude")
        ax.grid(True)
        plt.show()

    def run_simulation(self):
        self.deconflict_paths()
        self.plot_paths()

# Run the simulation
simulation = DroneSimulation()
simulation.run_simulation()
