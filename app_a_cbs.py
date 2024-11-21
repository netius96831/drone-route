import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import heapq

# Constants
NO_GO_RADIUS = 20  # Safety radius in meters
ALTITUDE_STEP = 30  # Altitude separation step in meters
GRID_SIZE = 100  # Size of the grid (100x100x100 units)
OBSTACLE_COUNT = 10
MAX_ADJUSTMENTS = 10  # Maximum number of adjustments to avoid infinite loops

# UAV Class
class UAV:
    def __init__(self, uav_id, start, end, min_altitude, max_altitude, min_velocity, max_velocity):
        self.id = uav_id
        self.start = start
        self.end = end
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.route = []

    def generate_route(self, graph):
        # Ensure that the start and end positions are valid before pathfinding
        if self.start in graph and self.end in graph:
            self.route = sparse_a_star(graph, self.start, self.end)
        else:
            self.route = []  # Set an empty route if positions are invalid

# Sparse A* Algorithm
def sparse_a_star(graph, start, end):
    if start not in graph or end not in graph:
        return []  # Return empty route if start or end is not in the graph
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == end:
            return reconstruct_path(came_from, current)
        
        # Ensure the current node exists in the graph before accessing neighbors
        if current not in graph:
            continue
        
        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
    return []  # Return an empty route if no path is found

def heuristic(a, b):
    # Manhattan distance in 3D space
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Generate random obstacles as rectangular hexahedrons in 3D space
def generate_obstacles(grid_size, count):
    obstacles = []
    for _ in range(count):
        min_corner = np.random.randint(0, grid_size-10, size=3)  # Ensure some size for the obstacle
        max_corner = min_corner + np.random.randint(5, 20, size=3)  # Ensure the obstacle has a volume
        obstacles.append((tuple(min_corner), tuple(max_corner)))
    return obstacles

# Generate the graph for 3D pathfinding, removing nodes inside obstacles
def create_graph(grid_size, obstacles):
    G = nx.grid_graph(dim=[grid_size, grid_size, grid_size])
    for min_corner, max_corner in obstacles:
        nodes_to_remove = [node for node in G.nodes 
                           if all(min_corner[i] <= node[i] <= max_corner[i] for i in range(3))]
        G.remove_nodes_from(nodes_to_remove)
    return G

# Visualize the routes in 3D
def visualize_routes(uavs, grid_size, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, grid_size)

    # Plot obstacles as rectangular hexahedrons
    for min_corner, max_corner in obstacles:
        x = [min_corner[0], max_corner[0]]
        y = [min_corner[1], max_corner[1]]
        z = [min_corner[2], max_corner[2]]
        xx, yy = np.meshgrid(x, y)
        ax.plot_surface(xx, yy, z[0], color='red', alpha=0.3)
        ax.plot_surface(xx, yy, z[1], color='red', alpha=0.3)
        ax.plot_surface(xx, z, yy[0], color='red', alpha=0.3)
        ax.plot_surface(xx, z, yy[1], color='red', alpha=0.3)
        ax.plot_surface(z, xx, yy[0], color='red', alpha=0.3)
        ax.plot_surface(z, xx, yy[1], color='red', alpha=0.3)

    for uav in uavs:
        route = np.array(uav.route)
        if route.size > 0:
            ax.plot(route[:, 0], route[:, 1], route[:, 2], label=f'UAV {uav.id}')

    plt.legend()
    plt.show()

# High-Level Conflict Resolution using MCBS
def resolve_conflicts(uavs, graph):
    constraint_tree = []
    for uav in uavs:
        uav.generate_route(graph)
        # Use the length of the route as the priority
        heapq.heappush(constraint_tree, (len(uav.route), uav.id, uav))

    while constraint_tree:
        current_length, _, uav = heapq.heappop(constraint_tree)
        conflicts = detect_conflicts(uav, uavs)
        
        if not conflicts:
            continue
        
        for conflict in conflicts:
            resolve_conflict(conflict, uavs, graph)
            heapq.heappush(constraint_tree, (len(uav.route), uav.id, uav))

def detect_conflicts(uav, uavs):
    conflicts = []
    for other_uav in uavs:
        if uav.id != other_uav.id:
            for i, waypoint in enumerate(uav.route):
                if i < len(other_uav.route) and np.linalg.norm(np.array(waypoint) - np.array(other_uav.route[i])) < NO_GO_RADIUS:
                    conflicts.append((uav, other_uav, i))
    return conflicts

def resolve_conflict(conflict, uavs, graph):
    uav1, uav2, time_step = conflict
    # Simple conflict resolution by adjusting the route of the first UAV involved in the conflict
    uav1.start = adjust_start_position(uav1, time_step, graph)
    if uav1.start in graph:  # Ensure the new start position is still valid
        uav1.generate_route(graph)

def adjust_start_position(uav, time_step, graph):
    # Adjust the start position slightly to avoid conflict, this is a simplistic approach
    new_start = list(uav.start)
    adjustment_count = 0

    # Ensure the new start position is valid
    while tuple(new_start) not in graph and adjustment_count < MAX_ADJUSTMENTS:
        new_start[2] += ALTITUDE_STEP
        adjustment_count += 1
        if new_start[2] > GRID_SIZE:
            break  # Prevent going out of bounds

    if tuple(new_start) in graph:
        return tuple(new_start)
    else:
        return uav.start  # Return original start position if no valid position found

# Main Simulation
def simulate():
    grid_size = GRID_SIZE
    obstacles = generate_obstacles(grid_size, OBSTACLE_COUNT)
    graph = create_graph(grid_size, obstacles)
    uavs = []

    # Create UAVs with random start/end points, altitudes, and velocities
    for i in range(20):
        start = (np.random.randint(0, grid_size-1), np.random.randint(0, grid_size-1), np.random.randint(0, grid_size-1))
        end = (np.random.randint(0, grid_size-1), np.random.randint(0, grid_size-1), np.random.randint(0, grid_size-1))
        min_altitude = np.random.randint(0, grid_size//2) * ALTITUDE_STEP
        max_altitude = min_altitude + ALTITUDE_STEP
        min_velocity = np.random.uniform(5, 10)
        max_velocity = np.random.uniform(10, 20)
        uav = UAV(i, start, end, min_altitude, max_altitude, min_velocity, max_velocity)
        uavs.append(uav)

    # Resolve conflicts
    resolve_conflicts(uavs, graph)

    # Visualize the routes in 3D
    visualize_routes(uavs, grid_size, obstacles)

simulate()
