import osmnx as ox
import networkx as nx
from itertools import permutations

# Define starting point (e.g., warehouse location)
warehouse_location = (latitude_warehouse, longitude_warehouse)

# Define delivery points (stores, etc.) with a dictionary
delivery_points = {
    "Store A": (latitude_A, longitude_A),
    "Store B": (latitude_B, longitude_B),
    "Store C": (latitude_C, longitude_C)
}

# Combine warehouse and delivery points
all_points = {"Warehouse": warehouse_location}
all_points.update(delivery_points)

# Download the road network near the warehouse (adjust distance for larger areas)
G = ox.graph_from_point(warehouse_location, dist=20000, network_type='drive')


# Function to calculate road distance using OSMnx
def calculate_route_distance(G, start_point, end_point):
    orig_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
    dest_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])
    route = ox.shortest_path(G, orig_node, dest_node, weight='length')
    route_length_km = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length')) / 1000
    return route_length_km


# Step 1: Calculate pairwise distances between all points (including warehouse)
pairwise_distances = {}
for point_a, loc_a in all_points.items():
    pairwise_distances[point_a] = {}
    for point_b, loc_b in all_points.items():
        if point_a != point_b:
            distance_km = calculate_route_distance(G, loc_a, loc_b)
            pairwise_distances[point_a][point_b] = distance_km


# Step 2: Find the optimal route using brute force (for small numbers of points)
def find_optimal_route(pairwise_distances, start_point):
    # Get all points except the start point
    points = list(pairwise_distances.keys())
    points.remove(start_point)

    # Generate all permutations of the delivery points
    all_permutations = permutations(points)

    # Initialize variables to track the best route
    optimal_route = None
    minimal_distance = float('inf')

    # Check each permutation
    for perm in all_permutations:
        # Calculate total distance for the current permutation
        total_distance = 0
        current_point = start_point
        for next_point in perm:
            total_distance += pairwise_distances[current_point][next_point]
            current_point = next_point
        # Add the return to the warehouse
        total_distance += pairwise_distances[current_point][start_point]

        # Update the optimal route if this is the shortest distance
        if total_distance < minimal_distance:
            minimal_distance = total_distance
            optimal_route = [start_point] + list(perm) + [start_point]

    return optimal_route, minimal_distance


# Step 3: Find the optimal route from the warehouse
optimal_route, minimal_distance = find_optimal_route(pairwise_distances, "Warehouse")

# Step 4: Define the cost variables
gas_cost_per_km = 0.5  # Cost of gasoline per kilometer
driver_cost_per_hour = 15  # Driver's wage per hour
auxiliary_cost_per_hour = 10  # Auxiliary personnel wage per hour
average_speed_kmh = 60  # Assumed average speed in kilometers per hour

# Step 5: Calculate the time and cost for the optimal route
time_hours = minimal_distance / average_speed_kmh
total_gas_cost = minimal_distance * gas_cost_per_km
total_driver_cost = time_hours * driver_cost_per_hour
total_auxiliary_cost = time_hours * auxiliary_cost_per_hour
total_cost = total_gas_cost + total_driver_cost + total_auxiliary_cost

# Step 6: Display the results
print(f"Optimal Route: {' -> '.join(optimal_route)}")
print(f"Total Distance: {minimal_distance:.2f} km")
print(f"Total Time: {time_hours:.2f} hours")
print(f"Total Cost: ${total_cost:.2f}")
