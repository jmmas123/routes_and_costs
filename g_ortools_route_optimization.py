import osmnx as ox
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Define delivery points as (name, (latitude, longitude))
locations = {
    "Warehouse": (lat_warehouse, lon_warehouse),
    "Store A": (lat_A, lon_A),
    "Store B": (lat_B, lon_B),
    "Store C": (lat_C, lon_C)
}

# Download the road network for the area around the warehouse
warehouse_location = locations["Warehouse"]
G = ox.graph_from_point(warehouse_location, dist=10000, network_type='drive')


# Function to calculate road distance using OSMnx
def calculate_road_distance(G, start_location, end_location):
    orig_node = ox.distance.nearest_nodes(G, X=start_location[1], Y=start_location[0])
    dest_node = ox.distance.nearest_nodes(G, X=end_location[1], Y=end_location[0])
    route = ox.shortest_path(G, orig_node, dest_node, weight='length')
    route_length_km = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length')) / 1000
    return route_length_km


# Automate the creation of the distance matrix
def create_distance_matrix(locations):
    location_names = list(locations.keys())
    n = len(location_names)

    # Initialize an empty distance matrix
    distance_matrix = [[0] * n for _ in range(n)]

    # Fill the matrix with road distances between all location pairs
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = calculate_road_distance(G, locations[location_names[i]],
                                                                locations[location_names[j]])

    return distance_matrix, location_names


# Generate the distance matrix
distance_matrix, location_names = create_distance_matrix(locations)

# Cost variables
gas_cost_per_km = 0.5  # Cost of gasoline per kilometer
driver_cost_per_hour = 15  # Driver's wage per hour
auxiliary_cost_per_hour = 10  # Auxiliary personnel wage per hour
average_speed_kmh = 60  # Assumed average speed in kilometers per hour

# Optimization type: either "distance" or "cost"
optimize_for = "cost"  # or set to "distance"

# Create the routing index manager
manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)

# Create routing model
routing = pywrapcp.RoutingModel(manager)


# Create a callback function to return distances between points
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node][to_node]


# Register the distance callback
transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Define the cost of each arc (route segment) based on distance or cost
if optimize_for == "distance":
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
else:
    def cost_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        distance_km = distance_matrix[from_node][to_node]
        time_hours = distance_km / average_speed_kmh
        total_gas_cost = distance_km * gas_cost_per_km
        total_driver_cost = time_hours * driver_cost_per_hour
        total_auxiliary_cost = time_hours * auxiliary_cost_per_hour
        total_cost = total_gas_cost + total_driver_cost + total_auxiliary_cost
        return int(total_cost * 100)  # Convert to integer for OR-Tools


    cost_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

# Set the search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem
solution = routing.SolveWithParameters(search_parameters)

# Extract the route and calculate costs
if solution:
    index = routing.Start(0)
    route = []
    total_distance = 0
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        next_index = solution.Value(routing.NextVar(index))
        total_distance += distance_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
        index = next_index
    route.append(manager.IndexToNode(index))  # Return to start point

    # Decode the route to store names
    decoded_route = [location_names[i] for i in route]
    print("Optimal Route:", " -> ".join(decoded_route))

    # Calculate and display results
    if optimize_for == "distance":
        print(f"Total Distance: {total_distance:.2f} km")
    else:
        # Recalculate total cost for the entire route
        time_hours = total_distance / average_speed_kmh
        total_gas_cost = total_distance * gas_cost_per_km
        total_driver_cost = time_hours * driver_cost_per_hour
        total_auxiliary_cost = time_hours * auxiliary_cost_per_hour
        total_cost = total_gas_cost + total_driver_cost + total_auxiliary_cost
        print(f"Total Cost: ${total_cost:.2f}")
