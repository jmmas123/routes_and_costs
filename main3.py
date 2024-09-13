from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Distance matrix (pairwise distances between points)
distance_matrix = [
    [0, 12, 23, 34],  # Warehouse
    [12, 0, 45, 56],  # Store A
    [23, 45, 0, 67],  # Store B
    [34, 56, 67, 0],  # Store C
]

# Cost variables
gas_cost_per_km = 0.5  # Cost of gasoline per kilometer
driver_cost_per_hour = 15  # Driver's wage per hour
auxiliary_cost_per_hour = 10  # Auxiliary personnel wage per hour
average_speed_kmh = 60  # Assumed average speed in kilometers per hour

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

# Define the cost of each arc (route segment)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Set the search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem
solution = routing.SolveWithParameters(search_parameters)

# Extract the route and calculate costs
if solution:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    # Define delivery points (latitude, longitude)
    delivery_points = {
        "Warehouse": (latitude_warehouse, longitude_warehouse),
        "Store A": (latitude_A, longitude_A),
        "Store B": (latitude_B, longitude_B),
        "Store C": (latitude_C, longitude_C),
    }

    # Distance matrix (pairwise distances between points)
    distance_matrix = [
        [0, 12, 23, 34],  # Warehouse
        [12, 0, 45, 56],  # Store A
        [23, 45, 0, 67],  # Store B
        [34, 56, 67, 0],  # Store C
    ]

    # Cost variables
    gas_cost_per_km = 0.5  # Cost of gasoline per kilometer
    driver_cost_per_hour = 15  # Driver's wage per hour
    auxiliary_cost_per_hour = 10  # Auxiliary personnel wage per hour
    average_speed_kmh = 60  # Assumed average speed in kilometers per hour

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

    # Define the cost of each arc (route segment)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

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
        decoded_route = ['Warehouse'] + [list(delivery_points.keys())[i] for i in route[1:-1]] + ['Warehouse']
        print("Optimal Route:", " -> ".join(decoded_route))

        # Calculate the total time and costs
        time_hours = total_distance / average_speed_kmh
        total_gas_cost = total_distance * gas_cost_per_km
        total_driver_cost = time_hours * driver_cost_per_hour
        total_auxiliary_cost = time_hours * auxiliary_cost_per_hour
        total_cost = total_gas_cost + total_driver_cost + total_auxiliary_cost

        print(f"Total Distance: {total_distance:.2f} km")
        print(f"Total Time: {time_hours:.2f} hours")
        print(f"Total Cost: ${total_cost:.2f}")
