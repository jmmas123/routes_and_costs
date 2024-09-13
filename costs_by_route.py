import networkx as nx
import osmnx as ox
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Define multiple routes (each with 2 or more delivery points)
routes = [
    # Route 1
    {
        "name": "C1,C2",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "C1": (13.700334013638587, -89.19656792236255),
            "C2": (13.699614129603189, -89.18805153457025),
        },
        "driver_wage_per_hour": 15,
        "aux_personnel_wage_per_hour": 10,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.5,
        "average_speed_kmh": 60,
    },
    # Route 2
    # {
    #     "name": "Route 2",
    #     "points": {
    #         "Warehouse": (lat_warehouse, lon_warehouse),
    #         "Store C": (lat_C, lon_C),
    #         "Store D": (lat_D, lon_D),
    #     },
    #     "driver_wage_per_hour": 15,
    #     "aux_personnel_wage_per_hour": 10,
    #     "num_aux_personnel": 1,
    #     "gas_cost_per_km": 0.5,
    #     "average_speed_kmh": 60,
    # }
]

# Function to calculate road distance using OSMnx
def create_road_distance_matrix(route):
    locations = list(route["points"].values())
    n = len(locations)

    # Get the road network near the first location (assumed to be the warehouse)
    G = ox.graph_from_point(locations[0], dist=20000, network_type='drive')

    # Placeholder for distance matrix
    distance_matrix = [[0] * n for _ in range(n)]

    # Find the nearest road nodes for all locations
    nodes = [ox.distance.nearest_nodes(G, loc[1], loc[0]) for loc in locations]

    # Calculate road distance between each pair of nodes
    for i in range(n):
        for j in range(n):
            if i != j:
                # Get the shortest path distance between nodes
                try:
                    route_length = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length') / 1000  # in km
                    distance_matrix[i][j] = route_length
                except Exception as e:
                    print(f"Error calculating route from {i} to {j}: {e}")
                    distance_matrix[i][j] = float('inf')  # Use a large number to signify no connection

    return distance_matrix


# Function to calculate the total cost for a route
def calculate_route_cost(route, total_distance):
    time_hours = total_distance / route["average_speed_kmh"]

    # Calculate individual costs
    gas_cost = total_distance * route["gas_cost_per_km"]
    driver_cost = time_hours * route["driver_wage_per_hour"]
    aux_personnel_cost = time_hours * route["aux_personnel_wage_per_hour"] * route["num_aux_personnel"]

    # Total cost for the route
    total_cost = gas_cost + driver_cost + aux_personnel_cost
    return total_cost


# Process each route and calculate costs
for route in routes:
    print(f"Processing {route['name']}")

    # Step 1: Create distance matrix for the route using OSMnx
    distance_matrix = create_road_distance_matrix(route)

    # Step 2: Set up OR-Tools routing model
    n_locations = len(route["points"])
    manager = pywrapcp.RoutingIndexManager(n_locations, 1, 0)  # 1 vehicle, starting at node 0 (Warehouse)
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback to get distances
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]


    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)

    # Step 3: Extract the solution and calculate the total distance
    total_distance = 0
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            total_distance += distance_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
            index = next_index

    print(f"Total Distance for {route['name']}: {total_distance:.2f} km")

    # Step 4: Calculate total cost for the route
    total_cost = calculate_route_cost(route, total_distance)
    print(f"Total Cost for {route['name']}: ${total_cost:.2f}\n")
