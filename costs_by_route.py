import googlemaps
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Initialize Google Maps API client
gmaps = googlemaps.Client(key='***REMOVED***')

# Define multiple routes (each with 2 or more delivery points)
routes = [
    {
        "name": "C1,C2",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "C1": (13.700274849301179, -89.19658727198426),
            "C2": (13.699592083787923, -89.18796916668566),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Usulutan, Zacatecoluca",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Zacatecoluca": (13.508156875451148, -88.87071996613791),
            "Usulutan": (13.343501570460036, -88.43294264877568),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "San Miguel, San Miguel EE",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "San Miguel": (13.482360485812624, -88.17610609308709),
            "San Miguel EE": (13.463518936844718, -88.16620547920196),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "San Vicente, Gotera",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "San Vicente": (13.643651339058886, -88.78490442744153),
            "Gotera": (13.697070523244175, -88.10436389687582),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Quezaltepeque, Metro Sur, Salvador del mundo",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Quezaltepeque": (13.831238182901616, -89.27163100009051),
            "Metro Sur": (13.704360322759676, -89.21402777415823),
            "Salvador del mundo": (13.701135550084564, -89.22052841523039),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Santa Tecla, Cascadas",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Santa Tecla": (13.67294569959625, -89.28502017015064),
            "Cascadas": (13.678180325512601, -89.25021950575061),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Chalchuapa, Ahuachapan",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Chalchuapa": (13.985296777730383, -89.6775984964213),
            "Ahuachapan": (13.924841153067355, -89.84505308415532),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Metapan, Santa Ana",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Metapan": (14.33101137444799, -89.44344776909121),
            "Santa Ana": (13.993745561931757, -89.5575659095613),
        },
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },

]

# Function to calculate road distance using Google Maps API
def create_googlemaps_distance_matrix(route):
    locations = list(route["points"].values())
    n = len(locations)

    # Placeholder for distance and time matrices
    distance_matrix = [[0] * n for _ in range(n)]
    time_matrix = [[0] * n for _ in range(n)]

    # Create distance and time matrices by querying Google Maps API
    for i in range(n):
        for j in range(n):
            if i != j:
                # Get driving distance and time from Google Maps API
                origin = locations[i]
                destination = locations[j]

                result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")

                # Extract distance in kilometers
                distance_km = result["rows"][0]["elements"][0]["distance"]["value"] / 1000  # Convert meters to km
                distance_matrix[i][j] = distance_km

                # Extract travel time in hours
                travel_time_seconds = result["rows"][0]["elements"][0]["duration"]["value"]  # Time in seconds
                travel_time_hours = travel_time_seconds / 3600  # Convert seconds to hours
                time_matrix[i][j] = travel_time_hours

    return distance_matrix, time_matrix


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

    # Step 1: Create distance and time matrices for the route using Google Maps API
    distance_matrix, time_matrix = create_googlemaps_distance_matrix(route)

    # Step 2: Set up OR-Tools routing model
    n_locations = len(route["points"])
    manager = pywrapcp.RoutingIndexManager(n_locations, 1, 0)  # 1 vehicle, starting at node 0 (Warehouse)
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback to get distances
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)  # OR-Tools works in meters, so we multiply by 1000

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)

    # Step 3: Extract the solution and calculate the total distance and time
    total_distance = 0
    total_travel_time = 0  # Initialize variable for travel time
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            # Add the distance and time between current and next index
            total_distance += distance_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
            total_travel_time += time_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
            index = next_index

    # Print the total distance
    print(f"Total Distance for {route['name']}: {total_distance:.2f} km")

    # Print the total travel time in hours
    print(f"Total Travel Time for {route['name']}: {total_travel_time:.2f} hours")

    # Step 4: Calculate total cost for the route based on the distance
    total_cost = calculate_route_cost(route, total_distance)
    print(f"Total Cost for {route['name']}: ${total_cost:.2f}\n")
