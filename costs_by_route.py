import googlemaps
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium

# Initialize Google Maps API client
gmaps = googlemaps.Client(key='***REMOVED***')

# Define multiple routes (each with 2 or more delivery points)
routes = [
    {
        "name": "C1, C2",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "C1": (13.700274849301179, -89.19658727198426),
            "C2": (13.699592083787923, -89.18796916668566),
        },
        "unloading_time_h_per_store": 2.0,
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
        "unloading_time_h_per_store": 2.0,
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
        "unloading_time_h_per_store": 2.0,
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
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Soyapango, Venecia, San Martin",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Soyapango": (13.702721412595531, -89.1488075058194),
            "Venecia": (13.715569379866428, -89.14405464527051),
            "San Martin": (13.73715057948368, -89.055748653535),
        },
        "unloading_time_h_per_store": 2.0,
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
        "unloading_time_h_per_store": 2.0,
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
        "unloading_time_h_per_store": 2.0,
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
        'unloading_time_h_per_store': 2.0,
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
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Lourdes, Sonsonate",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Lourdes": (13.722546962095567, -89.36819169097939),
            "Sonsonate": (13.717906246822801, -89.72425805578887),
        },
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Apopa, San Luis",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Apopa": (13.79996666691705, -89.17740146468317),
            "San Luis": (13.71587512531585, -89.21281628594396),
        },
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Mercado, Mejicanos",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Mercado": (13.70035372408547, -89.19660503156757),
            "Mejicanos": (13.722615588871603, -89.18888120921),
        },
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Cojute, Iloabasco",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Cojute": (13.722794399195264, -88.93393263231938),
            "Ilobasco": (13.842682983328299, -88.85068395980095),
        },
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "Aguilares, Chalatenango",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Aguilares": (13.957413890800401, -89.18619658970312),
            "Chalatenango": (14.042284566312114, -88.93687057229887),
        },
        "unloading_time_h_per_store": 2.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
    {
        "name": "San Luis, Laico",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Aguilares": (13.71587512531585, -89.21281628594396),
            "Chalatenango": (13.712915, -89.199730),
        },
        "unloading_time_h_per_store": 1.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },
{
        "name": "Alsasa",
        "points": {
            "PLISA": (13.814771381058584, -89.40960526517033),
            "Alsasa": (13.736415681105896, -89.40258351847194),

        },
        "unloading_time_h_per_store": 1.0,
        "driver_wage_per_hour": 2.5,
        "aux_personnel_wage_per_hour": 2.0,
        "num_aux_personnel": 2,
        "gas_cost_per_km": 0.30,
        "average_speed_kmh": 60,
    },

]

def calculate_unique_stores(route):
    # Extract delivery points (stores) and count unique ones
    unique_stores = set(route["points"].keys())
    num_stores = len(unique_stores)
    return num_stores

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

# Updated function to calculate route cost, now using travel time directly
def calculate_route_cost(route, total_distance, total_travel_time):
    # Calculate the unique number of stores (delivery points)
    num_stores = calculate_unique_stores(route)

    # Calculate unloading time cost for the stores
    unloading_time_total = route['unloading_time_h_per_store'] * (num_stores-1)

    # Total time including unloading (use Google Maps travel time)
    total_time = total_travel_time + unloading_time_total

    # Calculate individual costs
    gas_cost = total_distance * route["gas_cost_per_km"]
    driver_cost = total_time * route["driver_wage_per_hour"]
    aux_personnel_cost = total_time * route["aux_personnel_wage_per_hour"] * route["num_aux_personnel"]

    # Total cost for the route
    total_cost = gas_cost + driver_cost + aux_personnel_cost

    print(f"Total unloading time: {unloading_time_total} hours")
    print(f"Total travel time: {total_travel_time:.2f} hours")
    print(f"Total time: {total_time:.2f} hours")

    return total_cost

# Function to process each route and calculate costs
for route in routes:
    print(f"Processing {route['name']}\n")

    # Create distance and time matrices for the route using Google Maps API
    distance_matrix, time_matrix = create_googlemaps_distance_matrix(route)

    # Set up OR-Tools routing model
    n_locations = len(route["points"])
    manager = pywrapcp.RoutingIndexManager(n_locations, 1, 0)  # 1 vehicle, starting at node 0 (PLISA)
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback to get distances
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)  # Convert km to meters for OR-Tools

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve the problem with the cheapest path strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)
    total_travel_time = 0  # Initialize variable for travel time

    # Extract the solution and calculate the total distance and time
    if solution:
        index = routing.Start(0)
        plan_output = f"Specified route order:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            plan_output += f"{manager.IndexToNode(index)} -> "
            route_distance += distance_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
            total_travel_time += time_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]

            index = next_index

        # Ensure to print the return to the starting point explicitly
        plan_output += f"{manager.IndexToNode(routing.Start(0))}\n"  # Shows return to start
        plan_output += f"Total Distance: {route_distance:.2f} km\n"

        # Calculate total cost based on the distance and actual travel time
        total_cost = calculate_route_cost(route, route_distance, total_travel_time)
        plan_output += f"Total Cost: ${total_cost:.2f}\n"
        plan_output += f"Suggested Price: ${total_cost * 1.5:.2f}\n"
        print(plan_output)

# Function to initialize the map
def init_map(center_location, zoom_start=8):
    return folium.Map(location=center_location, zoom_start=zoom_start)


# Function to add routes to the map
def add_route_to_map(m, route, line_color):
    route_points = list(route["points"].values())

    # Add markers for each point in the route
    for point_name, point in route["points"].items():
        folium.Marker(
            location=point,
            popup=f"<strong>{point_name}</strong>",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # Add lines to connect the points
    folium.PolyLine(route_points, color=line_color, weight=5, opacity=0.8).add_to(m)


# Colors for different routes
colors = ['black', 'darkpurple', 'blue', 'green', 'lightred', 'gray', 'cadetblue', 'white', 'darkgreen', 'lightblue',
          'purple', 'pink', 'beige', 'orange', 'red', 'lightgreen', 'darkred', 'lightgray', 'darkblue']

# Initialize the map centered around the first point of the first route
map_obj = init_map(center_location=list(routes[0]["points"].values())[0])

# Add each route to the map with a different color
for idx, route in enumerate(routes):
    add_route_to_map(map_obj, route, colors[
        idx % len(colors)])  # Use modulo to cycle through colors if there are more routes than colors

# Save the map as an HTML file
map_obj.save('routes_costs_analyzed.html')
