import googlemaps
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium

# Initialize Google Maps API client
gmaps = googlemaps.Client(key='***REMOVED***')

# Dictionary of all unique locations
delivery_points = {
    "PLISA": (13.814771381058584, -89.40960526517033),
    "C1": (13.700274849301179, -89.19658727198426),
    "C2": (13.699592083787923, -89.18796916668566),
    "Zacatecoluca": (13.508156875451148, -88.87071996613791),
    "Usulutan": (13.343501570460036, -88.43294264877568),
    "San Miguel": (13.482360485812624, -88.17610609308709),
    "San Miguel EE": (13.463518936844718, -88.16620547920196),
    "San Vicente": (13.643651339058886, -88.78490442744153),
    "Gotera": (13.697070523244175, -88.10436389687582),
    "Soyapango": (13.702721412595531, -89.1488075058194),
    "Venecia": (13.715569379866428, -89.14405464527051),
    "San Martin": (13.73715057948368, -89.055748653535),
    "Quezaltepeque": (13.831238182901616, -89.27163100009051),
    "Metro Sur": (13.704360322759676, -89.21402777415823),
    "Salvador del mundo": (13.701135550084564, -89.22052841523039),
    "Santa Tecla": (13.67294569959625, -89.28502017015064),
    "Cascadas": (13.678180325512601, -89.25021950575061),
    "Chalchuapa": (13.985296777730383, -89.6775984964213),
    "Ahuachapan": (13.924841153067355, -89.84505308415532),
    "Metapan": (14.33101137444799, -89.44344776909121),
    "Santa Ana": (13.993745561931757, -89.5575659095613),
    "Lourdes": (13.722546962095567, -89.36819169097939),
    "Sonsonate": (13.717906246822801, -89.72425805578887),
    "Apopa": (13.79996666691705, -89.17740146468317),
    "San Luis": (13.71587512531585, -89.21281628594396),
    "Mercado": (13.70035372408547, -89.19660503156757),
    "Mejicanos": (13.722615588871603, -89.18888120921),
    "Cojute": (13.722794399195264, -88.93393263231938),
    "Ilobasco": (13.842682983328299, -88.85068395980095),
    "Aguilares": (13.957413890800401, -89.18619658970312),
    "Chalatenango": (14.042284566312114, -88.93687057229887),
    # Add other locations as needed
}

# Exclude PLISA from clustering but include in routes
cluster_points = {key: val for key, val in delivery_points.items() if key != 'PLISA'}
coords = np.array(list(cluster_points.values()))
number_of_vehicles = 3  # Number of vehicles/routes available

# Perform K-Means clustering with the number of clusters equal to the number of vehicles
kmeans = KMeans(n_clusters=number_of_vehicles, random_state=0).fit(coords)
labels = kmeans.labels_

# Group delivery points by clusters
clustered_points = {i: [] for i in range(number_of_vehicles)}
for label, point_key in zip(labels, cluster_points.keys()):
    clustered_points[label].append(point_key)

# Add PLISA as the start of each route
for cluster in clustered_points.values():
    cluster.insert(0, 'PLISA')

# Function to create a distance matrix and time matrix for a list of points
def create_distance_and_time_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n), dtype=int)
    time_matrix = np.zeros((n, n), dtype=float)  # Time matrix in hours
    for i in range(n):
        for j in range(n):
            if i != j:
                origin = delivery_points[points[i]]
                destination = delivery_points[points[j]]
                result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
                distance_km = result['rows'][0]['elements'][0]['distance']['value'] / 1000
                distance_matrix[i][j] = int(distance_km * 1000)  # Convert km to meters for OR-Tools
                travel_time_seconds = result['rows'][0]['elements'][0]['duration']['value']
                time_matrix[i][j] = travel_time_seconds / 3600  # Convert seconds to hours
    return distance_matrix, time_matrix

# Define the map
m = folium.Map(location=delivery_points['PLISA'], zoom_start=10, tiles='cartodbpositron')

# Function to draw routes on the map
def draw_route(route, color):
    waypoints = [delivery_points[point] for point in route]
    folium.PolyLine(waypoints, color=color, weight=5, opacity=0.8).add_to(m)
    for point in route:
        folium.Marker(delivery_points[point], icon=folium.Icon(color=color), popup=point).add_to(m)

# Colors for different routes
colors = ['red', 'blue', 'green', 'black', 'yellow']

# Process each cluster
for idx, points in clustered_points.items():
    if len(points) > 1:
        distance_matrix, time_matrix = create_distance_and_time_matrix(points)
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            index = routing.Start(0)
            route = []
            total_distance_km = 0
            total_time_hrs = 0
            while not routing.IsEnd(index):
                next_index = solution.Value(routing.NextVar(index))
                route.append(points[manager.IndexToNode(index)])
                total_distance_km += distance_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)] / 1000  # Meters to km
                total_time_hrs += time_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
                index = next_index
            route.append('PLISA')  # Append PLISA to close the loop if round trip is needed
            print(f'Route for Cluster {idx + 1}:')
            print(' -> '.join(route))
            print(f'Total Distance: {total_distance_km:.2f} km')
            print(f'Total Travel Time: {total_time_hrs:.2f} hours')
            draw_route(route, colors[idx % len(colors)])

# Save the map to an HTML file
m.save('routes_map_google_or_maps.html')