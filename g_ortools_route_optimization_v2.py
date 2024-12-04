import googlemaps
import numpy as np
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
import os
import pandas as pd

# Initialize Google Maps API client
gmaps = googlemaps.Client(key='***REMOVED***')

pd.set_option(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.expand_frame_repr", False
)

# Define the paths to your data files
def get_base_path(file_type):
    if os.name == 'nt':  # Windows
        if file_type == 'routing':
            return r'\\192.168.10.18\Bodega General\HE\VARIOS\rutas'
    else:  # MacOS
        if file_type == 'routing':
            return '/Users/j.m./Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/HE/VARIOS/rutas'


# Get base paths
overtime_t_base_path = get_base_path('routing')

income_overtime_client_path = os.path.join(overtime_t_base_path, 'control de rutas y fletes.xlsx')

# Read document containing Routing information
df_control = pd.read_excel(income_overtime_client_path, sheet_name='ISSS_N')


print("Destinations for analysis:\n", df_control)

# Converting to dictionary
delivery_points = {
    row['Direccion']: tuple(map(float, row['Coordenada'].split(', ')))
    for _, row in df_control.iterrows()
}

number_of_vehicles = 4
# Number of vehicles/routes available
max_stores_per_vehicle = 6  # Maximum number of stores per vehicle

# Exclude PLISA from clustering but include it in the routes
cluster_points = {key: val for key, val in delivery_points.items() if key != 'PLISA'}
coords = np.array(list(cluster_points.values()))

# Perform initial K-Means clustering
kmeans = KMeans(n_clusters=number_of_vehicles, random_state=0).fit(coords)
labels = kmeans.labels_

# Group delivery points by clusters
clustered_points = {i: [] for i in range(number_of_vehicles)}
for label, point_key in zip(labels, cluster_points.keys()):
    clustered_points[label].append(point_key)

# Add PLISA as the start of each route
for cluster in clustered_points.values():
    cluster.insert(0, 'PLISA')


# Function to further cluster points that exceed the store limit
def split_large_cluster(cluster, cluster_coords, max_stores):
    # Remove PLISA for sub-clustering
    cluster_points = cluster[1:]  # Remove PLISA

    # Calculate the number of sub-clusters needed
    num_subclusters = (len(cluster_points) // max_stores) + (1 if len(cluster_points) % max_stores != 0 else 0)

    # Initialize sub-clusters
    sub_clusters = {i: [] for i in range(num_subclusters)}

    # Manually divide the points into sub-clusters
    for i, point_key in enumerate(cluster_points):
        sub_cluster_index = i // max_stores  # Determine which sub-cluster this point should go into
        sub_clusters[sub_cluster_index].append(point_key)

    # Add PLISA to the beginning of each sub-cluster
    for sub_cluster in sub_clusters.values():
        sub_cluster.insert(0, 'PLISA')

    return sub_clusters


# Check if any cluster exceeds the store limit and split if needed
final_clusters = {}
cluster_id = 0

for idx, cluster in clustered_points.items():
    if len(cluster) - 1 > max_stores_per_vehicle:  # Exclude PLISA from the count
        print(f"Cluster {idx + 1} exceeds {max_stores_per_vehicle} stores. Splitting into sub-clusters.")
        cluster_coords = [delivery_points[point] for point in cluster]
        sub_clusters = split_large_cluster(cluster, cluster_coords, max_stores_per_vehicle)
        for sub_cluster in sub_clusters.values():
            final_clusters[cluster_id] = sub_cluster
            cluster_id += 1
    else:
        final_clusters[cluster_id] = cluster
        cluster_id += 1


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


def draw_route(route, color):
    # Get the waypoints coordinates for the route
    waypoints = [delivery_points[point] for point in route]

    # Loop over the route waypoints to draw the road routes between consecutive points
    for i in range(len(waypoints) - 1):
        origin = waypoints[i]
        destination = waypoints[i + 1]

        # Use Google Maps Directions API to get the road path between two points
        directions_result = gmaps.directions(
            origin=origin,
            destination=destination,
            mode="driving",  # Can also use "walking", "bicycling", etc.
            optimize_waypoints=True
        )

        # Extract the polyline points (encoded)
        if directions_result and 'overview_polyline' in directions_result[0]:
            polyline = directions_result[0]['overview_polyline']['points']
            decoded_polyline = googlemaps.convert.decode_polyline(polyline)

            # Convert the decoded polyline from a list of dicts to a list of tuples
            polyline_tuples = [(point['lat'], point['lng']) for point in decoded_polyline]

            # Debug: Print decoded polyline to ensure it's being decoded correctly
            # print(f"Decoded polyline for {origin} -> {destination}: {polyline_tuples}")

            # Draw the road path using the decoded polyline points
            folium.PolyLine(polyline_tuples, color=color, weight=5, opacity=0.8).add_to(m)
        # else:
            # print(f"Polyline not found for {origin} -> {destination}")

    # Mark the waypoints on the map
    for point in route:
        folium.Marker(delivery_points[point], icon=folium.Icon(color=color), popup=point).add_to(m)

# Colors for different routes
colors = ['black', 'blue', 'green', 'gray', 'cadetblue', 'darkgreen', 'lightblue',
 'purple', 'pink', 'orange', 'red', 'lightgreen', 'darkred', 'lightgray', 'darkblue', 'cyan', 'magenta', 'yellow']


# Initialize a variable to store total distance for all routes
total_km_all_routes = 0

# Process each final cluster for routing
for idx, points in final_clusters.items():
    if len(points) > 1:  # Only process clusters with delivery points
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
                total_distance_km += distance_matrix[manager.IndexToNode(index)][
                                         manager.IndexToNode(next_index)] / 1000  # Meters to km
                total_time_hrs += time_matrix[manager.IndexToNode(index)][manager.IndexToNode(next_index)]
                index = next_index
            route.append('PLISA')  # Append PLISA to close the loop if round trip is needed
            print(f'Route for Cluster {idx + 1}:')
            print(' -> '.join(route))
            print(f'Total Distance: {total_distance_km:.2f} km')
            print(f'Total Travel Time: {total_time_hrs:.2f} hours\n')

            # Add the total distance of this route to the overall total
            total_km_all_routes += total_distance_km

            # Draw the route on the map
            draw_route(route, colors[idx % len(colors)])
        else:
            print(f"No feasible route found for Cluster {idx + 1}.")

# Print the total kilometers for all routes
print(f'Total Distance for all routes: {total_km_all_routes:.2f} km')


# Save the map to an HTML file
m.save('routes_map_google_or_maps_v2.html')
