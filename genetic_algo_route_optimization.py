import random
import numpy as np
import googlemaps
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans
import folium

# Initialize Google Maps API client
gmaps = googlemaps.Client(key='***REMOVED***')  # Ensure this is your actual Google Maps API key

# Define delivery points with their geographical coordinates
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

# Prepare data for clustering
coords = np.array(list(delivery_points.values()))
n_vehicles = 3  # Number of vehicles/clusters

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_vehicles, random_state=0).fit(coords)
labels = kmeans.labels_

# Group delivery points by assigned cluster
clustered_delivery_points = {i: [] for i in range(n_vehicles)}
for point_label, point_key in zip(labels, delivery_points.keys()):
    clustered_delivery_points[point_label].append(point_key)

# Ensure PLISA is at the start of each route but remove from clusters
for key in clustered_delivery_points:
    if 'PLISA' in clustered_delivery_points[key]:
        clustered_delivery_points[key].remove('PLISA')

# Fetch the distance matrix using Google Maps API
def fetch_distance_matrix():
    locations = list(delivery_points.values())
    n_points = len(locations)
    distance_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                result = gmaps.distance_matrix(origins=[locations[i]], destinations=[locations[j]], mode="driving")
                distance_km = result['rows'][0]['elements'][0]['distance']['value'] / 1000
                distance_matrix[i][j] = distance_km
    return distance_matrix

distance_matrix = fetch_distance_matrix()

# Create individuals
def create_individual():
    routes = []
    for cluster_key, points in clustered_delivery_points.items():
        random.shuffle(points)  # Shuffle points within the cluster
        vehicle_route = ['PLISA'] + points  # Prepend 'PLISA' to each route
        routes.append(vehicle_route)
    return routes

# Mutate route
def mutate_route(individual):
    for route in individual:
        if len(route) > 2:
            start = 2
            end = random.randint(start, len(route) - 1)
            slice_to_shuffle = route[start:end]
            random.shuffle(slice_to_shuffle)
            route[start:end] = slice_to_shuffle
    return individual,

# Evaluate the routes using the distance matrix
def eval_vrp(individual):
    total_distance = 0
    dp_keys = list(delivery_points.keys())
    for vehicle_route in individual:
        if not vehicle_route or len(vehicle_route) < 2:
            continue
        route_distance = 0
        plisa_index = dp_keys.index('PLISA')
        route_distance += distance_matrix[plisa_index][dp_keys.index(vehicle_route[1])]
        for i in range(1, len(vehicle_route)-1):
            start = dp_keys.index(vehicle_route[i])
            end = dp_keys.index(vehicle_route[i+1])
            route_distance += distance_matrix[start][end]
        route_distance += distance_matrix[dp_keys.index(vehicle_route[-1])][plisa_index]
        total_distance += route_distance
    return (total_distance,)

# Setup DEAP Framework
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_vrp)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_route)  # Corrected here
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the Genetic Algorithm
def run_ga():
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    return hof.items[0]

best_solution = run_ga()
print("Optimal solutions found:", best_solution)
print("Best Fitness:", eval_vrp(best_solution)[0])


# Helper function to plot routes using Folium
def plot_routes(routes, delivery_points):
    map_center = list(delivery_points.values())[0]  # Center the map around the first delivery point
    folium_map = folium.Map(location=map_center, zoom_start=12, tiles='cartodbpositron')

    # Colors for different routes
    colors = ['red', 'blue', 'green', 'black', 'yellow']

    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        for i in range(len(route) - 1):
            point_a = delivery_points[route[i]]
            point_b = delivery_points[route[i + 1]]
            folium.Marker(point_a, icon=folium.Icon(color=color), popup=route[i]).add_to(folium_map)
            folium.PolyLine([point_a, point_b], color=color, weight=5).add_to(folium_map)
        folium.Marker(delivery_points[route[-1]], icon=folium.Icon(color=color), popup=route[-1]).add_to(folium_map)

    folium_map.save('routes_map_gen_algo.html')
    return folium_map

# After the GA run, use the best_solution to plot routes
map_created = plot_routes(best_solution, delivery_points)
map_created