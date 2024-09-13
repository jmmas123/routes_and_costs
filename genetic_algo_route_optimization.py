import random
from deap import base, creator, tools, algorithms
import numpy as np

# Optimization type: either "distance" or "cost"
optimize_for = "cost"

# Define delivery points (latitude, longitude)
delivery_points = {
    "PLISA": (13.814771381058584, -89.40960526517033),
    "C1": (13.700334013638587, -89.19656792236255),
    "C2": (13.699614129603189, -89.18805153457025),

}

# Number of points including the warehouse
n_points = len(delivery_points)


# Calculate geodesic distances (straight-line distances)
distance_wa = geodesic(Warehouse, store_a).kilometers
distance_ab = geodesic(store_a, store_b).kilometers
distance_bc = geodesic(store_b, store_c).kilometers

# Distance matrix (pairwise distances between points)
distance_matrix = np.array([
    [0, 12, 23, 34],  # Warehouse
    [12, 0, 45, 56],  # Store A
    [23, 45, 0, 67],  # Store B
    [34, 56, 67, 0],  # Store C
])

# Cost variables
gas_cost_per_km = 0.5  # Cost of gasoline per kilometer
driver_cost_per_hour = 15  # Driver's wage per hour
auxiliary_cost_per_hour = 10  # Auxiliary personnel wage per hour
average_speed_kmh = 60  # Assumed average speed in kilometers per hour

# Genetic algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(1, n_points), n_points - 1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness function that calculates total cost or total distance
def eval_route(individual):
    total_distance = distance_matrix[0, individual[0]]  # Start from warehouse
    for i in range(len(individual) - 1):
        total_distance += distance_matrix[individual[i], individual[i + 1]]
    total_distance += distance_matrix[individual[-1], 0]  # Return to warehouse

    if optimize_for == "distance":
        # Optimize for total distance
        return total_distance,
    elif optimize_for == "cost":
        # Optimize for total cost
        time_hours = total_distance / average_speed_kmh
        total_gas_cost = total_distance * gas_cost_per_km
        total_driver_cost = time_hours * driver_cost_per_hour
        total_auxiliary_cost = time_hours * auxiliary_cost_per_hour
        total_cost = total_gas_cost + total_driver_cost + total_auxiliary_cost
        return total_cost,


toolbox.register("evaluate", eval_route)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# Genetic algorithm process
def genetic_algorithm_route():
    population = toolbox.population(n=100)  # Create initial population
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Evolve population
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=500,
                        stats=stats, halloffame=hall_of_fame, verbose=True)

    return hall_of_fame[0]


# Find the best route using the genetic algorithm
best_route = genetic_algorithm_route()

# Decode route
decoded_route = ['Warehouse'] + [list(delivery_points.keys())[i] for i in best_route] + ['Warehouse']
print("Optimal Route:", " -> ".join(decoded_route))

# Evaluate and display the chosen optimization (distance or cost)
if optimize_for == "distance":
    optimal_distance = eval_route(best_route)[0]
    print(f"Total Distance: {optimal_distance:.2f} km")
else:
    optimal_cost = eval_route(best_route)[0]
    print(f"Total Cost: ${optimal_cost:.2f}")