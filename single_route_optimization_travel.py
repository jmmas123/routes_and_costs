# pip install googlemaps ortools folium
from typing import Dict, Tuple, List, Optional
import os
import time
import numpy as np
import folium
import googlemaps
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# -------------------------
# INPUT
# -------------------------
# Example: Japan trip points
delivery_points: Dict[str, Tuple[float, float]] = {
    "HANEDA": (35.5494429370959, 139.77784388239684),
    "HAKONE": (35.23096375274929, 139.028038056898),
    "ASABA":  (34.97218683767839, 138.9245598369769),
    "KYOTO":  (35.01395721829809, 135.7651184462139),
    "TOKYO":  (35.68986949007,    139.372656842858),
}

from dotenv import load_dotenv
load_dotenv()

GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
gmaps = googlemaps.Client(key=GMAPS_API_KEY)

# Settings
ALLOWED_MODES = ["driving", "transit"]        # We pick the best of these per edge
OPTIMIZE_FOR = "time"  # 'time' or 'distance'
DEPARTURE_TIME = None  # For transit you can set a datetime for more accurate times
DRAW_MAP = True
MAP_ZOOM = 8
MAP_TILE = 'cartodbpositron'
OUTPUT_HTML = 'trip_map.html'

# -------------------------
# HELPERS
# -------------------------
def _best_leg_between(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    modes: List[str],
    departure_time=None,
    retry: int = 2,
    sleep_s: float = 0.2,
) -> Optional[dict]:
    """
    Query Google Directions for each mode, return the *best* (by time, tie-break by distance).
    Returns dict: {'mode', 'distance_m', 'duration_s', 'polyline': [(lat, lng), ...]}
    or None if no results.
    """
    best = None
    for mode in modes:
        for attempt in range(retry + 1):
            try:
                dr_args = dict(origin=origin, destination=destination, mode=mode)
                if mode == "transit" and departure_time is not None:
                    dr_args["departure_time"] = departure_time

                dirs = gmaps.directions(**dr_args)
                if not dirs:
                    break

                leg = dirs[0]["legs"][0]
                dist_m = leg["distance"]["value"]
                dur_s = leg["duration"]["value"]  # transit includes waiting; more realistic
                # Use overview polyline if available; fallback to leg steps
                polyline_points = []
                if "overview_polyline" in dirs[0] and "points" in dirs[0]["overview_polyline"]:
                    decoded = googlemaps.convert.decode_polyline(dirs[0]["overview_polyline"]["points"])
                    polyline_points = [(p["lat"], p["lng"]) for p in decoded]
                else:
                    for step in leg.get("steps", []):
                        if "polyline" in step and "points" in step["polyline"]:
                            dec = googlemaps.convert.decode_polyline(step["polyline"]["points"])
                            polyline_points.extend([(p["lat"], p["lng"]) for p in dec])

                candidate = dict(mode=mode, distance_m=dist_m, duration_s=dur_s, polyline=polyline_points)

                if best is None:
                    best = candidate
                else:
                    # Primary: shortest time; Secondary: shortest distance
                    if candidate["duration_s"] < best["duration_s"] or (
                        candidate["duration_s"] == best["duration_s"] and candidate["distance_m"] < best["distance_m"]
                    ):
                        best = candidate
                break
            except Exception:
                # transient issues / rate-limit; simple backoff
                time.sleep(sleep_s * (attempt + 1))
                continue
    return best

def build_matrices(
    points_order: List[str],
    optimize_for: str = "time",
    modes: List[str] = ALLOWED_MODES,
    departure_time=None
):
    """
    Returns:
        distance_matrix (int meters),
        time_matrix (float hours),
        leg_meta (dict of dict with chosen 'mode' and 'polyline' per i->j)
    """
    n = len(points_order)
    distance_matrix = np.zeros((n, n), dtype=int)
    time_matrix = np.zeros((n, n), dtype=float)
    leg_meta: List[List[Optional[dict]]] = [[None]*n for _ in range(n)]

    for i in range(n):
        oi = delivery_points[points_order[i]]
        for j in range(n):
            if i == j:
                continue
            dj = delivery_points[points_order[j]]
            best = _best_leg_between(oi, dj, modes=modes, departure_time=departure_time)
            if best is None:
                # Fallback: set a very large cost to discourage this edge
                distance_matrix[i, j] = 10**9
                time_matrix[i, j] = 10**9
                leg_meta[i][j] = None
            else:
                distance_matrix[i, j] = int(best["distance_m"])
                time_matrix[i, j] = best["duration_s"] / 3600.0
                leg_meta[i][j] = best
    return distance_matrix, time_matrix, leg_meta

def solve_tsp(points_order: List[str], distance_matrix, time_matrix, optimize_for="time"):
    """
    Solve a single-vehicle TSP; returns node visit order list of indices.
    """
    n = len(points_order)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # start at index 0 (change if you want different start)
    routing = pywrapcp.RoutingModel(manager)

    if optimize_for == "time":
        matrix = (time_matrix * 3600).astype(int)  # seconds as int cost
    else:
        matrix = distance_matrix.copy()

    def cost_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(matrix[i][j])

    transit_cb_idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None

    # Extract order
    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    order.append(manager.IndexToNode(index))  # return to start (optional; comment if you don’t want a loop)
    return order

def draw_solution_map(points_order: List[str], visit_order: List[int], leg_meta, zoom=MAP_ZOOM, tile=MAP_TILE):
    start = delivery_points[points_order[visit_order[0]]]
    fmap = folium.Map(location=start, zoom_start=zoom, tiles=tile)

    # draw markers
    for name in points_order:
        folium.Marker(delivery_points[name], popup=name).add_to(fmap)

    # draw legs with their chosen polyline
    for k in range(len(visit_order) - 1):
        i, j = visit_order[k], visit_order[k+1]
        meta = leg_meta[i][j]
        if not meta or not meta["polyline"]:
            continue
        # Optional: color by mode
        color = "blue" if meta["mode"] == "driving" else "green"
        folium.PolyLine(meta["polyline"], color=color, weight=5, opacity=0.8).add_to(fmap)

    return fmap

# -------------------------
# RUN
# -------------------------
# 1) Choose an order of points. We’ll fix start at index 0 (e.g., HANEDA).
points_order = list(delivery_points.keys())  # ['HANEDA', 'HAKONE', 'ASABA', 'KYOTO', 'TOKYO']

# 2) Build best-of-modes matrices
distance_matrix, time_matrix, leg_meta = build_matrices(
    points_order,
    optimize_for=OPTIMIZE_FOR,
    modes=ALLOWED_MODES,
    departure_time=DEPARTURE_TIME
)

# 3) Solve TSP minimizing either total time or total distance
visit_order = solve_tsp(points_order, distance_matrix, time_matrix, optimize_for=OPTIMIZE_FOR)
if visit_order is None:
    raise RuntimeError("No feasible route found.")

# 4) Compute totals and pretty-print
total_dist_km = 0.0
total_time_hr = 0.0
route_names = [points_order[i] for i in visit_order]
for k in range(len(visit_order) - 1):
    i, j = visit_order[k], visit_order[k+1]
    total_dist_km += distance_matrix[i][j] / 1000.0
    total_time_hr += time_matrix[i][j]

print("Optimized visit order:", " -> ".join(route_names))
print(f"Total Distance: {total_dist_km:,.2f} km")
print(f"Total Travel Time: {total_time_hr:,.2f} hours")

# 5) Optional: draw map
if DRAW_MAP:
    m = draw_solution_map(points_order, visit_order, leg_meta)
    m.save(OUTPUT_HTML)
    print(f"Map saved to {OUTPUT_HTML}")