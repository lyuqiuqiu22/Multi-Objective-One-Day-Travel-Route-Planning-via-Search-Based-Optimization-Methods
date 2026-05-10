import random
import pandas as pd
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATABASE_PATH = DATA_DIR / "database.xlsx"
# -----------------------------
# Helper: parse visit time "a-b" into floats (hours)
# -----------------------------
def parse_visit_time(time_str):
    time_str = str(time_str).replace("–", "-")
    a, b = map(float, time_str.split("-"))
    return a, b

# -----------------------------
# Load visit time intervals for each attraction from Excel
# visit_intervals[name] = (a, b)
# -----------------------------
def load_spot_visit_intervals(excel_path="database.xlsx"):
    df = pd.read_excel(excel_path)
    visit_intervals = {}
    for _, row in df.iterrows():
        visit_intervals[row["Name"]] = parse_visit_time(row["VisitTime_hr"])
    return visit_intervals

# -----------------------------
# Map comfort_rank to daily time limit (hours)
# rank 1 -> 8h, 2 -> 10h, 3 -> 12h
# -----------------------------
def determine_day_limit(comfort_rank):
    if comfort_rank == 1:
        return 8
    elif comfort_rank == 2:
        return 10
    else:
        return 12

# -----------------------------
# Load attraction scores from Excel
# attraction_scores[name] = score
# -----------------------------
def load_attraction_scores(excel_path="database.xlsx"):
    df = pd.read_excel(excel_path)
    if "Attraction_Score" not in df.columns:
        return {row["Name"]: 0 for _, row in df.iterrows()}
    return {row["Name"]: row["Attraction_Score"] for _, row in df.iterrows()}

# -----------------------------
# Sample truncated normal time in [a, b]
# mu = (a + b) / 2, sigma = (b - a) / 6  => [a, b] ≈ [μ - 3σ, μ + 3σ]
# Rejection sampling: keep drawing until a ≤ x ≤ b
# -----------------------------
def sample_normal_time_in_interval(a, b):
    if a == b:
        return a
    mu = (a + b) / 2
    sigma = (b - a) / 6
    while True:
        x = random.gauss(mu, sigma)
        if a <= x <= b:
            return x

# -----------------------------
# Build adjacency dict from transport_df
# For each (Start, End), pick the row with minimal Composite_Cost
# Store best mode, travel time and ticket price
# adjacency[u][v] = {cost, time, mode, ticket}
# -----------------------------
def build_adjacency_from_transport_df(df):


    adjacency = {}


    idx = df.groupby(["Start", "End"])["Composite_Cost"].idxmin()
    best_df = df.loc[idx].copy()



    def get_best_mode_and_time(row):
        if row["Walk_Cost_Composite"] == row["Composite_Cost"]:
            return "walk", row["Walk_Time"]
        elif row["Subway_Cost_Composite"] == row["Composite_Cost"]:
            return "subway", row["Subway_Time"]
        else:
            return "taxi", row["Taxi_Time"]

    best_df["Best_Mode"], best_df["Best_Time"] = zip(
        *best_df.apply(get_best_mode_and_time, axis=1)
    )


    for _, row in best_df.iterrows():
        u = row["Start"]
        v = row["End"]

        if u not in adjacency:
            adjacency[u] = {}

        adjacency[u][v] = {
            "cost": row["Composite_Cost"],
            "time": row["Best_Time"],
            "mode": row["Best_Mode"],
            "ticket": row["Ticket_Price"]
        }

    return adjacency

# -----------------------------
# Evaluate one route order under daily time limit
# Input: a permutation route_order (without start)
# 1) Simulate visiting in this order
# 2) Check at each step if we can still go back to start within day_limit
# 3) Accumulate:
#    - total_cost_internal: sum of Composite_Cost (for SA optimization) + penalties
#    - total_cost_real: sum of (Composite_Cost + ticket) along the route
#    - total_travel_time: sum of travel_time + stay_time + final return time
# -----------------------------
def evaluate_route_with_limit_sa(route_order, start, adjacency,
                                 visit_intervals, attraction_scores, day_limit):

    current = start
    total_travel_time = 0.0
    total_cost_internal = 0.0
    total_cost_real = 0.0
    total_score = 0.0
    visited = []
    edge_details = []

    for place in route_order:
        edge = adjacency.get(current, {}).get(place)
        if edge is None:
            return 1e12, 0.0, [], [], 1e12,0.0

        travel_time = edge["time"]

        a, b = visit_intervals.get(place, (1.0, 2.0))
        stay_time = sample_normal_time_in_interval(a, b)

        back_edge = adjacency.get(place, {}).get(start)
        if back_edge is None:
            return 1e12, 0.0, [], [], 1e12, 0.0

        back_time = back_edge["time"]
        # If we add this place and then go back, total time must still be ≤ day_limit
        if total_travel_time + travel_time + stay_time +back_time> day_limit:
            break

        total_travel_time += travel_time + stay_time

        # Internal cost: only Composite_Cost (for optimization)
        total_cost_internal += edge["cost"]
        # Real cost: Composite_Cost + ticket for each leg
        total_cost_real += edge["cost"] + edge["ticket"]

        total_score += attraction_scores.get(place, 0)
        visited.append(place)

        edge_details.append({
            "from": current,
            "to": place,
            "mode": edge["mode"],
            "edge_cost": edge["cost"],
            "travel_time": travel_time
        })

        current = place

    if not visited:
        return 1e12, 0.0, [], [], 1e12,0.0

    # Finally add return-to-start leg
    back_edge = adjacency.get(current, {}).get(start)
    if back_edge is None:
        return 1e12, total_score, visited, edge_details, 1e12,0.0

    total_travel_time += back_edge["time"]
    total_cost_internal += back_edge["cost"]
    total_cost_real += back_edge["cost"] + back_edge["ticket"]

    edge_details.append({
        "from": current,
        "to": start,
        "mode": back_edge["mode"],
        "edge_cost": back_edge["cost"],
        "travel_time": back_edge["time"]

    })

    # Penalty: enforce at least min_spots attractions
    min_spots = 5
    if len(visited) < min_spots:
        missing = min_spots - len(visited)
        total_cost_internal += missing * 5

    # Reward: each visited spot reduces internal cost
    # (reward_per_spot is negative, so more spots -> lower internal cost)
    reward_per_spot = -3
    total_cost_internal += reward_per_spot * len(visited)

    # Return:
    # total_cost_internal: for SA objective (Composite + penalty/reward)
    # total_cost_real: for reporting (Composite + ticket, no penalty)
    return total_cost_internal, total_score, visited, edge_details, total_cost_real,total_travel_time

# -----------------------------
# Simulated Annealing core:
# - current_route: current permutation of attractions (excluding start)
# - neighbor: swap two positions in current_route
# - Acceptance rule (Metropolis):
#   Δ = new_cost - current_cost
#   accept if Δ < 0 or exp(-Δ / T) > U(0,1)
# - Track best solution seen so far (best_*)
# -----------------------------
def simulated_annealing_select_places(adjacency, start, all_places,
                                      visit_intervals, attraction_scores,
                                      day_limit,
                                      T_init=100.0, T_min=1e-3,
                                      cooling=0.995, max_iter=2000):

    attractions = [p for p in all_places if p != start]

    # Random initial permutation
    if attractions:
        current_route = attractions[:]
        random.shuffle(current_route)
    else:
        current_route = []

    current_cost, current_score, current_visited, current_edges, current_real_cost,current_time = evaluate_route_with_limit_sa(
        current_route, start, adjacency, visit_intervals, attraction_scores, day_limit,
    )

    best_route = current_route[:]
    best_cost = current_cost
    best_score = current_score
    best_visited = current_visited[:]
    best_edges = current_edges[:]
    best_real_cost = current_real_cost
    best_total_time = current_time
    T = T_init
    it = 0

    while T > T_min and it < max_iter and len(attractions) > 1:
        # Neighbor: swap two attractions in the current route
        i, j = random.sample(range(len(current_route)), 2)
        new_route = current_route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_cost, new_score, new_visited, new_edges,new_real_cost,new_time  = evaluate_route_with_limit_sa(
            new_route, start, adjacency, visit_intervals, attraction_scores, day_limit
        )

        delta = new_cost - current_cost

        # Metropolis acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_route = new_route
            current_cost = new_cost
            current_score = new_score
            current_visited = new_visited
            current_edges = new_edges
            current_real_cost = new_real_cost
            current_time = new_time

        # Update global best (tie-breaker: higher score if costs are almost equal)
        if (new_cost < best_cost) or (
            abs(new_cost - best_cost) < 1e-6 and new_score > best_score
        ):
            best_cost = new_cost
            best_score = new_score
            best_route = new_route[:]
            best_visited = new_visited[:]
            best_edges = new_edges[:]
            best_real_cost = new_real_cost
            best_total_time = new_time

        T *= cooling
        it += 1

    return best_visited, best_cost, best_score, best_edges,best_real_cost,best_total_time

# -----------------------------
# Entry point called from main:
# - Filters edges by user-chosen weather
# - Builds adjacency
# - Loads visit intervals & scores
# - Runs SA and prints results in unified format
# -----------------------------
def run_sa_main(transport_df, places_with_start, start_name, comfort_rank,ticket_price,     # ticket_price dict passed from main
        weather):

    # Use only edges that match user-selected weather
    transport_df = transport_df[transport_df["Weather"] == weather].copy()

    adjacency = build_adjacency_from_transport_df(transport_df)

    visit_intervals = load_spot_visit_intervals(DATABASE_PATH)
    attraction_scores = load_attraction_scores(DATABASE_PATH)

    DAY_LIMIT = determine_day_limit(comfort_rank)

    (
        sa_places,
        sa_internal_cost,
        sa_score,
        sa_edges,
        sa_real_cost,
        sa_total_time
    ) = simulated_annealing_select_places(
        adjacency=adjacency,
        start=start_name,
        all_places=places_with_start,
        visit_intervals=visit_intervals,
        attraction_scores=attraction_scores,
        day_limit=DAY_LIMIT
    )
    print("【Final Route】")

    full_path = [start_name] + sa_places + [start_name]
    print(" → ".join(full_path))

    # Sum ticket cost for all visited attractions (start has 0 ticket)
    ticket_cost = sum(ticket_price.get(p, 0) for p in sa_places)

    # Real cost = sum(Composite + ticket) along the route (from evaluate)
    # Therefore traffic_cost = real_cost - ticket_cost
    traffic_cost = sa_real_cost - ticket_cost

    total_cost = sa_real_cost

    print(f"\n● Total travel duration: {sa_total_time:.2f} hours")

    print(f"● Total traffic cost: {traffic_cost:.2f}")
    print(f"● Total ticket cost: RM {ticket_cost:.2f}")
    print(f"● Real Total Cost (without penalty): RM {total_cost:.2f}")
    print(f"● Final Cost for Optimization (with reward/penalty): {sa_internal_cost:.2f}")

    print("----------------------------------------")
    print("【Route Details】")

    for i, e in enumerate(sa_edges, 1):
        print(f"{i}. {e['from']} → {e['to']}")
        print(f"   ● Transportation mode: {e['mode']}")

    print("----------------------------------------")
    print("【Algorithm Information】")
    print("● Algorithm type: Simulated Annealing")
    print("● Initial Temperature: 100")
    print("● Final Temperature: 0.001")
    print("● Cooling Rate: 0.995")
    print("● Max Iterations: 2000")
    print("----------------------------------------\n")

    # main() can still use these returned results if needed
    return sa_places, sa_real_cost, sa_internal_cost, sa_score, sa_edges
