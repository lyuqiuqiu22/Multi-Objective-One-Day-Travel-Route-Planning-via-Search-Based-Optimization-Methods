import random
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATABASE_PATH = DATA_DIR / "database.xlsx"
TRANSPORT_PATH = DATA_DIR / "transport_composite_cost.xlsx"

# ===========================================================
#  PART 1 ——— Read user input
#  (play time, maximum play time, transportation cost matrix, attraction appeal scores)
# ===========================================================
def parse_visit_time(time_str):
    time_str = str(time_str).replace("–", "-")
    parts = time_str.split("-")
    a = float(parts[0])
    b = float(parts[1])
    return a, b

def load_spot_visit_intervals(excel_path=DATABASE_PATH):
    df = pd.read_excel(excel_path)
    visit_intervals = {}

    for _, row in df.iterrows():
        visit_intervals[row["Name"]] = parse_visit_time(row["VisitTime_hr"])

    return visit_intervals

def determine_day_limit(comfort_rank):
    """
    Based on user comfort ranking (1 being most important), return the maximum daily playtime (hours):
        1 → 8h
        2 → 10h
        3 → 12h
    """
    if comfort_rank == 1:
        return 8
    elif comfort_rank == 2:
        return 10
    else:
        return 12


def build_cost_and_time_matrices_real(excel_path, time_period="offpeak", weather="sunny", start_name="Hotel"):
    df = pd.read_excel(excel_path)

    df = df[(df["Time_Period"] == time_period) & (df["Weather"] == weather)]
    all_places = list(set(df["Start"].tolist() + df["End"].tolist()))

    if start_name not in all_places:
        raise ValueError("The hotel name is not in the transportation matrix. Please check the main file.")

    other_places = [p for p in all_places if p != start_name]
    places = [start_name] + sorted(other_places)

    n = len(places)
    idx_of = {name: i for i, name in enumerate(places)}

    # Initialize the matrix
    cost_matrix = np.full((n, n), np.inf)
    time_matrix = np.full((n, n), np.inf)
    mode_matrix = [["" for _ in range(n)] for _ in range(n)]   

    np.fill_diagonal(cost_matrix, 0.0)
    np.fill_diagonal(time_matrix, 0.0)

    for _, row in df.iterrows():
        i = idx_of[row["Start"]]
        j = idx_of[row["End"]]

        # 3 types of transport
        walk_cost = row["Walk_Cost_Composite"]
        subway_cost = row["Subway_Cost_Composite"]
        taxi_cost = row["Taxi_Cost_Composite"]

        walk_time = row["Walk_Time"]
        subway_time = row["Subway_Time"]
        taxi_time = row["Taxi_Time"]

        # Choose the optimal way of transportation
        cost_options = {
            "Walk": walk_cost,
            "Subway": subway_cost,
            "Taxi": taxi_cost
        }

        best_mode = min(cost_options, key=cost_options.get)
        best_cost = cost_options[best_mode]
        best_time = min(walk_time, subway_time, taxi_time)

        # Write to the matrix (overwrite if better)
        if best_cost < cost_matrix[i][j]:
            cost_matrix[i][j] = best_cost
            mode_matrix[i][j] = best_mode

        if best_time < time_matrix[i][j]:
            time_matrix[i][j] = best_time

    return places, cost_matrix, time_matrix, mode_matrix

def load_attraction_scores(excel_path):
    df = pd.read_excel(excel_path)
    scores = {}
    for _, row in df.iterrows():
        scores[row["Name"]] = float(row["Attraction_Score"])
    return scores

# ======================================================================
# PART 2 — GA's total cost function for a route (transportation + random exploration time)
# ======================================================================
def build_route(mask, order):
    # Choose attractions according to mask（0/1），and order according to "order"
    chosen = [city for city in order if mask[city - 1] == 1]
    return chosen


def route_cost(route, cost_matrix, time_matrix,
               visit_intervals, places, DAY_LIMIT, ticket_costs):

    if len(route) < 2:
        return 9999999, 9999999, 0, 0, 0

    total_cost = 0.0
    total_time = 0.0
    total_traffic_cost = 0.0
    total_ticket_cost = 0.0

    prev = 0

    def name_of(idx):
        return places[idx]

    # calculate route by route
    for city in route:

        traffic_cost = cost_matrix[prev][city]
        travel_time = time_matrix[prev][city]

        total_cost += traffic_cost
        total_time += travel_time
        total_traffic_cost += traffic_cost

        # Use normal random variable to calculate the time in attractions
        a, b = visit_intervals[name_of(city)]
        mu = (a + b) / 2
        sigma = (b - a) / 6

        stay = random.gauss(mu, sigma)
        total_time += stay

        # Sum the ticket cost
        ticket = ticket_costs.get(name_of(city), 0)
        total_cost += ticket
        total_ticket_cost += ticket

        prev = city

    # Get back to hotel
    traffic_cost = cost_matrix[prev][0]
    travel_time = time_matrix[prev][0]

    total_cost += traffic_cost
    total_time += travel_time
    total_traffic_cost += traffic_cost

    # Timeout penalty
    overtime = total_time - DAY_LIMIT
    penalty = 0
    if overtime > 0:
        penalty = overtime ** 2 * 10

    raw_cost = total_cost
    final_cost = total_cost + penalty

    # Multi-Attraction Reward
    num_spots = len(route)
    final_cost = final_cost - num_spots * 3

    return raw_cost, final_cost, total_time, total_traffic_cost, total_ticket_cost

# Calculate the summation of route attractions
def get_route_attraction(route, places, attraction_scores):
    return sum(attraction_scores[places[i]] for i in route)

# ======================================================================
# PART 3 — GA components：fitness / selection / crossover / mutation
# ======================================================================
def fitness(indv, cost_matrix, time_matrix, visit_intervals, places, DAY_LIMIT, ticket_costs):
    mask, order = indv
    route = build_route(mask, order)
    raw_cost, final_cost, total_time, total_traffic_cost, total_ticket_cost = \
        route_cost(route, cost_matrix, time_matrix,
                   visit_intervals, places, DAY_LIMIT, ticket_costs)

    return 1 / (final_cost + 1e-9)


def tournament_selection(pop, cost_matrix, time_matrix, visit_intervals, places, DAY_LIMIT, ticket_costs, k=3):
    sample = random.sample(pop, k)
    best = max(sample, key=lambda ind: fitness(ind, cost_matrix, time_matrix,
                                               visit_intervals, places, DAY_LIMIT, ticket_costs))
    return best


def mask_crossover(m1, m2):
    point = random.randint(1, len(m1) - 1)
    return m1[:point] + m2[point:], m2[:point] + m1[point:]


def order_crossover(p1, p2):
    n = len(p1)
    child = [None] * n
    a, b = sorted(random.sample(range(n), 2))

    child[a:b + 1] = p1[a:b + 1]

    p2_ptr = 0
    for i in range(n):
        if child[i] is None:
            while p2[p2_ptr] in child:
                p2_ptr += 1
            child[i] = p2[p2_ptr]

    return child


def mutation(individual, mutation_rate=0.1):
    mask, order = individual

    # Mask mutation (random flip)
    new_mask = mask[:]
    if random.random() < mutation_rate:
        pos = random.randint(0, len(new_mask) - 1)
        new_mask[pos] = 1 - new_mask[pos]

    # order Exchange mutation
    new_order = order[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(new_order)), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]

    return new_mask, new_order


# ======================================================================
# PART 4 — GA Main Program (Select Attractions + Sorting)
# ======================================================================
def init_individual(n_city):
    # Initialize mask using 0/1
    mask = [random.choice([0, 1]) for _ in range(n_city)]
    if sum(mask) == 0:
        mask[random.randint(0, n_city - 1)] = 1  # Have to go to at least one attraction

    # Randomly sorting order
    order = list(range(1, n_city + 1))
    random.shuffle(order)

    return mask, order


def GA_search(places, cost_matrix, time_matrix,
              visit_intervals, comfort_rank, ticket_costs, attraction_scores,
              pop_size=60, generations=300,
              crossover_rate=0.9, mutation_rate=0.1):

    DAY_LIMIT = determine_day_limit(comfort_rank)

    n_city = len(places) - 1  # Excluding hotel
    population = [init_individual(n_city) for _ in range(pop_size)]

    best_ind = None
    best_cost = float("inf")
    best_raw_cost = float("inf")
    best_gen= None

    for gen in range(generations):

        # Update Current Optimal (Considering attraction score)
        for ind in population:
            route = build_route(ind[0], ind[1])
            raw_cost, final_cost, total_time, total_traffic_cost, total_ticket_cost = \
                route_cost(route, cost_matrix, time_matrix,
                           visit_intervals, places, DAY_LIMIT, ticket_costs)

            attraction_now = get_route_attraction(route, places, attraction_scores)

            if best_ind is None:
                # initialize
                best_ind = ind
                best_cost = final_cost
                best_raw_cost = raw_cost
                best_attraction = attraction_now
                best_total_time = total_time
                best_total_traffic_cost = total_traffic_cost
                best_total_ticket_cost = total_ticket_cost
                best_gen = gen + 1

            elif final_cost < best_cost:
                # First goal：Minimize the final_cost
                best_ind = ind
                best_cost = final_cost
                best_raw_cost = raw_cost
                best_attraction = attraction_now
                best_total_time = total_time
                best_total_traffic_cost = total_traffic_cost
                best_total_ticket_cost = total_ticket_cost
                best_gen = gen + 1

            elif abs(final_cost - best_cost) < 1e-6:
                # Second goal：When costs are equal → Choose the route with the higher attractive score
                if attraction_now > best_attraction:
                    best_ind = ind
                    best_raw_cost = raw_cost
                    best_attraction = attraction_now
                    best_total_time = total_time
                    best_total_traffic_cost = total_traffic_cost
                    best_total_ticket_cost = total_ticket_cost
                    best_gen = gen + 1

        new_pop = []

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, cost_matrix, time_matrix,
                                      visit_intervals, places, DAY_LIMIT,ticket_costs)
            p2 = tournament_selection(population, cost_matrix, time_matrix,
                                      visit_intervals, places, DAY_LIMIT,ticket_costs)

            # crossover
            if random.random() < crossover_rate:
                m1, m2 = mask_crossover(p1[0], p2[0])
                o1 = order_crossover(p1[1], p2[1])
                o2 = order_crossover(p2[1], p1[1])
                c1 = (m1, o1)
                c2 = (m2, o2)
            else:
                c1, c2 = p1, p2

            c1 = mutation(c1, mutation_rate)
            c2 = mutation(c2, mutation_rate)

            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]

    return (best_ind, best_cost, best_raw_cost,
            best_total_time, best_total_traffic_cost, best_total_ticket_cost,
            pop_size, generations, best_gen)


# ======================================================================
# PART 5 — main GA function (Final Test Portal)
# ======================================================================
def run_GA(comfort_rank, ticket_costs, weather, start_name):
    random.seed(42)
    np.random.seed(42)

    # --- 1. Read the time range for visiting attractions ---
    visit_intervals = load_spot_visit_intervals(DATABASE_PATH)

    # --- 2. Read the actual traffic matrix ---
    places, cost_matrix, time_matrix, mode_matrix = build_cost_and_time_matrices_real(
        TRANSPORT_PATH,
        time_period="offpeak",
        weather=weather,
        start_name=start_name
    )

    # --- 3. Read attraction appeal scores ---
    attraction_scores = load_attraction_scores(DATABASE_PATH)

    # --- 4. Assign a score of 0 to locations (such as hotels) with missing ratings in the database. ---
    for name in places:
        if name not in attraction_scores:
            attraction_scores[name] = 0.0

    for name in places:
        if name not in visit_intervals:
            visit_intervals[name] = (0.0, 0.0)

    # --- 5. Implement GA ---
    GA_POP_SIZE = 60
    GA_GENERATIONS = 300
    GA_CROSSOVER_RATE = 0.9
    GA_MUTATION_RATE = 0.1

    best_ind, best_cost, best_raw_cost, \
        best_total_time, best_total_traffic_cost, best_total_ticket_cost, \
        pop_size, generations, best_gen = GA_search(
        places, cost_matrix, time_matrix,
        visit_intervals, comfort_rank, ticket_costs, attraction_scores,
        pop_size=GA_POP_SIZE,
        generations=GA_GENERATIONS,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE)

    # --- 6. Output results ---
    mask, order = best_ind
    final_route = build_route(mask, order)

    print("【Final Route】")
    route_names = [start_name] + [places[i] for i in final_route] + [start_name]
    print(" → ".join(route_names))

    print(f"● Total travel duration: {best_total_time:.2f} hours")
    print(f"● Total traffic cost: RM {best_total_traffic_cost:.2f}")
    print(f"● Total ticket cost: RM {best_total_ticket_cost:.2f}")
    print(f"● Raw Total Cost (traffic + tickets): RM {best_raw_cost:.2f}")
    print(f"● Final Cost (for optimization): {best_cost:.2f}")

    print("\n------------------------------------------------")
    print("【Route Details】")
    # Traversal the route
    for idx, city in enumerate(final_route):
        if idx == 0:
            frm = start_name  # hotel
        else:
            frm = places[final_route[idx - 1]]
        to = places[city]

        mode = mode_matrix[final_route[idx - 1]][city] if idx > 0 else mode_matrix[0][city]

        print(f"{idx + 1}. {frm} → {to}")
        print(f"   ● Transportation mode: {mode}")
    # --- Add final return to hotel ---
    last_city = final_route[-1]
    back_mode = mode_matrix[last_city][0]  # back to index 0 (hotel)

    print(f"{len(final_route) + 1}. {places[last_city]} → {start_name}")
    print(f"   ● Transportation mode: {back_mode}")

    print("\n------------------------------------------------")
    print("【Algorithm Information of GA】")
    print(f"● Number of generations: {generations}")
    print(f"● Population size: {pop_size}")
    print(f"● Generation where the best solution first appeared: {best_gen}")



