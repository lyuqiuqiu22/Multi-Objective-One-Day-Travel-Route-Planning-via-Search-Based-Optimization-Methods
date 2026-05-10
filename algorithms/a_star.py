import pandas as pd
import heapq
import numpy as np


# =============================
#  Data Loading and Utilities
# =============================

def load_attractions_data():
    return pd.read_excel('database.xlsx')


def parse_ticket_price(value):
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "free":
            return 0.0
        if "-" in value:
            try:
                low, high = map(float, value.split("-"))
                return (low + high) / 2
            except:
                return 0.0
        try:
            return float(value)
        except:
            return 0.0

    try:
        return float(value)
    except:
        return 0.0


def determine_day_limit(comfort_rank):
    return 8 if comfort_rank == 1 else 10 if comfort_rank == 2 else 12


# =============================
#  Transportation Lookup
# =============================

def get_best_transport(start, end, df):
    """Unified best transport reader"""
    record = None

    # direct match
    rec = df[(df['Start'] == start) & (df['End'] == end)]
    if len(rec) > 0:
        record = rec.iloc[0]
    else:
        # reverse match
        rec = df[(df['Start'] == end) & (df['End'] == start)]
        if len(rec) > 0:
            record = rec.iloc[0]

    if record is None:
        return {
            'start': start, 'end': end,
            'best_mode': 'taxi',
            'travel_time': 0.5,
            'composite_cost': 5.0,
            'distance': 0
        }

    # choose best mode
    costs = {
        'walk': record['Walk_Cost_Composite'],
        'subway': record['Subway_Cost_Composite'],
        'taxi': record['Taxi_Cost_Composite']
    }
    best_mode = min(costs, key=costs.get)

    if best_mode == 'walk':
        travel_time = record['Walk_Time']
    elif best_mode == 'subway':
        travel_time = record['Subway_Time']
    else:
        travel_time = record['Taxi_Time']

    return {
        'start': start,
        'end': end,
        'best_mode': best_mode,
        'travel_time': travel_time,
        'composite_cost': record['Composite_Cost'],
        'distance': record['Distance_km']
    }


# =============================
#  A* Search (Core Logic)
# =============================

def a_star_time_limit_tsp(start_name, attractions_info, time_limit_hours, filtered_transport, local_user):
    all_attractions = [name for name in attractions_info.keys() if name != start_name]

    # precompute transport table
    all_transport = {
        (loc1, loc2): get_best_transport(loc1, loc2, filtered_transport)
        for loc1 in [start_name] + all_attractions
        for loc2 in [start_name] + all_attractions
        if loc1 != loc2
    }

    class State:
        def __init__(self, visited, loc, used_time, cost, path, h=0):
            self.visited = visited
            self.loc = loc
            self.used_time = used_time
            self.cost = cost
            self.path = path
            self.h = h

        def __lt__(self, other):
            return (self.cost - self.h) < (other.cost - other.h)

        def return_cost(self):
            if self.loc == start_name:
                return 0
            return all_transport[(self.loc, start_name)]['composite_cost']

        def effective_cost(self):
            return self.cost + self.return_cost()

        def heuristic(self, remaining):
            remain_time = time_limit_hours - self.used_time
            if remain_time <= 0:
                return 0

            candidates = []
            for a in remaining:
                t1 = all_transport[(self.loc, a)]['travel_time']
                t2 = all_transport[(a, start_name)]['travel_time']
                vt = attractions_info[a]['visit_time']
                total = t1 + vt + t2
                if total <= remain_time:
                    candidates.append((attractions_info[a]['attraction_score'], total))

            if not candidates:
                return 0

            candidates.sort(key=lambda x: x[0] / x[1], reverse=True)

            score = 0
            time_left = remain_time
            for sc, need in candidates:
                if need <= time_left:
                    score += sc
                    time_left -= need
            return -score * 10

    pq = []
    init = State(set(), start_name, 0, 0, [start_name])
    init.h = init.heuristic(all_attractions)
    heapq.heappush(pq, init)

    best = None
    visited_states = {}
    max_iter = 50000
    iteration = 0
    pushed_nodes = 1  # initial push
    max_pq_size = len(pq)
    while pq and iteration < max_iter:
        iteration += 1
        cur = heapq.heappop(pq)

        if len(pq) > max_pq_size:
            max_pq_size = len(pq)

        key = (frozenset(cur.visited), cur.loc, round(cur.used_time, 2))
        if key in visited_states and visited_states[key] <= cur.cost:
            continue
        visited_states[key] = cur.cost

        cur_count = len(cur.visited)
        cur_score = sum(attractions_info[a]['attraction_score'] for a in cur.visited)
        cur_cost = cur.effective_cost()

        if best is None:
            update = True
        else:
            b_count = len(best['visited'])
            b_cost = best['effective']
            b_score = best['score']

            if cur_count == b_count:
                update = cur_cost < b_cost or (cur_cost == b_cost and cur_score > b_score)
            else:
                update = cur_count > b_count

        if update:
            best = {
                'visited': cur.visited.copy(),
                'loc': cur.loc,
                'used': cur.used_time,
                'cost': cur.cost,
                'effective': cur_cost,
                'path': cur.path.copy(),
                'score': cur_score,
            }

        remaining = [a for a in all_attractions if a not in cur.visited]

        for nxt in remaining:
            t = all_transport[(cur.loc, nxt)]
            new_time = cur.used_time + t['travel_time'] + attractions_info[nxt]['visit_time']
            ret_time = all_transport[(nxt, start_name)]['travel_time']

            if new_time + ret_time > time_limit_hours:
                continue

            ticket = attractions_info[nxt]['ticket_price_local'] if local_user else attractions_info[nxt]['ticket_price_non_local']
            new_cost = cur.cost + t['composite_cost'] + ticket

            new_visited = cur.visited.copy()
            new_visited.add(nxt)

            st = State(new_visited, nxt, new_time, new_cost, cur.path + [nxt])
            st.h = st.heuristic([a for a in remaining if a != nxt])
            heapq.heappush(pq, st)

            pushed_nodes += 1
            if len(pq) > max_pq_size:
                max_pq_size = len(pq)

    # append return
    if best and best['path'][-1] != start_name:
        ret = all_transport[(best['path'][-1], start_name)]
        best['path'].append(start_name)
        best['used'] += ret['travel_time']
        best['cost'] += ret['composite_cost']
        best['effective'] = best['cost']

    if iteration >= max_iter:
        termination_reason = f"Reached maximum iteration limit ({max_iter})"
    else:
        # loop ends because pq is empty
        termination_reason = "Priority queue exhausted (no more states to expand)"
        # ----------------------------------------------------

    stats = {
        'expanded_nodes': iteration,
        'pushed_nodes': pushed_nodes,
        'unique_states': len(visited_states),
        'max_pq_size': max_pq_size,
        'termination_reason': termination_reason,
        'max_iter': max_iter
    }

    return {'best': best, 'stats': stats}


# =============================
#  Main Output (Clean Display)
# =============================

def run_a_star_planner(start_name, local_user, weather, comfort_rank):
    # load transport
    transport_df = pd.read_excel("transport_composite_cost.xlsx")

    # load attractions
    data = load_attractions_data()

    day_limit = determine_day_limit(comfort_rank)

    filtered_transport = transport_df[(transport_df['Weather'] == weather)]

    # attractions info
    attractions_info = {}
    for _, row in data.iterrows():
        name = row['Name']
        visit = str(row['VisitTime_hr'])

        if '-' in visit:
            low, high = map(float, visit.split('-'))
            mu = (low + high) / 2
            sigma = (high - low) / 6 if high > low else 0.1 * mu
        else:
            try:
                mu = float(visit)
                sigma = 0.1 * mu
            except:
                mu, sigma = 1.0, 0.1

        visit_time = max(0.1, np.random.normal(mu, sigma))

        attractions_info[name] = {
            'visit_time': visit_time,
            'attraction_score': float(row['Attraction_Score']),
            'ticket_price_local': parse_ticket_price(row['TicketFee_RM(Malaysian)']),
            'ticket_price_non_local': parse_ticket_price(row['TicketFee_RM(Non-Malaysian)'])
        }

    attractions_info[start_name] = {
        'visit_time': 0,
        'attraction_score': 0,
        'ticket_price_local': 0,
        'ticket_price_non_local': 0
    }

    # run A*
    result = a_star_time_limit_tsp(start_name, attractions_info, day_limit, filtered_transport, local_user)
    solution = result.get('best')
    stats = result.get('stats', {})
    if not solution:
        print("❌ No route meets the time limit.")
        return

    path = solution['path']
    total_time = solution['used']
    total_score = solution['score']

    total_ticket = 0
    total_travel = 0
    route_details = []

    for i in range(1, len(path)):
        f, t = path[i-1], path[i]
        info = get_best_transport(f, t, filtered_transport)
        total_travel += info['composite_cost']

        if t != start_name:
            total_ticket += attractions_info[t]['ticket_price_local'] if local_user else attractions_info[t]['ticket_price_non_local']

        route_details.append((f, t, info['best_mode']))

    final_cost = solution['effective']

    print("【Final Route】")
    print(" → ".join(path))
    print(f"● Total travel duration: {total_time:.2f} hours")
    print(f"● Total traffic cost: RM {total_travel:.2f}")
    print(f"● Total ticket cost: RM {total_ticket:.2f}")
    print(f"● Final Cost (for optimization): {final_cost:.2f}")

    print("\n------------------------------------------------")
    print("\n【Route Details】")
    for i, (f, t, mode) in enumerate(route_details, 1):
        print(f"{i}. {f} → {t}")
        print(f" ● Transportation mode: {mode}")

    print("\n------------------------------------------------")
    print("\n【Algorithm Information】")
    print("● Algorithm type: A* Search")
    print("● Heuristic: Remaining score estimation with return constraint")
    print(f"● Expanded nodes: {stats.get('expanded_nodes', 'N/A')}")
    print(f"● Unique states visited: {stats.get('unique_states', 'N/A')}")
    print(f"● Termination reason: {stats.get('termination_reason', 'N/A')}")
