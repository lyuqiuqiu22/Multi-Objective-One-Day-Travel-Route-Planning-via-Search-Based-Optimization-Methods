import numpy as np


# choose the best transportation mode with the lowest composite cost
def get_best_edge(start, end, transport_df, weather, time_period):
    df = transport_df[
        (transport_df["Start"] == start) &
        (transport_df["End"] == end) &
        (transport_df["Weather"] == weather) &
        (transport_df["Time_Period"] == time_period)
        ]

    if df.empty:
        return None  # no available record

    row = df.iloc[0]  # take the first matched row

    costs = {  # composite costs of 3 modes
        "Walk": row["Walk_Cost_Composite"],
        "Subway": row["Subway_Cost_Composite"],
        "Taxi": row["Taxi_Cost_Composite"]
    }

    best_mode = min(costs, key=costs.get)  # select the smallest cost

    if best_mode == "Walk":
        t = row["Walk_Time"]
    elif best_mode == "Subway":
        t = row["Subway_Time"]
    else:
        t = row["Taxi_Time"]

    return {
        "mode": best_mode,
        "travel_time": t,
        "travel_cost": costs[best_mode]
    }


# main greedy algorithm: select next stop by minimal local cost
def greedy_search(
        start_name,
        places_with_start,
        transport_df,
        ticket_price,
        visit_time_map,
        weather,
        time_period,
        day_limit_hours
):
    current = start_name
    unvisited = [p for p in places_with_start if p != start_name]

    route = [start_name]  # record travel sequence
    edges = []  # record detailed transitions
    total_time = 0.0  # total travel + stay hours
    total_cost = 0.0  # sum of travel + ticket cost

    while unvisited:

        candidates = []  # store feasible next choices

        for dst in unvisited:

            edge = get_best_edge(current, dst, transport_df, weather, time_period)
            if edge is None:
                continue  # skip if no transport data

            stay = visit_time_map[dst]  # stay duration
            travel_time = edge["travel_time"]

            return_edge = get_best_edge(dst, start_name, transport_df, weather, time_period)
            if return_edge:
                return_time = return_edge["travel_time"]
            else:
                return_time = float('inf')

            if total_time + stay + travel_time + return_time > day_limit_hours:
                continue

            ticket = ticket_price[dst]  # entrance fee
            local_cost = edge["travel_cost"] + ticket  # greedy cost criterion

            candidates.append((local_cost, dst, edge, stay, ticket))

        if not candidates:
            break  # no more feasible destinations

        candidates.sort(key=lambda x: x[0])  # choose minimal local cost
        best_cost, best_dst, best_edge, best_stay, best_ticket = candidates[0]

        route.append(best_dst)  # update route
        total_cost += best_cost  # update total cost
        total_time += best_edge["travel_time"] + best_stay  # update total time

        edges.append({
            "from": current,
            "to": best_dst,
            "mode": best_edge["mode"],
            "travel_time": best_edge["travel_time"],
            "stay_time": best_stay,
            "ticket": best_ticket,
            "local_cost": best_cost
        })

        current = best_dst  # update current location
        unvisited.remove(best_dst)  # mark as visited

    if current != start_name:  # if not at hotel at the end
        # calculate transportation back to hotel
        return_edge = get_best_edge(current, start_name, transport_df, weather, time_period)

        if return_edge:
            return_travel_time = return_edge["travel_time"]
            return_cost = return_edge["travel_cost"]

            # ALWAYS return to hotel regardless of time limit
            route.append(start_name)  # add hotel to end of route
            total_time += return_travel_time
            total_cost += return_cost

            # add return edge record
            edges.append({
                "from": current,
                "to": start_name,
                "mode": return_edge["mode"],
                "travel_time": return_travel_time,
                "stay_time": 0.0,  # no visit when returning to hotel
                "ticket": 0.0,  # no ticket for hotel
                "local_cost": return_cost
            })

            # update current location to hotel
            current = start_name

    return route, edges, total_time, total_cost


# wrapper so main.py can call this algorithm
def run_greedy(start_name,
               places_with_start,
               transport_df,
               ticket_price,
               visit_time_map,
               weather,
               time_period,
               day_limit_hours):
    # generate a reasonable stay duration for each attraction
    def sample_visit_time(visit_time_str):
        if "-" in visit_time_str:  # if time is a range
            low, high = map(float, visit_time_str.split("-"))  # parse boundaries
            mu = (low + high) / 2.0  # mean of the range
            sigma = (high - low) / 6.0  # narrow distribution
        else:
            value = float(visit_time_str)  # fixed duration
            mu = value  # mean equals value
            sigma = 0.1 * value  # small variation

        while True:
            t = np.random.normal(mu, sigma)  # sample stay duration
            if t > 0:
                return t  # ensure positive value

    # ========== Convert string format to numeric format ==========
    sampled_visit_time_map = {}
    for place, time_str in visit_time_map.items():
        if place == start_name:
            sampled_visit_time_map[place] = 0.0  # hotel stays 0 hours
        else:
            sampled_visit_time_map[place] = sample_visit_time(time_str)


    # run algorithm
    route, edges, total_time, total_cost = greedy_search(
        start_name,
        places_with_start,
        transport_df,
        ticket_price,
        sampled_visit_time_map,
        weather,
        time_period,
        day_limit_hours
    )

    # ----------- Format all numbers to 2 decimal places -----------
    total_time = round(total_time, 2)
    total_cost = round(total_cost, 2)

    # format all values in edges
    for edge in edges:
        edge["travel_time"] = round(edge["travel_time"], 2)
        edge["stay_time"] = round(edge["stay_time"], 2)
        edge["ticket"] = round(edge["ticket"], 2)
        edge["local_cost"] = round(edge["local_cost"], 2)

    # calculate summary costs
    traffic_cost = round(sum(e["local_cost"] - e["ticket"] for e in edges), 2)
    ticket_cost = round(sum(e["ticket"] for e in edges), 2)


    # Final Route
    print("【Final Route】")
    print(" → ".join(route))


    print(f"\n● Total travel duration: {total_time:.2f} hours")
    print(f"● Total traffic cost: RM {traffic_cost:.2f}")
    print(f"● Total ticket cost: RM {ticket_cost:.2f}")
    print(f"● Final Total Cost (for optimization): RM {total_cost:.2f}")

    # Route Details
    print("\n[Route Details]")
    for i, e in enumerate(edges, start=1):
        print(f"{i}. {e['from']} → {e['to']}")
        print(f"   ● Transportation mode: {e['mode'].lower()}")

    # Information
    print("\n[Algorithm Information]")
    print("● Algorithm Type: Greedy")
    print("● Decision Rule: Select next stop with minimum local cost")
    print("● No backtracking / no global search")
    print("● Fast but not globally optimal")

    return route, edges, total_time, total_cost