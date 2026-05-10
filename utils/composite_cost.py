import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import numpy as np


def parse_ticket_price(value):
    if isinstance(value, str):
        value = value.strip()

        if value.lower() == "free":
            return 0.0

        if "-" in value:
            parts = value.split("-")
            try:
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2.0
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


def load_base_data(excel_path="database1.xlsx"):
    data = pd.read_excel(excel_path)[[
        'Name', 'Latitude', 'Longitude',
        'TicketFee_RM(Malaysian)',
        'TicketFee_RM(Non-Malaysian)',
        'Attraction_Score'
    ]]
    places = data['Name'].tolist()
    latitudes = data['Latitude'].tolist()
    longitudes = data['Longitude'].tolist()
    return data, places, latitudes, longitudes


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def calc_subway_fare(stations_diff):
    if stations_diff <= 3:
        return 1.4
    elif stations_diff <= 6:
        return 2.2
    elif stations_diff <= 10:
        return 3.0
    else:
        return min(5.0, 1.4 + 0.2 * (stations_diff - 3))


def calc_taxi_cost(base_cost, weather='sunny'):
    """Return taxi cost multiplier by weather only.
    """
    multiplier = 1.0
    if weather == 'rainy':
        multiplier = 1.5
    return round(base_cost * multiplier, 2)


def build_ticket_price_mapping(data, start_name, local_user):
    ticket_column = "TicketFee_RM(Malaysian)" if local_user else "TicketFee_RM(Non-Malaysian)"

    ticket_price = {}
    ticket_price[start_name] = 0.0

    for _, row in data.iterrows():
        name = row["Name"]
        raw_value = row[ticket_column]
        numeric_value = parse_ticket_price(raw_value)
        ticket_price[name] = numeric_value

    return ticket_price


def build_distance_matrix(places_with_start, latitudes_with_start, longitudes_with_start):
    n = len(places_with_start)
    dist_matrix = pd.DataFrame(index=places_with_start, columns=places_with_start, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix.iloc[i, j] = 0.0
            else:
                dist_matrix.iloc[i, j] = round(
                    haversine(
                        latitudes_with_start[i], longitudes_with_start[i],
                        latitudes_with_start[j], longitudes_with_start[j]
                    ),
                    2
                )
    return dist_matrix


def compute_weights(time_rank, comfort_rank, cost_rank, weather):
    rank_to_weight = {1: 0.5, 2: 0.3, 3: 0.2}
    alpha = rank_to_weight[time_rank]
    gamma = rank_to_weight[comfort_rank]
    beta = rank_to_weight[cost_rank]

    if weather == "rainy":
        alpha += 0.2
        gamma += 0.3
        beta -= 0.1
        
    if cost_rank == 1:
        beta += 0.2
        alpha -= 0.1
        gamma -= 0.1

    alpha = max(alpha, 0.05)
    beta = max(beta, 0.05)
    gamma = max(gamma, 0.05)

    total = alpha + beta + gamma
    alpha, beta, gamma = alpha / total, beta / total, gamma / total

    return alpha, beta, gamma


def build_transport_dataframe(
    places_with_start,
    latitudes_with_start,
    longitudes_with_start,
    ticket_price,
    weather,
    time_rank,
    comfort_rank,
    cost_rank,
    random_seed=42
):
    np.random.seed(random_seed)

    n = len(places_with_start)
    dist_matrix = build_distance_matrix(places_with_start, latitudes_with_start, longitudes_with_start)

    transport_df = pd.DataFrame(columns=[
        'Start', 'End',
        'Walk_Time', 'Walk_Cost',
        'Subway_Time', 'Subway_Cost',
        'Taxi_Time', 'Taxi_Cost',
        'Weather'
    ])
    weathers_all = ['sunny', 'rainy']

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            d = dist_matrix.iloc[i, j]
            walk_speed = np.random.uniform(3, 5)
            subway_speed = np.random.uniform(35, 45)
            taxi_speed = np.random.uniform(30, 50)

            stations_diff = max(1, round(d))
            subway_cost = round(calc_subway_fare(stations_diff), 2)

            for w in weathers_all:
                    base_taxi_cost = 10 + d * 2
                    taxi_cost = calc_taxi_cost(base_taxi_cost, w)
                    row = {
                        'Start': places_with_start[i],
                        'End': places_with_start[j],
                        'Walk_Time': round(d / walk_speed, 2),
                        'Distance_km': d,
                        'Walk_Cost': 0,
                        'Subway_Time': round(d / subway_speed, 2),
                        'Subway_Cost': subway_cost,
                        'Taxi_Time': round(d / taxi_speed, 2),
                        'Taxi_Cost': taxi_cost,
                        'Weather': w
                    }
                    transport_df = pd.concat(
                        [transport_df, pd.DataFrame([row])],
                        ignore_index=True
                    )


    alpha, beta, gamma = compute_weights(time_rank, comfort_rank, cost_rank, weather)

    transport_df["Fatigue_Walk"] = transport_df["Walk_Time"] * (1 + gamma)
    transport_df["Fatigue_Subway"] = transport_df["Subway_Time"] * (0.3 + 0.7 * gamma)
    transport_df["Fatigue_Taxi"] = transport_df["Taxi_Time"] * (0.5 + 0.5 * gamma)

   
