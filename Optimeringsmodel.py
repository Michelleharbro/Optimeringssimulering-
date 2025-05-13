import pandas as pd
import numpy as np

# --- INDLÆS PASSAGERDATA ---
passenger_path = r"C:\Users\miche\OneDrive\Computer\Dokumenter\GBE\6. semester\Projekt\Passenger demand.xlsx"
df_passenger_raw = pd.read_excel(passenger_path, sheet_name=0)
df_passenger_raw.columns = df_passenger_raw.iloc[0]
df_passenger_raw = df_passenger_raw[1:]
df_passenger_raw.rename(columns={df_passenger_raw.columns[0]: "Time"}, inplace=True)
df_passenger_valid = df_passenger_raw[df_passenger_raw["Time"].str.contains(":", na=False)]
df_passenger_long = df_passenger_valid.melt(id_vars=["Time"], var_name="Weekday", value_name="Passenger Demand")
df_passenger_long["Passenger Demand"] = pd.to_numeric(df_passenger_long["Passenger Demand"], errors="coerce")
df_passenger_long["Hour"] = df_passenger_long["Time"].str.extract(r"(\d+):")[0].astype(int)

# --- INDLÆS GODSDATA ---
goods_path = r"C:\Users\miche\OneDrive\Computer\Dokumenter\GBE\6. semester\Projekt\Goods demand.xlsx"
df_goods_raw = pd.read_excel(goods_path, sheet_name=0)
df_goods_raw.columns = df_goods_raw.iloc[0]
df_goods_raw = df_goods_raw[1:]
df_goods_raw.rename(columns={df_goods_raw.columns[0]: "Time"}, inplace=True)
df_goods_valid = df_goods_raw[df_goods_raw["Time"].str.contains(":", na=False)]
df_goods_long = df_goods_valid.melt(id_vars=["Time"], var_name="Weekday", value_name="Goods Demand")
df_goods_long["Goods Demand"] = pd.to_numeric(df_goods_long["Goods Demand"], errors="coerce")
df_goods_long["Hour"] = df_goods_long["Time"].str.extract(r"(\d+):")[0].astype(int)

# --- MERGE DE TO ---
df_merged = pd.merge(df_passenger_long, df_goods_long, on=["Time", "Weekday", "Hour"])


# --- PARAMETRE (baseret på tekniske RAPID-forudsætninger) ---

# Togbegrænsninger
max_length = 816           # meter – maks længde inkl. passager- og godsvogne
max_weight = 1200          # ton – maks vægt ekskl. lokomotiv

# Vognspecifikationer
passengers_per_wagon = 100
containers_per_wagon = 2

passenger_wagon_length = 51.4   # meter
freight_wagon_length = 25.7     # meter

passenger_wagon_weight = 80     # ton
freight_wagon_weight = 64       # ton (vogn 30 ton + 2 containere 2*17 ton)

# Lokomotiv-ekstra kapacitet (kun til passagerer)
locomotive_passenger_capacity = 40  # ekstra kapacitet udover vogne

# Økonomi
wagon_cost = 500  # kr per vogn per time (samme for passager og gods)

# Billetpriser
ticket_peak = 159
ticket_offpeak = 59

# Containerpris (afstandsafhængig – bruges i optimering)
container_rate = 6*124  # kr per km per container

# Penaltys (kan justeres – 0 hvis man blot tracker backlog)
penalty_passenger_peak = 0
penalty_passenger_offpeak = 0
penalty_container = 0

# Peak-hours definition
peak_hours = list(range(6, 10)) + list(range(15, 19))  # fx 06–09 og 15–18

# Nedetid
downtime_chance = 0.06  # 6% chance for nedbrud i en given time

# Dynamisk vognbegrænsning (minimalt af vægt og længde)
max_wagons = int(min(
    max_weight / min(passenger_wagon_weight, freight_wagon_weight),
    max_length / min(passenger_wagon_length, freight_wagon_length)
))  # ≈ 19 vogne ved 64 tons og 25,7 m

# --- SIMULERING ---
results = []

# --- INITIAL BACKLOG ---
goods_backlog = 0  # fx 100 containere venter fra starten
passenger_backlog = 0  # passagerer starter typisk fra 0


for day in df_merged["Weekday"].unique():
    day_data = df_merged[df_merged["Weekday"] == day]

    for _, row in day_data.iterrows():
        hour = row["Hour"]

        p_demand = int(row["Passenger Demand"])
        g_demand = int(row["Goods Demand"])

        # 1. Total demand = aktuel + ventende fra tidligere time
        p_demand_total = int(row["Passenger Demand"]) + passenger_backlog
        g_demand_total = int(row["Goods Demand"]) + goods_backlog

        # 2. Nedbrud
        if np.random.rand() < downtime_chance:
            p_transported = 0
            g_transported = 0
            final_penalty = 0
            final_cost = 0
            final_revenue = 0
            note = "Nedbrud"
        else:
            # 3. Optimering: Find bedste vognfordeling
            best_profit = -np.inf

            for p_wagons in range(0, max_wagons + 1):
                for g_wagons in range(0, max_wagons + 1 - p_wagons):

                    total_weight = p_wagons * passenger_wagon_weight + g_wagons * freight_wagon_weight
                    total_length = p_wagons * passenger_wagon_length + g_wagons * freight_wagon_length

                    # Hvis løsningen overskrider vægt eller længde, spring den over
                    if total_weight > max_weight or total_length > max_length:
                        continue

                    # Beregner kapacitet og faktisk transport
                    p_cap = p_wagons * passengers_per_wagon
                    if p_demand > 0:
                        p_cap += locomotive_passenger_capacity  # Lokomotiv bidrager med 40 pladser

                    g_cap = g_wagons * containers_per_wagon

                    p_trans = min(p_demand, p_cap)
                    g_trans = min(g_demand, g_cap)

                    price = ticket_peak if hour in peak_hours else ticket_offpeak
                    penalty_passenger = penalty_passenger_peak if hour in peak_hours else penalty_passenger_offpeak
                    revenue = p_trans * price + g_trans * container_rate
                    cost = (p_wagons + g_wagons) * wagon_cost
                    penalty = (p_demand - p_trans) * penalty_passenger + (g_demand - g_trans) * penalty_container
                    profit = revenue - cost - penalty

                if profit > best_profit:
                    best_profit = profit
                    p_transported = p_trans
                    g_transported = g_trans
                    final_penalty = penalty
                    final_revenue = revenue

            note = ""

        # 4. Opdater backlog EFTER transport er beregnet
        passenger_backlog = max(0, p_demand - p_transported)
        goods_backlog = max(0, g_demand - g_transported)

        # Beregn faktisk brugte vogne baseret på transporteret antal
        actual_p_wagons_used = int(np.ceil(p_transported / passengers_per_wagon))
        actual_g_wagons_used = int(np.ceil(g_transported / containers_per_wagon))
        final_cost = (actual_p_wagons_used + actual_g_wagons_used) * wagon_cost

        # Beregn hvor mange vogne der ville være nødvendige for at dække al efterspørgsel
        passenger_wagons_needed = np.ceil(p_demand / passengers_per_wagon)
        goods_wagons_needed = np.ceil(g_demand / containers_per_wagon)

        results.append({
            "Day": day,
            "Hour": hour,
            "Passenger Demand": p_demand,
            "Goods Demand": g_demand,
            "Passenger demand + backlog": p_demand_total,
            "Goods demand + backlog": g_demand_total,
            "Passenger Transported": p_transported,
            "Goods Transported": g_transported,
            "Actual Passenger Wagons Used": actual_p_wagons_used,
            "Actual Freight Wagons Used": actual_g_wagons_used,
            "Passenger Backlog": passenger_backlog,
            "Goods Backlog": goods_backlog,
            "Passenger Wagons Needed": passenger_wagons_needed,
            "Goods Wagons Needed": goods_wagons_needed,
            "Total Revenue": final_revenue,
            "Wagon Cost": final_cost,
            "Penalty Cost": final_penalty,
            "Profit": best_profit,
            "Note": note
        })

# --- RESULTATER ---
df_final = pd.DataFrame(results)
df_final.to_excel(r"C:\Users\miche\OneDrive\Computer\Dokumenter\combined_simulation_result.xlsx", index=False)

print(df_final.head(40))


