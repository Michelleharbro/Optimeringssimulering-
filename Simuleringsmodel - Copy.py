import pandas as pd
import numpy as np
import time
import re
import random
from collections import defaultdict

# Dynamisk seed baseret på systemets tilfældighedskilde
random_gen = random.SystemRandom()
seed = random_gen.randint(0, 2 ** 32 - 1)  # Tildel seed en værdi
np.random.seed(seed)  # Brug dette seed til NumPy
print(f"Seed used: {seed}")  # Udskriv det anvendte seed

# Konfigurer pandas til at vise alle rækker og kolonner
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Sti til Excel-filen
file_path = "/Users/kristofferkim/Desktop/Simulering/InputdataMedNedbrud.xlsx"

# Læs alle faner fra Excel
data = pd.read_excel(file_path, sheet_name=None)

# Læs Kapacitet, Priser og Afgange fra Excel
kapacitet = data["Kapacitet"]
priser = data["Priser"]
afgange = data["Afgange"]

# Afstande fra byerne til Kauslunde (i km)
distances_to_kauslunde = {
    "Aalborg": 264,
    "Odense": 45,
    "Esbjerg": 97,
    "Herning": 114,
    "Kastrup": 212,
    "Kbh": 206,
    "Ringsted": 142,
    "Padborg": 88,
    "Glostrup": 186,
    "Århus": 124,
    "Kauslunde": 0
}

# Parse fanenavne for at finde byerne
city_data = {}
for sheet_name, sheet_data in data.items():
    match = re.match(r"(\w+)_(pass|cont)_(int|dest)", sheet_name)
    if match:
        city, entity, data_type = match.groups()
        if city not in city_data:
            city_data[city] = {}
        city_data[city][f"{entity}_{data_type}"] = sheet_data

# Få listen over alle byerne i simuleringen
cities = list(city_data.keys())


# Helper-funktioner

def debug_distribution_stats(params_df):
    """Prints debugging information for distributions"""
    print("\nAnalyzing distributions and parameters...")
    days_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Sørg for, at kommaer bliver erstattet af punktum i Excel-data, hvis komma bruges som decimaltegn
    for day in days_map:
        for hour in range(24):
            dist_col = f"{day}_Distribution"
            mu_col = f"{day}_mu"
            sigma_col = f"{day}_sigma"

            params_df[mu_col] = params_df[mu_col].str.replace(',', '.').astype(float)
            params_df[sigma_col] = params_df[sigma_col].str.replace(',', '.').astype(float)

            # Hent distributionsparametre for hver time og dag
            dist_type = params_df.loc[params_df["Hour"] == hour, dist_col].values[0]
            mu = params_df.loc[params_df["Hour"] == hour, mu_col].values[0]
            sigma = params_df.loc[params_df["Hour"] == hour, sigma_col].values[0]
            print(f"Day: {day}, Hour: {hour}, Distribution: {dist_type}, Mu: {mu}, Sigma: {sigma}")


def clean_probabilities(dest_df):
    """Cleans and normalizes the probabilities in the destination dataframe"""
    if pd.api.types.is_numeric_dtype(dest_df["Probability"]):
        numeric_probs = dest_df["Probability"]
    else:
        # Hvis 'Probability' er en streng, skal vi konvertere den
        numeric_probs = pd.to_numeric(dest_df["Probability"].str.replace(",", "."), errors='coerce').fillna(0)

    dest_df["Probability"] = numeric_probs / numeric_probs.sum()

    return dest_df

def generate_interarrival_times(params_df):
    """Generates interarrival times based on distribution parameters"""
    times = []
    debug_stats = []
    arrivals_per_hour = {}  # Dictionary to keep track of arrivals per hour per day

    days_map = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day_index, day in enumerate(days_map):
        arrivals_per_hour[day] = []  # Initialize list for each day

        for hour in range(24):
            dist_col = f"{day}_Distribution"
            mu_col = f"{day}_mu"
            sigma_col = f"{day}_sigma"

            dist_type = params_df.loc[params_df["Hour"] == hour, dist_col].values[0]
            mu = params_df.loc[params_df["Hour"] == hour, mu_col].values[0]
            sigma = params_df.loc[params_df["Hour"] == hour, sigma_col].values[0]

            # Sæt seed inden generering for at sikre tilfældighed
            np.random.seed(random.SystemRandom().randint(0, 2 ** 32 - 1))

            arrivals = []

            # Generér stokastisk baseret på fordelingen
            if dist_type.lower() == "expon":
                arrivals = np.random.exponential(scale=mu, size=int(60 / mu))
            elif dist_type.lower() == "norm":
                arrivals = np.random.normal(loc=mu, scale=sigma, size=int(60 / mu))
                arrivals = arrivals[arrivals > 0] 
            elif dist_type.lower() == "exponweib":
                c = 1.5  # Shape parameter for Weibull
                arrivals = np.random.weibull(c, size=int(60 / mu)) * mu
            elif dist_type.lower() == "lognorm":
                s = sigma  # Standard deviation
                scale = np.exp(mu)  # Scale parameter
                arrivals = np.random.lognormal(mean=mu, sigma=s, size=int(60 / mu))
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

            # Tilføj antal ankomster per time til dictionary
            arrivals_per_hour[day].append(len(arrivals))

            # Debug-statistik
            debug_stats.append((hour, day, dist_type, mu, sigma, len(arrivals)))

            # Konverter til kumulative tider og tilføj til listen
            cumulative_times = np.cumsum(arrivals) + day_index * 1440 + hour * 60
            times.extend(cumulative_times[cumulative_times < (day_index + 1) * 1440])

    # Udskriv debugging-statistik
    print("\nGenerated interarrival times stats:")
    for stat in debug_stats:
        hour, day, dist_type, mu, sigma, count = stat
        print(f"Hour {hour}, {day}: {dist_type}, mu = {mu}, sigma = {sigma}, arrivals generated = {count}")

    return times, arrivals_per_hour


def assign_destinations(num_arrivals, dest_df):
    """Assigns destinations based on probabilities"""
    probabilities = dest_df["Probability"].values
    destination_labels = dest_df["Destination"].values
    destinations = [np.random.choice(destination_labels, p=probabilities) for _ in range(num_arrivals)]
    return destinations

def create_od_matrix(origins, destinations):
    """Creates an OD matrix from origin and destination data"""
    od_df = pd.DataFrame({'Origin': origins, 'Destination': destinations})
    od_matrix = pd.crosstab(od_df['Origin'], od_df['Destination'])
    return od_matrix

def calculate_revenue(passenger_count, container_count, distances, ticket_price):
    """Calculates revenue from passengers and containers"""
    passenger_revenue = passenger_count * ticket_price
    container_revenue = container_count * distances * 6  # Container pris pr. km
    return passenger_revenue, container_revenue

# Behandling af data for hver by
global_passenger_od = []
global_container_od = []
total_passenger_revenue = 0
total_container_revenue = 0
train_schedule = []


# For tracking transported passengers and containers
transported_passenger_data = []
transported_container_data = []


# To track waiting and late arrivals for creating OD matrices
waiting_passenger_data = []
waiting_container_data = []
late_passenger_data = []
late_container_data = []

# Set til at spore unikke forsinkelser
unique_late_passengers = set()
unique_late_containers = set()

# Log kapacitet, ankomster og ventende per time
hourly_log_data = []

# New data structure to track train capacity at Kauslunde
kauslunde_train_capacity = defaultdict(lambda: {'passenger_capacity': 0, 'container_capacity': 0})


# Generer interarrival-tider og behandl data
for city, city_sheets in city_data.items():
    print(f"\nGenerating interarrival times for city: {city}")
    # Generer interarrival-tider og antallet af ankomster for hver time
    _, passenger_arrivals_per_hour = generate_interarrival_times(city_sheets["pass_int"])
    _, container_arrivals_per_hour = generate_interarrival_times(city_sheets["cont_int"])

    city_sheets["pass_dest"] = clean_probabilities(city_sheets["pass_dest"])
    city_sheets["cont_dest"] = clean_probabilities(city_sheets["cont_dest"])

    # Globale lister for at tracke ventende passagerer og containere over hele simuleringen
    global_waiting_passengers = []  # Track passengers waiting across multiple days
    global_waiting_containers = []  # Track containers waiting across multiple days

    # Gennemgå hver dag og time, og håndtér ventende passagerer og containere
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        # Bemærk at vi ikke nulstiller 'waiting_passengers' og 'waiting_containers' hver dag, de holdes globalt
        for hour in range(24):
            # Beregn samlede ankomster som ankomster denne time + ventende fra sidste time
            passenger_arrivals = passenger_arrivals_per_hour[day][hour] + len(global_waiting_passengers)
            container_arrivals = container_arrivals_per_hour[day][hour] + len(global_waiting_containers)

            # Assign destinations to the arriving passengers and containers
            passenger_destinations = assign_destinations(passenger_arrivals, city_sheets["pass_dest"])
            container_destinations = assign_destinations(container_arrivals, city_sheets["cont_dest"])

            # Hent kapacitet for den givne by, dag og time
            kapacitet_row = kapacitet.loc[
                (kapacitet["City"] == city) &
                (kapacitet["Day"] == day) &
                (kapacitet["Hour"] == hour)
                ]

            if kapacitet_row.empty:
                passenger_count = 0
                container_count = 0
            else:
                # Kapaciteten er begrænset af togkapaciteten
                passenger_count = min(kapacitet_row["Kapacitet passager"].values[0], passenger_arrivals)
                container_count = min(kapacitet_row["Kapacitet container"].values[0], container_arrivals)

            # Tilføj det nye kodeblok her for at logge transporterede passagerer og containere.
            for dest in passenger_destinations[:passenger_count]:
                transported_passenger_data.append({'Origin': city, 'Destination': dest, 'Transported': 1})

            for dest in container_destinations[:container_count]:
                transported_container_data.append({'Origin': city, 'Destination': dest, 'Transported': 1})

            # Fordel ventende passagerer og containere (de der ikke kom med)
            global_waiting_passengers = passenger_destinations[passenger_count:]
            global_waiting_containers = container_destinations[container_count:]

            # Tilføj ventende passagerer og containere til data til senere brug i OD-matricer (Total Misses)
            for dest in global_waiting_passengers:
                waiting_passenger_data.append({'Origin': city, 'Destination': dest, 'Waiting': 1})
            for dest in global_waiting_containers:
                waiting_container_data.append({'Origin': city, 'Destination': dest, 'Waiting': 1})

            # Beregn omsætning for passagerer og containere
            ticket_price = priser.loc[
                (priser["Day"] == day) & (priser["Hour"] == hour), "Pris [DKK]"
            ].values[0]
            passenger_revenue, container_revenue = calculate_revenue(
                passenger_count, container_count, distances_to_kauslunde[city], ticket_price
            )
            total_passenger_revenue += passenger_revenue
            total_container_revenue += container_revenue

            # Fordel ventende passagerer og containere (de der ikke kom med)
            waiting_passengers = passenger_destinations[passenger_count:]
            waiting_containers = container_destinations[container_count:]

            # Tilføj ventende passagerer og containere til data til senere brug i OD-matricer (Total Misses)
            # Tilføj ventende passagerer til forsinkelseslisten (late_passenger_data)
            for dest in waiting_passengers:
                key = (city, dest)  # Brug en tuple som nøgle
                if key not in unique_late_passengers:
                    unique_late_passengers.add(key)
                    late_passenger_data.append({'Origin': city, 'Destination': dest, 'Late': 1})
                    print(f"Late passenger added: {city} -> {dest}")

            for dest in waiting_containers:
                waiting_container_data.append({'Origin': city, 'Destination': dest, 'Waiting': 1})

            # Tilføj til unikke forsinkelser (Unique Misses) kun hvis de ikke allerede er set
            for dest in waiting_passengers:
                key = (city, dest)  # Brug en tuple som nøgle for at holde styr på parret
                if key not in unique_late_passengers:
                    unique_late_passengers.add(key)
                    late_passenger_data.append({'Origin': city, 'Destination': dest, 'Late': 1})

            for dest in waiting_containers:
                key = (city, dest)  # Brug en tuple som nøgle for at holde styr på parret
                if key not in unique_late_containers:
                    unique_late_containers.add(key)
                    late_container_data.append({'Origin': city, 'Destination': dest, 'Late': 1})
            # Hent antal afgange for den aktuelle by, dag og time
            if not afgange.loc[
                (afgange["City"] == city) &
                (afgange["Day"] == day) &
                (afgange["Hour"] == hour)
            ].empty:
                num_departures = afgange.loc[
                    (afgange["City"] == city) &
                    (afgange["Day"] == day) &
                    (afgange["Hour"] == hour),
                    "Antal afgange"
                ].values[0]
            else:
                num_departures = 0

            # Log kapacitet, ankomster og ventende i listen
            hourly_log_data.append({
                'City': city,
                'Day': day,
                'Hour': hour,
                'Capacity Passengers': kapacitet_row['Kapacitet passager'].values[0] if not kapacitet_row.empty else 0,
                'Arrivals Passengers': passenger_arrivals,
                'Waiting Passengers': len(waiting_passengers),
                'Capacity Containers': kapacitet_row['Kapacitet container'].values[0] if not kapacitet_row.empty else 0,
                'Arrivals Containers': container_arrivals,
                'Waiting Containers': len(waiting_containers)
            })
            # Udskriv kapacitet, ankomster og ventende
            print(f"City: {city}, Day: {day}, Hour: {hour}")
            print(f"  Capacity Passengers: {kapacitet_row['Kapacitet passager'].values[0] if not kapacitet_row.empty else 0}, Arrivals: {passenger_arrivals}")
            print(f"  Capacity Containers: {kapacitet_row['Kapacitet container'].values[0] if not kapacitet_row.empty else 0}, Arrivals: {container_arrivals}")
            print(f"  Waiting Passengers: {len(waiting_passengers)}, Waiting Containers: {len(waiting_containers)}\n")

hourly_log_df = pd.DataFrame(hourly_log_data)

print("Late Passenger Data:", late_passenger_data)


# Opret DataFrames for forsinkede passagerer og containere
late_passenger_df = pd.DataFrame(late_passenger_data)
late_container_df = pd.DataFrame(late_container_data)

# Skab OD-matricer for forsinkede passagerer og containere (Unique Misses)
late_passenger_od_matrix = pd.crosstab(late_passenger_df['Origin'], late_passenger_df['Destination'], values=late_passenger_df['Late'], aggfunc='sum').fillna(0)
late_container_od_matrix = pd.crosstab(late_container_df['Origin'], late_container_df['Destination'], values=late_container_df['Late'], aggfunc='sum').fillna(0)


# Opret DataFrames for ventende passagerer og containere
waiting_passenger_df = pd.DataFrame(waiting_passenger_data)
waiting_container_df = pd.DataFrame(waiting_container_data)

# Opret OD-matricer for ventende passagerer og containere
waiting_passenger_od_matrix = pd.crosstab(waiting_passenger_df['Origin'], waiting_passenger_df['Destination'], values=waiting_passenger_df['Waiting'], aggfunc='sum').fillna(0)
waiting_container_od_matrix = pd.crosstab(waiting_container_df['Origin'], waiting_container_df['Destination'], values=waiting_container_df['Waiting'], aggfunc='sum').fillna(0)

# Skab OD-matricer for forsinkede passagerer og containere (Unique Misses)
late_passenger_od_matrix = pd.crosstab(late_passenger_df['Origin'], late_passenger_df['Destination'], values=late_passenger_df['Late'], aggfunc='sum').fillna(0)
late_container_od_matrix = pd.crosstab(late_container_df['Origin'], late_container_df['Destination'], values=late_container_df['Late'], aggfunc='sum').fillna(0)


# Opret OD-matricer for transporterede passagerer og containere
transported_passenger_df = pd.DataFrame(transported_passenger_data)
transported_container_df = pd.DataFrame(transported_container_data)

# Skab OD-matricer for transporterede passagerer og containere
transported_passenger_od_matrix = pd.crosstab(transported_passenger_df['Origin'], transported_passenger_df['Destination'], values=transported_passenger_df['Transported'], aggfunc='sum').fillna(0)
transported_container_od_matrix = pd.crosstab(transported_container_df['Origin'], transported_container_df['Destination'], values=transported_container_df['Transported'], aggfunc='sum').fillna(0)

# Sikr at alle OD-matricer indeholder alle byer på begge akser
for od_matrix_name, od_matrix in {
    "waiting_passenger_od_matrix": waiting_passenger_od_matrix,
    "waiting_container_od_matrix": waiting_container_od_matrix,
    "late_passenger_od_matrix": late_passenger_od_matrix,
    "late_container_od_matrix": late_container_od_matrix,
    "transported_passenger_od_matrix": transported_passenger_od_matrix,
    "transported_container_od_matrix": transported_container_od_matrix,
}.items():
    # Reindex for at sikre, at alle byer er til stede på både rækker og kolonner
    updated_matrix = od_matrix.reindex(index=cities, columns=cities, fill_value=0)
    globals()[od_matrix_name] = updated_matrix
    print(f"{od_matrix_name}: index = {updated_matrix.index}, columns = {updated_matrix.columns}")  # Debugging

# Debugging af late_passenger_data
if not late_passenger_data:
    print("late_passenger_data is empty!")
else:
    print("late_passenger_data has entries.")
    print(late_passenger_data[:5])  # Print de første 5 rækker for at inspicere

# Debugging af late_passenger_df
print("Late Passenger DataFrame:")
print(late_passenger_df.head())

# Hvis 'Origin' mangler, tilføj den som fallback
if "Origin" not in late_passenger_df.columns:
    print("Adding missing 'Origin' column.")
    late_passenger_df["Origin"] = np.nan
if "Destination" not in late_passenger_df.columns:
    print("Adding missing 'Destination' column.")
    late_passenger_df["Destination"] = np.nan
if "Late" not in late_passenger_df.columns:
    print("Adding missing 'Late' column.")
    late_passenger_df["Late"] = 0


# Skriv alle resultater til en Excel-fil
output_file_path = "/Users/kristofferkim/Desktop/Simulering/SimulationResultsRETTELSEAFTOURISFEJLNYNYNYNYNYNY1.xlsx"
with pd.ExcelWriter(output_file_path) as writer:
    waiting_passenger_od_matrix.to_excel(writer, sheet_name='Waiting_Passengers_OD')
    waiting_container_od_matrix.to_excel(writer, sheet_name='Waiting_Containers_OD')
    late_passenger_od_matrix.to_excel(writer, sheet_name='Late_Passengers_OD')
    late_container_od_matrix.to_excel(writer, sheet_name='Late_Containers_OD')
    transported_passenger_od_matrix.to_excel(writer, sheet_name='Transported_Passengers_OD')
    transported_container_od_matrix.to_excel(writer, sheet_name='Transpor
                                             

# Loop simuleringen 10.000 gange
all_results = []
for i in range(10000):
    print(f"Running simulation iteration {i+1}/10000...")
    result = run_simulation()
    all_results.append(result)

# Beregn gennemsnit og ekstreme værdier
aggregated_results = pd.concat(all_results)
average_results = aggregated_results.mean()
max_results = aggregated_results.max()
min_results = aggregated_results.min()

# Udskriv resultater
print("\nAverage Results:")
print(average_results)

print("\nMaximum Results (Extremes):")
print(max_results)

print("\nMinimum Results (Extremes):")
print(min_results)

print("Simulations completed.")
