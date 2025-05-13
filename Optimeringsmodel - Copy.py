import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import math

import re

# Load the Excel file with all city-day sheets
file_path = "Demand Monday1.xlsx"
sheets = pd.ExcelFile(file_path).sheet_names

# Parameters
S_g = 2 #capacity per goods wagon
S_p = 110 #capacity per passenger wagon
R_g = 1242 #revenue per goods unit 
R_p = 129 #revenue per passenger unit
C_w = 500 #cost of adding a wagon
P_bg = 300 #penalty in revenue for backlogging goods
P_bp = 129 #penalty for backlogging passengers

# Results storage
all_results = []

# Process each sheet separately
for sheet in sheets:
    data = pd.read_excel(file_path, sheet_name=sheet)
    required_columns = ['Goods Demand', 'Passenger Demand']
    if not all(col in data.columns for col in required_columns):
        print(f"Missing columns in sheet '{sheet}'. Skipping.")
        continue
    
    day_mapping = ['Mandag', 'Tirsdag', 'Onsdag', 'Torsdag', 'Fredag', 'Lørdag', 'Søndag']
    city = re.sub('|'.join(day_mapping), '', sheet).strip()
    day = next((d for d in day_mapping if d in sheet), 'Unknown')

    data.fillna(0, inplace=True)
    t_max = len(data)
    D_g = data["Goods Demand"].tolist()
    D_p = data["Passenger Demand"].tolist()

    # Definer Problemet
    problem = LpProblem(f"Maximize_Profit_{city}_{day}", LpMaximize)

    # Beslutningsvariable
    x_g = LpVariable.dicts("x_g", range(t_max), lowBound=0, cat="Integer")
    x_p = LpVariable.dicts("x_p", range(t_max), lowBound=0, cat="Integer")
    U_g = LpVariable.dicts("U_g", range(t_max), lowBound=0, cat="Continuous")
    U_p = LpVariable.dicts("U_p", range(t_max), lowBound=0, cat="Continuous")
    met_goods = LpVariable.dicts("met_goods", range(t_max), lowBound=0)
    met_passengers = LpVariable.dicts("met_passengers", range(t_max), lowBound=0)

    # Objektivfunktin opstilles
    revenue = lpSum([R_g * met_goods[t] + R_p * met_passengers[t] for t in range(t_max)])
    cost = lpSum([C_w * (x_g[t] + x_p[t]) for t in range(t_max)])
    penalty = lpSum([P_bg * U_g[t] + P_bp * U_p[t] for t in range(t_max)])
    profit = revenue - cost - penalty
    problem += profit

    # Constraints
    for t in range(t_max):
        prev_U_g = U_g[t - 1] if t > 0 else 0
        prev_U_p = U_p[t - 1] if t > 0 else 0
        current_goods_demand = D_g[t] + prev_U_g
        current_passenger_demand = D_p[t] + prev_U_p

        # Ensure met goods and passengers do not exceed wagon capacity
        problem += met_goods[t] <= x_g[t] * S_g
        problem += met_passengers[t] <= x_p[t] * S_p

        # Ensure met goods and passengers do not exceed current demand
        problem += met_goods[t] <= current_goods_demand
        problem += met_passengers[t] <= current_passenger_demand

        # Update backlog
        problem += U_g[t] == current_goods_demand - met_goods[t]
        problem += U_p[t] == current_passenger_demand - met_passengers[t]

        # Limit total number of wagons
        problem += x_g[t] + x_p[t] <= 19

    # Solve with time limit and print status
    status = problem.solve(PULP_CBC_CMD(timeLimit=120)) 
    if status != 1:
        print(f"Warning: Optimization for sheet '{sheet}' did not converge or was not optimal.")

    # Store results and calculate actual profit for each hour
    for t in range(t_max):
        actual_met_goods = math.ceil(min(D_g[t], met_goods[t].varValue)) if met_goods[t].varValue is not None else 0
        actual_met_passengers = math.ceil(min(D_p[t], met_passengers[t].varValue)) if met_passengers[t].varValue is not None else 0
        actual_revenue = R_g * actual_met_goods + R_p * actual_met_passengers
        actual_cost = C_w * (math.ceil(x_g[t].varValue) + math.ceil(x_p[t].varValue)) if x_g[t].varValue is not None and x_p[t].varValue is not None else 0
        actual_penalty = P_bg * math.ceil(U_g[t].varValue) + P_bp * math.ceil(U_p[t].varValue) if U_g[t].varValue is not None and U_p[t].varValue is not None else 0
        actual_profit = actual_revenue - actual_cost - actual_penalty
        extra_capacity = 19 - (math.ceil(x_g[t].varValue) + math.ceil(x_p[t].varValue)) if x_g[t].varValue is not None and x_p[t].varValue is not None else 0

        all_results.append({
            "City": city,
            "Day": day,
            "Hour": t,
            "Goods Wagons": math.ceil(x_g[t].varValue) if x_g[t].varValue is not None else 0,
            "Passenger Wagons": math.ceil(x_p[t].varValue) if x_p[t].varValue is not None else 0,
            "Goods Backlog": math.ceil(U_g[t].varValue) if U_g[t].varValue is not None else 0,
            "Passenger Backlog": math.ceil(U_p[t].varValue) if U_p[t].varValue is not None else 0,
            "Goods Units Transported": actual_met_goods,
            "Passengers Transported": actual_met_passengers,
            "Profit": actual_profit,
            "Extra Capacity": extra_capacity
        })

# Export results to HTML (web)
html_content = """
<html>
<head>
    <title>Optimization Results</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h1>Optimization Results</h1>
    <table>
        <tr>
            <th>City</th>
            <th>Day</th>
            <th>Hour</th>
            <th>Goods Wagons</th>
            <th>Passenger Wagons</th>
            <th>Goods Backlog</th>
            <th>Passenger Backlog</th>
            <th>Met Goods</th>
            <th>Met Passengers</th>
            <th>Profit</th>
            <th>Extra Capacity</th>
        </tr>
"""
for result in all_results:
    html_content += f"""
        <tr>
            <td>{result['City']}</td>
            <td>{result['Day']}</td>
            <td>{result['Hour']}</td>
            <td>{result['Goods Wagons']}</td>
            <td>{result['Passenger Wagons']}</td>
            <td>{result['Goods Backlog']}</td>
            <td>{result['Passenger Backlog']}</td>
            <td>{result['Goods Units Transported']}</td>
            <td>{result['Passengers Transported']}</td>
            <td>{result['Profit']}</td>
            <td>{result['Extra Capacity']}</td>
        </tr>
    """
html_content += """
    <tr><td colspan="11" style="text-align:right; font-weight:bold;">Total Profit: {sum(result['Profit'] for result in all_results)}</td></tr>
</table>
</body>
</html>
"""

with open("Optimization_Results.html", "w") as f:
    f.write(html_content)

# Export results to Excel (fil)
#Disse kommandoer sørger for at resultaterne bliver printet både i HTML og Excelformat
results_df = pd.DataFrame(all_results)
results_df.to_excel("Optimization_Results.xlsx", index=False)

print("Optimization completed. Results have been saved to 'Optimization_Results.xlsx' and 'Optimization_Results.html'.")
