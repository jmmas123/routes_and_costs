import re
import googlemaps
import numpy as np
import pandas as pd
import os
from datetime import time

# Initialize Google Maps API client
gmaps = googlemaps.Client(key='***REMOVED***')

def get_base_output_path():
    if os.name == 'nt':  # Windows
        obase_path = r'C:\Users\josemaria\Downloads'
    else:  # MacOS (or others)
        obase_path = r'/Users/j.m./Downloads'
    return obase_path

def load_data():
    # Define the paths to your data files
    def get_base_path(file_type):
        if os.name == 'nt':  # Windows
            if file_type == 'overtime':
                return r'\\192.168.10.18\Bodega General\HE\VARIOS\Horas'
            elif file_type == 'routing':
                return r'\\192.168.10.18\Bodega General\HE\VARIOS\rutas'
                # return r'C:\JM\GM\MOBU - OPL\Rutas'
            elif file_type == 'workforce':
                return r'C:\JM\GM\MOBU - OPL\Planilla'
        else:  # MacOS
            if file_type == 'overtime':
                return (r'/Users/j.m./Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - '
                        r'OPL/Horas extra')
            if file_type == 'routing':
                return '/Users/j.m./Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/HE/VARIOS/rutas'
            elif file_type == 'workforce':
                return '/Users/j.m./Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/Planilla'

    # Get base paths
    overtime_t_base_path = get_base_path('routing')
    overtime_base_path = get_base_path('overtime')
    workforce_base_path = get_base_path('workforce')

    # Construct file paths
    overtime_file_path = os.path.join(overtime_base_path, 'Horas extra NF.xlsx')
    workforce_and_salaries_path = os.path.join(workforce_base_path, 'Reporte de personal MORIBUS.xlsx')
    income_overtime_client_path = os.path.join(overtime_t_base_path, 'control de rutas y fletes.xlsx')

    # Read the second table from the second sheet
    df_delivery_overtime = pd.read_excel(overtime_file_path, sheet_name='Horas en ruta', header=0, dtype={'Codigo': str})
    # Read document containing salaries and workforce
    df_salary = pd.read_excel(workforce_and_salaries_path, sheet_name='Hora regular', header=0)
    # Read document containing Routing information
    df_control = pd.read_excel(income_overtime_client_path, sheet_name='Control de Rutas y Fletes')
    # Read document containing Route delivery points
    df_rutas = pd.read_excel(income_overtime_client_path, sheet_name='Rutas')
    # Read document containing Truck information
    df_camiones = pd.read_excel(income_overtime_client_path, sheet_name='Camiones')
    # Read document containing gas prices
    df_precios = pd.read_excel(income_overtime_client_path, sheet_name='Precios Gasolina')

    print("Overtime df:\n", df_delivery_overtime)
    print("Salaries df:\n", df_salary)
    print("Control df:\n", df_control)
    print("Rutas df:\n", df_rutas)
    print("Camiones df:\n", df_camiones)
    print("Precios df:\n", df_precios)

    return df_delivery_overtime, df_salary, df_control, df_rutas, df_camiones, df_precios


def clean_time_format(df, time_column):
    """
    Normalize time format in a specified column of the DataFrame by removing extra periods in "AM" or "PM" suffixes
    and replacing periods between digits with colons for correct time parsing.
    """

    def normalize_time(time_str):
        # If already in datetime.time format, return it as is
        if isinstance(time_str, time):
            return time_str
        # Handle NaN values
        if pd.isna(time_str):
            return None
        # If the value is a string, clean it up
        if isinstance(time_str, str):
            # Replace periods between digits with a colon
            cleaned_time = re.sub(r'(\d+)\.(\d+)', r'\1:\2', time_str)
            # Remove any periods in "AM" or "PM" suffix
            cleaned_time = cleaned_time.replace(".", "").upper()  # Replace periods and ensure uppercase "AM"/"PM"
            try:
                # Convert to datetime format
                return pd.to_datetime(cleaned_time, format='%I:%M %p').time()
            except ValueError:
                print(f"Unable to parse time: {time_str}")
                return None  # Return None if the time format is incorrect
        else:
            # For unexpected data types, log and return None
            print(f"Unexpected value type in time column: {time_str} (type: {type(time_str)})")
            return None

    # Apply normalization function to the specified time column
    df[time_column] = df[time_column].apply(normalize_time)
    return df


def get_fuel_price_on_date(precios_df, date):
    # Convert date columns to datetime if not already
    precios_df['Fecha'] = pd.to_datetime(precios_df['Fecha'])
    date = pd.to_datetime(date)

    # Filter prices up to the given date
    precios_before_date = precios_df[precios_df['Fecha'] <= date]

    if precios_before_date.empty:
        return None  # No prices available before the date

    # Get the latest available date before or equal to the route date
    latest_date = precios_before_date['Fecha'].max()

    # Filter for the latest date and 'Central' zone
    latest_prices = precios_before_date[
        (precios_before_date['Fecha'] == latest_date) & (precios_before_date['Zona'] == 'Central')]

    if latest_prices.empty:
        return None  # No prices available for 'Central' zone on that date

    diesel_price_per_gallon = latest_prices['Diesel'].iloc[0]

    return diesel_price_per_gallon

def process_control_df(df_control, df_salary, df_camiones, df_precios):
    routes = []

    # Create a copy of the DataFrame to avoid modifying the original
    df = df_control.copy()

    # Create a 'Ruta' column by converting 'Ruta (si fue agregado a una ruta)' to string
    df['Ruta'] = df['Ruta (si fue agregado a una ruta)'].astype(str)

    # Ensure 'Fecha' is in datetime format
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Create a unique route identifier by combining 'Ruta' and 'Fecha'
    # For routes without a number ('nan'), create a unique identifier using the date and index
    def generate_route_id(row):
        if row['Ruta'] == 'nan' or pd.isna(row['Ruta']):
            return f"Ruta Especial - {row['Fecha'].date()}"
        else:
            return f"Ruta {row['Ruta']} - {row['Fecha'].date()}"

    df['Route_ID'] = df.apply(generate_route_id, axis=1)

    # Now group by 'Route_ID'
    grouped = df.groupby('Route_ID')

    for route_id, group in grouped:
        # For each group (route), create a route dict
        route_name = f"{route_id}"
        empresa_name = group['Empresa'].iloc[0] if 'Empresa' in group.columns else "Unknown Empresa"
        points = {
            "PLISA": (13.814771381058584, -89.40960526517033)
        }

        # For each delivery point in the route, extract the 'Direccion' and 'Coordenada'
        for idx, row in group.iterrows():
            direccion = row['Direccion']
            coordenada = row['Coordenada']
            if isinstance(coordenada, str):
                try:
                    lat_str, lon_str = coordenada.strip().split(',')
                    lat = float(lat_str)
                    lon = float(lon_str)
                    points[direccion] = (lat, lon)
                except Exception as e:
                    print(f"Error parsing coordenada '{coordenada}' at index {idx}: {e}")
            else:
                print(f"Invalid coordenada at index {idx}: {coordenada}")

        # Calculate unloading time per store from 'Hora Inicio' and 'Hora Fin'
        unloading_times = []
        for idx, row in group.iterrows():
            hora_inicio = row['Hora Inicio']
            hora_fin = row['Hora Fin']
            if pd.notna(hora_inicio) and pd.notna(hora_fin):
                try:
                    time_format = '%H:%M:%S'
                    t_inicio = pd.to_datetime(hora_inicio, format=time_format)
                    t_fin = pd.to_datetime(hora_fin, format=time_format)
                    unloading_time = (t_fin - t_inicio).total_seconds() / 3600  # in hours
                    unloading_times.append(unloading_time)
                except Exception as e:
                    print(f"Error parsing times at index {idx}: {e}")
            else:
                # If times are missing, assume default unloading time
                unloading_times.append(1.0)  # Default unloading time per store

        # Calculate average unloading time per store
        if unloading_times:
            # Calculate total unloading time for the route
            unloading_time_h_per_store = sum(unloading_times) if unloading_times else len(group)  # Defaults to 1h/store
        else:
            unloading_time_h_per_store = 1.0  # Default value

        # Determine driver wage per hour
        placas = group['Placa Vehiculo'].unique()
        placa_vehiculo = placas[0] if pd.notna(placas[0]) else None

        if placa_vehiculo is None:
            driver_cargo = 'MOTORISTA'
            efficiency_km_per_gallon = None  # No truck specified
        else:
            # Match the placa with Camiones df to get capacity and efficiency
            camion_info = df_camiones[df_camiones['Placa'] == placa_vehiculo]
            if not camion_info.empty:
                capacidad_ton = camion_info['Capacidad (Ton)'].iloc[0]
                efficiency_km_per_gallon = camion_info['Eficiencia (km/gal)'].iloc[0]
                if capacidad_ton > 10:
                    driver_cargo = 'MOTORISTA LICENCIA PESADA'
                else:
                    driver_cargo = 'MOTORISTA'
            else:
                print(f"Error: No truck information found for placa '{placa_vehiculo}'.")
                # You can choose to raise an error or assign default values
                # For this example, we'll raise an exception
                raise ValueError(f"No truck information found for placa '{placa_vehiculo}' in Camiones df.")

        driver_salary_info = df_salary[df_salary['Cargo'] == driver_cargo]

        if not driver_salary_info.empty:
            try:
                driver_wage_per_hour = float(driver_salary_info['Salario/Hora'].iloc[0])
            except (IndexError, KeyError, ValueError) as e:
                print(f"Error retrieving 'Salario/Hora' for cargo '{driver_cargo}': {e}")
                print(f"Using default driver wage per hour: 2.5")
                driver_wage_per_hour = 2.5  # Default value
        else:
            print(f"No salary information found for cargo '{driver_cargo}'. Using default value $2.5.")
            driver_wage_per_hour = 2.5  # Default value

        # Determine auxiliary personnel wage per hour
        aux_cargo = 'DESPACHADOR'  # Assuming 'DESPACHADOR' as the role for auxiliary personnel
        aux_salary_info = df_salary[df_salary['Cargo'] == aux_cargo]
        if not aux_salary_info.empty:
            aux_personnel_wage_per_hour = aux_salary_info['Salario/Hora'].iloc[0]
        else:
            print(f"No salary information found for cargo '{aux_cargo}'. Using default value $2.0.")
            aux_personnel_wage_per_hour = 2.0  # Default value

        # Get the number of auxiliary personnel (take max value or default)
        num_aux_personnel = group['Num Auxiliares'].max()
        if pd.isna(num_aux_personnel):
            num_aux_personnel = 2  # Default value

        # Get the date of the route
        route_date = group['Fecha'].iloc[0]

        # Find the fuel price for the date and zone
        fuel_price_per_gallon = get_fuel_price_on_date(df_precios, route_date)

        if fuel_price_per_gallon is None:
            print(f"No fuel price found for date {route_date}. Using default value $4.00 per gallon.")
            fuel_price_per_gallon = 4.00  # Default value

        # Calculate gas cost per km
        if efficiency_km_per_gallon is not None:
            gas_cost_per_km = fuel_price_per_gallon / efficiency_km_per_gallon
        else:
            print(
                f"Efficiency per gallon is not available for route '{route_name}'. Cannot calculate gas cost per km.")
            gas_cost_per_km = 0.30  # default value

        route = {
            "name": route_name,
            "Empresa": empresa_name,
            "points": points,
            "unloading_time_h_per_store": unloading_time_h_per_store,
            "driver_wage_per_hour": driver_wage_per_hour,
            "aux_personnel_wage_per_hour": aux_personnel_wage_per_hour,
            "num_aux_personnel": int(num_aux_personnel),
            "gas_cost_per_km": gas_cost_per_km
        }

        routes.append(route)

    # Flatten each route's points dictionary into individual entries for easier DataFrame creation
    routes_flattened = []
    for route in routes:
        flattened_route = {k: v for k, v in route.items() if k != 'points'}  # keep all keys except 'points'
        for point_name, coords in route['points'].items():
            flattened_route[f"{point_name}_lat"] = coords[0]
            flattened_route[f"{point_name}_lon"] = coords[1]
        routes_flattened.append(flattened_route)

    # Convert to DataFrame and return
    routes_df = pd.DataFrame(routes_flattened)

    print("Route data:\n", routes_df)

    return routes_df


def calculate_distances_and_times(routes_df):
    # Initialize lists for distance, duration, and delivery points
    total_distances = []
    eta_hours_list = []
    delivery_points_count = []

    for _, row in routes_df.iterrows():
        # Extract points from the DataFrame row
        points = {col.replace("_lat", "").replace("_lon", ""): (row[f"{col}"], row[f"{col.replace('_lat', '_lon')}"])
                  for col in row.index if "_lat" in col}

        # Clean out NaN values from points
        points = {name: coords for name, coords in points.items() if not pd.isna(coords[0]) and not pd.isna(coords[1])}

        # Count delivery points
        delivery_points_count.append(len(points))

        # Convert points to a sorted list of coordinates
        sorted_points = sorted(points.items())
        waypoints_str = [f"{lat},{lon}" for name, (lat, lon) in sorted_points]

        total_distance = 0
        total_duration = 0

        # Calculate distance and ETA for each consecutive pair of points
        for i in range(len(waypoints_str) - 1):
            origin = waypoints_str[i]
            destination = waypoints_str[i + 1]
            try:
                # Google Maps API request for distance and duration
                result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
                distance = result["rows"][0]["elements"][0]["distance"]["value"]  # meters
                duration = result["rows"][0]["elements"][0]["duration"]["value"]  # seconds

                total_distance += distance / 1000  # convert meters to km
                total_duration += duration / 3600  # convert seconds to hours
            except Exception as e:
                print(f"Error retrieving distance/duration for {origin} to {destination}: {e}")
                continue

        # Add the return leg to make it a round trip
        if waypoints_str:
            origin = waypoints_str[-1]
            destination = waypoints_str[0]  # return to the starting point
            try:
                result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
                return_distance = result["rows"][0]["elements"][0]["distance"]["value"]  # meters
                return_duration = result["rows"][0]["elements"][0]["duration"]["value"]  # seconds

                total_distance += return_distance / 1000  # convert meters to km
                total_duration += return_duration / 3600  # convert seconds to hours
            except Exception as e:
                print(f"Error retrieving distance/duration for return trip {origin} to {destination}: {e}")

        total_distances.append(total_distance)
        eta_hours_list.append(total_duration)

    # Add total distance, ETA, and delivery points count columns to DataFrame
    routes_df["total_km"] = total_distances
    routes_df["eta_hours"] = eta_hours_list
    routes_df["total_delivery_points"] = delivery_points_count

    return routes_df


def clean_dataframe(routes_df):
    # Drop latitude and longitude columns
    lat_lon_cols = [col for col in routes_df.columns if "_lat" in col or "_lon" in col]
    routes_df = routes_df.drop(columns=lat_lon_cols)

    # Rename columns as specified
    routes_df = routes_df.rename(columns={
        'unloading_time_h_per_store': 'Unloading time (Total)',
        'eta_hours': 'Driving time (Total)',
        'total_delivery_points': 'Total delivery points',
        'total_km': 'Total Km traveled',
        'driver_wage_per_hour': 'Driver wage',
        'aux_personnel_wage_per_hour': 'Aux personnel wage',
        'num_aux_personnel': 'Number of aux personnel',
        'gas_cost_per_km': 'Gas cost/km',
    })

    # Reorder columns as per specified structure (excluding 'date' for now)
    routes_df = routes_df[[
        'name',
        'Unloading time (Total)',
        'Driving time (Total)',
        'Total delivery points',
        'Total Km traveled',
        'Driver wage',
        'Aux personnel wage',
        'Number of aux personnel',
        'Gas cost/km',
    ]]

    # Function to separate name and date into two columns
    def split_name_and_date(value):
        # Check for standard route pattern
        match = re.match(r'Ruta (\d+)\.\d+ - (\d{4}-\d{2}-\d{2})', value)
        if match:
            route_number = f"Ruta {match.group(1)}"  # Route name without decimal
            date_str = match.group(2)
            formatted_date = pd.to_datetime(date_str).strftime('%d/%m/%Y')
            return route_number, formatted_date

        # Check for special route pattern
        special_match = re.match(r'Ruta Especial - (\d{4}-\d{2}-\d{2})', value)
        if special_match:
            route_number = "Ruta Especial"
            date_str = special_match.group(1)
            formatted_date = pd.to_datetime(date_str).strftime('%d/%m/%Y')
            return route_number, formatted_date

        # Return original name and NaN if no match
        return value, pd.NaT

    # Apply the function to split 'name' into 'name' and 'date'
    routes_df[['Name', 'Date']] = routes_df['name'].apply(lambda x: pd.Series(split_name_and_date(x)))

    # Reorder columns to include the new 'date' column
    routes_df = routes_df[[
        'Date',
        'Name',
        'Unloading time (Total)',
        'Driving time (Total)',
        'Total delivery points',
        'Total Km traveled',
        'Driver wage',
        'Aux personnel wage',
        'Number of aux personnel',
        'Gas cost/km',
    ]]

    print("Routing info:\n", routes_df)

    return routes_df


def general_cost_calculation(routes_df):
    # Calculate total gas cost based on total km and gas cost per km
    routes_df['Total gas cost'] = routes_df['Total Km traveled'] * routes_df['Gas cost/km']

    # Calculate total wage cost
    # First, calculate the combined hourly wage for driver and auxiliary personnel
    routes_df['Total wage cost'] = (
            (routes_df['Driver wage'] + (routes_df['Aux personnel wage'] * routes_df['Number of aux personnel'])) *
            (routes_df['Unloading time (Total)'] + routes_df['Driving time (Total)'])
    )

    # Calculate the total cost as sum of gas cost and wage cost
    routes_df['Total route cost'] = routes_df['Total gas cost'] + routes_df['Total wage cost']

    # Select relevant columns to display
    cost_columns = [
        'Name', 'Date', 'Total Km traveled', 'Gas cost/km', 'Total gas cost',
        'Unloading time (Total)', 'Driving time (Total)', 'Driver wage',
        'Aux personnel wage', 'Number of aux personnel', 'Total wage cost', 'Total route cost'
    ]
    cost_df = routes_df[cost_columns]

    print("Cost calculation:\n", cost_df)

    return cost_df





def main():
    pd.set_option(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.expand_frame_repr", False
    )

    print("Main: Loading data...")

    df_delivery_overtime, df_salary, df_control, df_rutas, df_camiones, df_precios = load_data()

    df_control = clean_time_format(df_control, 'Hora Inicio')
    df_control = clean_time_format(df_control, 'Hora Fin')

    # Process Control df to create routes
    routes_df = process_control_df(df_control, df_salary, df_camiones, df_precios)

    routes_calc = calculate_distances_and_times(routes_df)

    routes_calc_df = clean_dataframe(routes_calc)

    routing_cost = general_cost_calculation(routes_calc_df)

    # # Save the results to a CSV file
    # output_path = os.path.join(get_base_output_path(), 'warehouse_proportional_operations.csv')
    # df_warehouse_proportional.to_csv(output_path, index=False)
    # print(f"\nResults saved to {output_path}")
    #
    # output_path = os.path.join(get_base_output_path(), 'warehouse_grouped_client.csv')
    # grouped_warehouse.to_csv(output_path, index=False)
    # print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
