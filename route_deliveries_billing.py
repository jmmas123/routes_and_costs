import re
import googlemaps
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import time
from datetime import datetime
import socket

# Initialize Google Maps API client
gmaps = googlemaps.Client(key=os.getenv('GMAPS_API_KEY'))


def parse_date(date_str):
    for fmt in ('%d-%m-%Y', '%d-%m-%y', '%d/%m/%Y', '%d/%m/%y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError("Invalid date format. Please enter dates in dd/mm/yy or dd-mm-yy format.")

def get_clean_hostname():
    hostname = socket.gethostname()
    if hostname.endswith('.local'):
        hostname = hostname.replace('.local', '')
    return hostname


def get_base_output_path():
    if os.name == 'nt':
        return r'C:\Users\josemaria\Downloads'
    else:
        hostname = get_clean_hostname()
        if hostname == 'JM-MBP':
            return '/Users/j.m./Downloads'
        elif hostname == 'JM-MS':
            return '/Users/jm/Downloads'
        return None

def load_data():
    # Define the paths to your data files
    def get_base_path(file_type):
        if os.name == 'nt':  # Windows
            if file_type == 'overtime':
                return r'\\10.5.5.11\mobu\supervisores\HE\VARIOS\Horas'
            elif file_type == 'routing':
                return r'\\10.5.5.11\mobu\supervisores\HE\VARIOS\rutas'
                # return r'C:\JM\GM\MOBU - OPL\Rutas'
            elif file_type == 'workforce':
                return r'C:\JM\GM\MOBU - OPL\Planilla'
        else:  # MacOS
            if file_type == 'overtime':
                return (r'/Users/jm/Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - '
                        r'OPL/HE/VARIOS/HORAS')
            if file_type == 'routing':
                return '/Users/jm/Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/HE/VARIOS/rutas'
            elif file_type == 'workforce':
                return '/Users/jm/Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/Planilla'

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
    # df_control = pd.read_excel(income_overtime_client_path, sheet_name='Costeo')
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
    precios_df['Fecha'] = pd.to_datetime(precios_df['Fecha'])
    date = pd.to_datetime(date)

    precios_before_date = precios_df[precios_df['Fecha'] <= date]

    if precios_before_date.empty:
        return None

    latest_date = precios_before_date['Fecha'].max()

    latest_prices = precios_before_date[(precios_before_date['Fecha'] == latest_date) & (precios_before_date['Zona'] == 'Central')]

    if latest_prices.empty:
        return None

    diesel_price_per_gallon = latest_prices['Diesel'].iloc[0]

    return diesel_price_per_gallon

def process_control_df(df_control, df_salary, df_camiones, df_precios):
    routes = []

    # Create a copy of the DataFrame to avoid modifying the original
    df = df_control.copy()

    # Ensure 'Fecha' is in datetime format
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Create a 'Ruta' column by converting 'Ruta (si fue agregado a una ruta)' to string
    df['Ruta'] = df['Ruta (si fue agregado a una ruta)'].astype(str)

    # Create a unique route identifier by combining 'Ruta' and 'Fecha'
    # For routes without a number ('nan'), create a unique identifier using 'Ruta Especial' and the date
    def generate_route_id(row):
        if row['Ruta'] == 'nan' or pd.isna(row['Ruta']):
            return f"Ruta Especial - {row['Fecha'].date()}"
        else:
            return f"Ruta {row['Ruta']} - {row['Fecha'].date()}"

    df['Route_ID'] = df.apply(generate_route_id, axis=1)

    # Group by 'Route_ID' to process each route individually
    grouped = df.groupby('Route_ID')

    for route_id, group in grouped:
        # Sort the group by 'Orden' to maintain the delivery sequence
        group = group.sort_values('Orden')

        # Get route-level information
        route_name = route_id
        route_date = group['Fecha'].iloc[0]
        empresa = group['Empresa'].iloc[0]
        placas = group['Placa Vehiculo'].dropna().unique()
        placa_vehiculo = placas[0] if len(placas) > 0 else None

        # Get vehicle and driver information
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
                driver_cargo = 'MOTORISTA'
                efficiency_km_per_gallon = None  # Default or None

        # Get driver wage per hour from Salaries df
        driver_salary_info = df_salary[df_salary['Cargo'] == driver_cargo]
        if not driver_salary_info.empty:
            try:
                driver_wage_per_hour = float(driver_salary_info['Salario/Hora'].iloc[0])
            except Exception as e:
                print(f"Error retrieving 'Salario/Hora' for cargo '{driver_cargo}': {e}")
                driver_wage_per_hour = 2.5  # Default value
        else:
            print(f"No salary information found for cargo '{driver_cargo}'. Using default value 2.5.")
            driver_wage_per_hour = 2.5  # Default value

        # Determine auxiliary personnel wage per hour
        aux_cargo = 'DESPACHADOR'  # Assuming 'DESPACHADOR' as the role for auxiliary personnel
        aux_salary_info = df_salary[df_salary['Cargo'] == aux_cargo]
        if not aux_salary_info.empty:
            try:
                aux_personnel_wage_per_hour = float(aux_salary_info['Salario/Hora'].iloc[0])
            except Exception as e:
                print(f"Error retrieving 'Salario/Hora' for cargo '{aux_cargo}': {e}")
                aux_personnel_wage_per_hour = 2.0  # Default value
        else:
            print(f"No salary information found for cargo '{aux_cargo}'. Using default value 2.0.")
            aux_personnel_wage_per_hour = 2.0  # Default value

        # Get the number of auxiliary personnel (take max value or default)
        num_aux_personnel = group['Num Auxiliares'].max()
        if pd.isna(num_aux_personnel):
            num_aux_personnel = 2  # Default value

        # Get the fuel price for the date
        fuel_price_per_gallon = get_fuel_price_on_date(df_precios, route_date)
        if fuel_price_per_gallon is None:
            print(f"No fuel price found for date {route_date}. Using default value $4.00 per gallon.")
            fuel_price_per_gallon = 4.00  # Default value

        # Calculate gas cost per km
        if efficiency_km_per_gallon is not None:
            gas_cost_per_km = fuel_price_per_gallon / efficiency_km_per_gallon
        else:
            print(f"Efficiency per gallon is not available for route '{route_id}'. Using default gas cost per km.")
            gas_cost_per_km = 0.30  # Default value

        # Process each delivery point in the route
        points = []
        unloading_times = []
        total_unloading_time = 0

        for idx, row in group.iterrows():
            direccion = row['Direccion']
            coordenada = row['Coordenada']
            orden = row['Orden']

            # Parse coordinates
            if isinstance(coordenada, str):
                try:
                    lat_str, lon_str = coordenada.strip().split(',')
                    lat = float(lat_str)
                    lon = float(lon_str)
                except Exception as e:
                    print(f"Error parsing coordenada '{coordenada}' at index {idx}: {e}")
                    lat = None
                    lon = None
            else:
                print(f"Invalid coordenada at index {idx}: {coordenada}")
                lat = None
                lon = None

            # Calculate unloading time from 'Hora Inicio' and 'Hora Fin'
            hora_inicio = row['Hora Inicio']
            hora_fin = row['Hora Fin']
            if pd.notna(hora_inicio) and pd.notna(hora_fin):
                try:
                    time_format = '%H:%M:%S'
                    t_inicio = pd.to_datetime(hora_inicio, format=time_format)
                    t_fin = pd.to_datetime(hora_fin, format=time_format)
                    unloading_time = (t_fin - t_inicio).total_seconds() / 3600  # in hours
                except Exception as e:
                    print(f"Error parsing times at index {idx}: {e}")
                    unloading_time = 1.0  # Default unloading time per store
            else:
                unloading_time = 1.0  # Default unloading time per store

            unloading_times.append(unloading_time)
            total_unloading_time += unloading_time

            # Store point data
            points.append({
                "direccion": direccion,
                "lat": lat,
                "lon": lon,
                "unloading_time": unloading_time,
                "order": orden
            })

        # Calculate average unloading time per store
        if unloading_times:
            avg_unloading_time_per_store = total_unloading_time / len(unloading_times)
        else:
            avg_unloading_time_per_store = 1.0  # Default value

        # Build the route dictionary
        route = {
            "name": route_name,
            "Empresa": empresa,
            "unloading_time_h_per_store": avg_unloading_time_per_store,
            "driver_wage_per_hour": driver_wage_per_hour,
            "aux_personnel_wage_per_hour": aux_personnel_wage_per_hour,
            "num_aux_personnel": int(num_aux_personnel),
            "gas_cost_per_km": gas_cost_per_km,
            "points": points
        }

        routes.append(route)

    # Flatten each route's points into columns
    routes_flattened = []
    for route in routes:
        flattened_route = {k: v for k, v in route.items() if k != 'points'}  # keep all keys except 'points'
        # Sort points by order
        route_points = sorted(route['points'], key=lambda x: x['order'])
        for point in route_points:
            order = int(point['order'])
            direccion = point['direccion']
            lat = point['lat']
            lon = point['lon']
            unloading_time = point['unloading_time']
            # Build column names
            base_name = f"Order_{order}_{direccion}"
            flattened_route[f"{base_name}_lat"] = lat
            flattened_route[f"{base_name}_lon"] = lon
            flattened_route[f"{base_name}_unloading_time"] = unloading_time
            flattened_route[f"{base_name}_order"] = order
        routes_flattened.append(flattened_route)

    # Convert to DataFrame and return
    routes_df = pd.DataFrame(routes_flattened)

    print("Route data:\n", routes_df)

    return routes_df

def calculate_distances_and_times(routes_df, gmaps):
    # Initialize a list to store per-leg details
    legs_details = []

    for _, row in routes_df.iterrows():
        route_name = row['name']
        cliente_name = row['Empresa']

        # Extract ordered delivery points from the DataFrame row
        # We need to find columns that match 'Order_<n>_<direccion>_lat' and '_lon'
        # We'll loop over the columns and extract the points in order

        # Find all columns that contain '_lat' and extract their base names
        lat_cols = [col for col in row.index if '_lat' in col and not pd.isna(row[col])]
        # Extract the order numbers and addresses from the column names
        points_info = []
        for col in lat_cols:
            # Example column name: 'Order_1_Deposito San Vicente_lat'
            base_name = col.replace('_lat', '')
            parts = base_name.split('_', 2)  # Splits into ['Order', '1', 'Direccion']
            if len(parts) >= 3:
                order_num = int(parts[1])
                direccion = parts[2]
            else:
                continue  # Skip if we can't parse the column name properly

            lat = row[col]
            lon_col = col.replace('_lat', '_lon')
            lon = row[lon_col]
            if pd.isna(lon):
                continue  # Skip if longitude is missing

            points_info.append({
                'order': order_num,
                'direccion': direccion,
                'lat': lat,
                'lon': lon
            })

        # Sort points by order
        points_info = sorted(points_info, key=lambda x: x['order'])

        # Build the list of waypoints (coordinates)
        waypoints = [(p['lat'], p['lon']) for p in points_info]

        # Include the starting point if necessary (e.g., PLISA)
        # For this example, let's assume PLISA coordinates are provided
        plisa_coords = (13.814771381058584, -89.40960526517033)
        waypoints = [plisa_coords] + waypoints  # Start from PLISA

        # Now, calculate distances and times between consecutive points
        for i in range(len(waypoints) - 1):
            origin = waypoints[i]
            destination = waypoints[i + 1]

            origin_str = f"{origin[0]},{origin[1]}"
            destination_str = f"{destination[0]},{destination[1]}"

            try:
                # Google Maps API request for distance and duration
                result = gmaps.distance_matrix(
                    origins=[origin_str],
                    destinations=[destination_str],
                    mode="driving",
                    units="metric",
                    region="sv"  # Assuming El Salvador
                )

                element = result["rows"][0]["elements"][0]
                if element['status'] != 'OK':
                    print(f"Error retrieving distance/duration for {origin_str} to {destination_str}: {element['status']}")
                    continue

                distance = element["distance"]["value"]  # meters
                duration = element["duration"]["value"]  # seconds

                # Collect leg details
                leg_detail = {
                    'route_name': route_name,
                    'client_name': cliente_name,
                    'leg_index': i + 1,  # Start from 1
                    'origin_lat': origin[0],
                    'origin_lon': origin[1],
                    'destination_lat': destination[0],
                    'destination_lon': destination[1],
                    'distance_km': distance / 1000,  # Convert meters to km
                    'duration_hours': duration / 3600,  # Convert seconds to hours
                }

                # If we have delivery point info, add it
                if i < len(points_info):
                    dest_point = points_info[i]
                    leg_detail['destination_order'] = dest_point['order']
                    leg_detail['destination_direccion'] = dest_point['direccion']
                else:
                    # For the return leg, there may be no destination point info
                    leg_detail['destination_order'] = None
                    leg_detail['destination_direccion'] = 'PLISA'

                legs_details.append(leg_detail)

            except Exception as e:
                print(f"Exception retrieving distance/duration for {origin_str} to {destination_str}: {e}")
                continue

        # Optionally, add the return leg back to PLISA
        # From last point back to PLISA
        origin = waypoints[-1]
        destination = plisa_coords

        origin_str = f"{origin[0]},{origin[1]}"
        destination_str = f"{destination[0]},{destination[1]}"

        try:
            result = gmaps.distance_matrix(
                origins=[origin_str],
                destinations=[destination_str],
                mode="driving",
                units="metric",
                region="sv"
            )

            element = result["rows"][0]["elements"][0]
            if element['status'] != 'OK':
                print(f"Error retrieving distance/duration for return trip {origin_str} to {destination_str}: {element['status']}")
                continue

            distance = element["distance"]["value"]  # meters
            duration = element["duration"]["value"]  # seconds

            # Collect return leg details
            leg_detail = {
                'route_name': route_name,
                'client_name': cliente_name,
                'leg_index': len(waypoints),  # Return leg index
                'origin_lat': origin[0],
                'origin_lon': origin[1],
                'destination_lat': destination[0],
                'destination_lon': destination[1],
                'distance_km': distance / 1000,
                'duration_hours': duration / 3600,
                'destination_order': None,
                'destination_direccion': 'PLISA'
            }

            legs_details.append(leg_detail)

        except Exception as e:
            print(f"Exception retrieving distance/duration for return trip {origin_str} to {destination_str}: {e}")
            continue

    # Convert legs_details to DataFrame
    legs_df = pd.DataFrame(legs_details)

    # Optionally, merge or join this legs_df with the original routes_df if needed

    print("Routing info with driving details:\n", legs_df)

    return legs_df


def filter_data_by_date(df, date_column, start_date, end_date):
    """
    Filter a DataFrame based on a date range.

    Args:
        df (pd.DataFrame): DataFrame to be filtered.
        date_column (str): Column name containing date values.
        start_date (pd.Timestamp): Start date of the range.
        end_date (pd.Timestamp): End date of the range.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df[date_column] = pd.to_datetime(df[date_column])  # Ensure date column is in datetime format
    return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]


def main():
    pd.set_option(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.expand_frame_repr", False
    )

    start_date_str = input("Enter the start date of analysis (dd/mm/yy or dd-mm-yy): ")
    end_date_str = input("Enter the end date of analysis (dd/mm/yy or dd-mm-yy): ")

    # Convert to datetime with error handling
    try:
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)

        if start_date > end_date:
            print("Error: Start date must be before or equal to end date.")
            return

        # Convert to pandas Timestamp for compatibility with DataFrame date columns
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
    except ValueError as e:
        print(e)
        return

    print("Main: Loading data...")

    # Load data
    df_delivery_overtime, df_salary, df_control, df_rutas, df_camiones, df_precios = load_data()

    print("Filtering data by date range...")

    # Apply date filtering
    df_control = filter_data_by_date(df_control, 'Fecha', start_date, end_date)
    df_delivery_overtime = filter_data_by_date(df_delivery_overtime, 'Fecha', start_date, end_date)

    # Clean time columns in df_control
    df_control = clean_time_format(df_control, 'Hora Inicio')
    df_control = clean_time_format(df_control, 'Hora Fin')

    # Process Control df to create routes
    routes_df = process_control_df(df_control, df_salary, df_camiones, df_precios)

    # Calculate distances and times for filtered data
    routes_calc = calculate_distances_and_times(routes_df, gmaps)

    # Optional: Save results
    output_path = os.path.join(get_base_output_path(), 'costos_de_ruta(26-25-02-25).csv')
    routes_calc.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()