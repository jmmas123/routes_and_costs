import re
import googlemaps
import numpy as np
import pandas as pd
import os
import socket
from datetime import time, datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

def load_data(mode='billing'):
    """
    Load data files for route pricing calculation.

    Args:
        mode (str): 'billing' or 'quoting'. Determines which sheet to read from control file.

    Returns:
        tuple: DataFrames for overtime, salary, control, routes, trucks, prices, and salary_overtime
    """
    if mode not in ['billing', 'quoting']:
        raise ValueError(f"Mode must be 'billing' or 'quoting'. Got: {mode}")

    # Define the paths to your data files
    def get_base_path(file_type):
        if os.name == 'nt':  # Windows
            if file_type == 'overtime':
                return r'\\192.168.10.18\Bodega General\HE\VARIOS\Horas'
            elif file_type == 'routing':
                return r'\\192.168.10.18\Bodega General\HE\VARIOS\rutas'
            elif file_type == 'workforce':
                return r'C:\JM\GM\MOBU - OPL\Planilla'
            else:
                raise ValueError(f"Unknown file type: {file_type}")
        else:  # macOS or others
            hostname = socket.gethostname()
            print(f"Detected hostname: {hostname}")
            if hostname == 'JM-MS.local':  # Replace with Mac Studio hostname
                if file_type == 'overtime':
                    return r'/Volumes/mobu/supervisores/HE/VARIOS/Horas'
                elif file_type == 'routing':
                    return '/Volumes/mobu/supervisores/HE/VARIOS/rutas'
                elif file_type == 'workforce':
                    return r'/Users/jm/Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/Planilla'
                else:
                    raise ValueError(f"Unknown file type: {file_type}")
            elif hostname == 'JM-MBP.local':  # Replace with MacBook Pro hostname
                if file_type == 'overtime':
                    return r'/Volumes/mobu/supervisores/HE/VARIOS/Horas'
                elif file_type == 'routing':
                    return '/Volumes/mobu/supervisores/HE/VARIOS/rutas'
                elif file_type == 'workforce':
                    return r'/Users/j.m./Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/Planilla'
                else:
                    raise ValueError(f"Unknown file type: {file_type}")
            else:
                print(f"Warning: Unknown hostname {hostname}. Using fallback paths.")
                # Default fallback paths for macOS
                if file_type == 'overtime':
                    return r'/Users/default_user/HE/VARIOS/Horas'
                elif file_type == 'routing':
                    return r'/Users/default_user/HE/VARIOS/rutas'
                elif file_type == 'workforce':
                    return r'/Users/default_user/Planilla'
                else:
                    raise ValueError(f"Unknown file type: {file_type}")

    # Get base paths
    overtime_t_base_path = get_base_path('routing')
    overtime_base_path = get_base_path('overtime')
    workforce_base_path = get_base_path('workforce')

    # Construct file paths
    overtime_file_path = os.path.join(overtime_base_path, 'Horas extra NF2.xlsx')
    workforce_and_salaries_path = os.path.join(workforce_base_path, 'Reporte de personal MORIBUS.xlsx')
    income_overtime_client_path = os.path.join(overtime_t_base_path, 'control de rutas y fletes.xlsx')

    # Read the second table from the second sheet
    df_delivery_overtime = pd.read_excel(overtime_file_path, sheet_name='Horas en ruta', header=0,
                                         dtype={'Codigo': str})
    # Read document containing salaries and workforce
    df_salary = pd.read_excel(workforce_and_salaries_path, sheet_name='Hora regular', header=0)
    df_salary_overtime = pd.read_excel(workforce_and_salaries_path, sheet_name='Empleados', header=0)

    # Read document containing Routing information - sheet depends on mode
    if mode == 'billing':
        control_sheet = 'Control de Rutas y Fletes'
    else:  # quoting
        control_sheet = 'Costeo'

    print(f"Loading data in {mode} mode - reading from sheet: '{control_sheet}'")
    df_control = pd.read_excel(income_overtime_client_path, sheet_name=control_sheet)

    # Read document containing Route delivery points
    df_rutas = pd.read_excel(income_overtime_client_path, sheet_name='Rutas')
    # Read document containing Truck information
    df_camiones = pd.read_excel(income_overtime_client_path, sheet_name='Camiones')
    # Read document containing gas prices
    df_precios = pd.read_excel(income_overtime_client_path, sheet_name='Precios Gasolina')

    print("Overtime df:\n", df_delivery_overtime)
    print("Salaries df:\n", df_salary)
    print("Salaries df for overtime:\n", df_salary_overtime)
    print("Control df:\n", df_control)
    print("Rutas df:\n", df_rutas)
    print("Camiones df:\n", df_camiones)
    print("Precios df:\n", df_precios)

    return df_delivery_overtime, df_salary, df_control, df_rutas, df_camiones, df_precios, df_salary_overtime


def clean_time_format(df, time_column):
    """
    Normalize time format in a specified column of the DataFrame by removing extra periods in "AM" or "PM" suffixes
    and replacing periods between digits with colons for correct time parsing.
    Handles multiple time formats including 24-hour and 12-hour with AM/PM.
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

            # Try multiple time formats
            time_formats = [
                "%H:%M:%S",      # 24-hour with seconds: 14:30:00
                "%H:%M",         # 24-hour without seconds: 14:30
                "%I:%M:%S %p",   # 12-hour with seconds and AM/PM: 02:30:00 PM
                "%I:%M %p",      # 12-hour without seconds and AM/PM: 02:30 PM
                "%I:%M:%S%p",    # 12-hour with seconds, no space: 02:30:00PM
                "%I:%M%p",       # 12-hour without seconds, no space: 02:30PM
            ]

            for fmt in time_formats:
                try:
                    return pd.to_datetime(cleaned_time, format=fmt).time()
                except ValueError:
                    continue

            # If all formats fail, print error and return None
            print(f"Unable to parse time with any known format: '{time_str}' (cleaned: '{cleaned_time}')")
            return None
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

    diesel_price_per_gallon = latest_prices['Superior'].iloc[0]

    return diesel_price_per_gallon


def assign_special_route(df_control):
    # Assign "Ruta Especial" to rows with no specific route
    df_control['Ruta (si fue agregado a una ruta)'] = (df_control['Ruta (si fue agregado a una ruta)'].fillna
                                                      ("Especial"))
    return df_control

def process_control_df(df_control, df_salary, df_camiones, df_precios):
    route_summaries = []
    client_breakdowns = []

    df = df_control.copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    grouped = df.groupby(['Ruta (si fue agregado a una ruta)', 'Fecha'])

    for (route_id, date), group in grouped:
        route_name = f"Ruta {route_id} - {date.date()}"

        # Default starting point (PLISA)
        points = {"PLISA": (13.814771381058584, -89.40960526517033)}

        # Add all delivery points to route
        for _, row in group.iterrows():
            if isinstance(row['Coordenada'], str):
                try:
                    lat, lon = map(float, row['Coordenada'].split(','))
                    points[row['Direccion']] = (lat, lon)
                except:
                    continue

        # Common values per route
        placa = group['Placa Vehiculo'].dropna().iloc[0] if not group['Placa Vehiculo'].dropna().empty else None
        camion_info = df_camiones[df_camiones['Placa'] == placa] if placa else pd.DataFrame()
        eff_km_per_gal = camion_info['Eficiencia (km/gal)'].iloc[0] if not camion_info.empty else 10

        fuel_price = get_fuel_price_on_date(df_precios, date) or 5.0
        gas_cost_km = fuel_price / eff_km_per_gal if eff_km_per_gal else 0.5

        wage_driver = df_salary[df_salary['Cargo'] == 'MOTORISTA']['Salario/Hora'].iloc[0]
        wage_aux = df_salary[df_salary['Cargo'] == 'DESPACHADOR']['Salario/Hora'].iloc[0]
        n_aux = group['Num Auxiliares'].max() if not group['Num Auxiliares'].isna().all() else 2

        route_summaries.append({
            'route_id': route_id,
            'date': date,
            'name': route_name,
            'gas_cost_per_km': gas_cost_km,
            'driver_wage': wage_driver,
            'aux_wage': wage_aux,
            'num_aux': n_aux,
            'points': points
        })

        total_points = len(group)
        for client, client_group in group.groupby('Empresa'):
            n_points = len(client_group)
            prop = n_points / total_points if total_points else 0

            # Unloading time
            unloading = 0
            for idx, row in client_group.iterrows():
                try:
                    # Parse times and combine with the route date to handle date rollovers
                    t1 = pd.to_datetime(row['Hora Inicio'], format="%H:%M:%S")
                    t2 = pd.to_datetime(row['Hora Fin'], format="%H:%M:%S")

                    # Check if times were successfully parsed (not None)
                    if t1 is None or pd.isna(t1):
                        raise ValueError(f"Hora Inicio is None or NaN: '{row['Hora Inicio']}'")
                    if t2 is None or pd.isna(t2):
                        raise ValueError(f"Hora Fin is None or NaN: '{row['Hora Fin']}'")

                    # Combine with actual date
                    datetime1 = pd.Timestamp.combine(date.date(), t1.time())
                    datetime2 = pd.Timestamp.combine(date.date(), t2.time())

                    # If end time is before start time, assume it rolled over to next day
                    if datetime2 < datetime1:
                        datetime2 += pd.Timedelta(days=1)

                    unloading += (datetime2 - datetime1).total_seconds() / 3600
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"\nWarning: Could not parse unloading times for {row.get('Empresa', 'unknown')}")
                    print(f"  Error: {e}")
                    print(f"  Date: {date.date()}")
                    print(f"  Route: {route_id}")
                    print(f"  Row index: {idx}")
                    print(f"  Hora Inicio: '{row.get('Hora Inicio', 'N/A')}'")
                    print(f"  Hora Fin: '{row.get('Hora Fin', 'N/A')}'")
                    print(f"  Direccion: '{row.get('Direccion', 'N/A')}'")
                    print(f"  Defaulting to 1.0 hour for this delivery\n")
                    unloading += 1.0

            client_breakdowns.append({
                'name': route_name,
                'Empresa': client,
                'client_proportion': prop,
                'unloading_time': unloading
            })

    return pd.DataFrame(route_summaries), pd.DataFrame(client_breakdowns)

def calculate_distances_and_times(route_summary_df):
    distances, durations, delivery_counts = [], [], []

    for _, row in route_summary_df.iterrows():
        points = row['points']
        coords = list(points.values())
        if len(coords) < 2:
            distances.append(0)
            durations.append(0)
            delivery_counts.append(len(coords))
            continue

        total_km = 0
        total_h = 0
        try:
            for i in range(len(coords) - 1):
                origin = f"{coords[i][0]},{coords[i][1]}"
                dest = f"{coords[i+1][0]},{coords[i+1][1]}"
                result = gmaps.distance_matrix([origin], [dest], mode="driving")['rows'][0]['elements'][0]
                if "distance" in result and "duration" in result:
                    total_km += result['distance']['value'] / 1000
                    total_h += result['duration']['value'] / 3600

            # Return trip
            origin = f"{coords[-1][0]},{coords[-1][1]}"
            dest = f"{coords[0][0]},{coords[0][1]}"
            result = gmaps.distance_matrix([origin], [dest], mode="driving")['rows'][0]['elements'][0]
            if "distance" in result and "duration" in result:
                total_km += result['distance']['value'] / 1000
                total_h += result['duration']['value'] / 3600

        except Exception as e:
            print(f"Error on route {row['name']}: {e}")
            total_km, total_h = 0, 0

        distances.append(total_km)
        durations.append(total_h)
        delivery_counts.append(len(coords))

    route_summary_df['total_km'] = distances
    route_summary_df['eta_hours'] = durations
    route_summary_df['total_delivery_points'] = delivery_counts
    return route_summary_df


def clean_dataframe(routes_df):
    # Drop latitude and longitude columns to simplify the data
    lat_lon_cols = [col for col in routes_df.columns if "_lat" in col or "_lon" in col]
    routes_df = routes_df.drop(columns=lat_lon_cols)

    # Rename columns for clarity
    routes_df = routes_df.rename(columns={
        'unloading_time_h_per_store': 'Unloading time (Total)',
        'eta_hours': 'Driving time (Total)',
        'total_delivery_points': 'Total delivery points',
        'total_km': 'Total Km traveled',
        'driver_wage_per_hour': 'Driver wage',
        'aux_personnel_wage_per_hour': 'Aux personnel wage',
        'num_aux_personnel': 'Number of aux personnel',
        'gas_cost_per_km': 'Gas cost/km',
        'client_proportion': 'Client Proportion'
    })

    # Reorder columns to align with expected output structure
    routes_df = routes_df[[
        'name', 'Empresa', 'Unloading time (Total)', 'Driving time (Total)', 'Total delivery points',
        'Total Km traveled', 'Driver wage', 'Aux personnel wage', 'Number of aux personnel',
        'Gas cost/km', 'Client Proportion'
    ]]

    # Split the 'name' column into separate 'Name' and 'Date' columns
    def split_name_and_date(value):
        match = re.match(r'Ruta (\d+)\.\d+ - (\d{4}-\d{2}-\d{2})', value)
        if match:
            route_number = f"Ruta {match.group(1)}"
            date_str = match.group(2)
            formatted_date = pd.to_datetime(date_str).strftime('%d/%m/%Y')
            return route_number, formatted_date

        special_match = re.match(r'Ruta Especial - (\d{4}-\d{2}-\d{2})', value)
        if special_match:
            route_number = "Ruta Especial"
            date_str = special_match.group(1)
            formatted_date = pd.to_datetime(date_str).strftime('%d/%m/%Y')
            return route_number, formatted_date

        return value, pd.NaT

    routes_df[['Name', 'Date']] = routes_df['name'].apply(lambda x: pd.Series(split_name_and_date(x)))

    # Reorder columns to include the new 'Name' and 'Date' columns
    routes_df = routes_df[[
        'Date', 'Name', 'Empresa', 'Unloading time (Total)', 'Driving time (Total)',
        'Total delivery points', 'Total Km traveled', 'Driver wage', 'Aux personnel wage',
        'Number of aux personnel', 'Gas cost/km', 'Client Proportion'
    ]]

    print("Cleaned Routing Info:\n", routes_df)

    return routes_df

def cost_calculator(df_delivery, df_salary, start_date=None, end_date=None):
    """
    Calculates overtime costs for delivery records using salary information.

    Args:
        df_delivery (pd.DataFrame): Delivery overtime records.
        df_salary (pd.DataFrame): Salary reference table with hourly rates.
        start_date (str or pd.Timestamp): Optional start date filter.
        end_date (str or pd.Timestamp): Optional end date filter.

    Returns:
        pd.DataFrame: Processed delivery DataFrame with calculated cost columns.
    """

    # print("df delivery: \n", df_delivery)
    # print("df_salary: \n", df_salary)

    # === Filter by date if provided ===
    if start_date:
        df_delivery = df_delivery[df_delivery['Fecha'] >= start_date]
    if end_date:
        df_delivery = df_delivery[df_delivery['Fecha'] <= end_date]

    df_salary['Codigo'] = df_salary['Codigo'].astype(str)
    df_delivery['Codigo'] = df_delivery['Codigo'].astype(str)

    # === Merge salary data ===
    df_delivery = pd.merge(df_salary, df_delivery, on='Codigo')

    # === Reorder columns ===
    ordered_columns = [
        'Fecha', 'Cliente', 'Tipo', 'Ruta', 'Puntos de entrega', 'Puntos adicionales',
        'Nombre', 'Cargo', 'Codigo', 'Hora de inicio', 'Hora de finalizacion',
        'Horas diurnas', 'Horas nocturnas', 'Total', 'Aprobacion',
        'Hora diurna', 'Hora nocturna', 'Hora domingo'
    ]
    df_delivery = df_delivery[ordered_columns]

    # === Convert date and ensure numeric types ===
    df_delivery['Fecha'] = pd.to_datetime(df_delivery['Fecha'], dayfirst=True)
    numeric_columns = ['Horas diurnas', 'Horas nocturnas', 'Hora diurna', 'Hora nocturna', 'Hora domingo']
    for col in numeric_columns:
        df_delivery[col] = pd.to_numeric(df_delivery[col], errors='coerce')

    # === Determine if date is Sunday ===
    df_delivery['Is_Sunday'] = df_delivery['Fecha'].dt.dayofweek == 6

    # === Calculate cost based on weekday/weekend ===
    df_delivery['Horas diurnas ($)'] = np.where(
        df_delivery['Is_Sunday'],
        df_delivery['Horas diurnas'] * df_delivery['Hora domingo'],
        df_delivery['Horas diurnas'] * df_delivery['Hora diurna']
    )

    df_delivery['Horas nocturnas ($)'] = np.where(
        df_delivery['Is_Sunday'],
        df_delivery['Horas nocturnas'] * df_delivery['Hora domingo'],
        df_delivery['Horas nocturnas'] * df_delivery['Hora nocturna']
    )

    # === Final cost computation and NaN handling ===
    df_delivery['Horas diurnas ($)'] = df_delivery['Horas diurnas ($)'].fillna(0)
    df_delivery['Horas nocturnas ($)'] = df_delivery['Horas nocturnas ($)'].fillna(0)
    df_delivery['Total ($)'] = df_delivery['Horas diurnas ($)'] + df_delivery['Horas nocturnas ($)']
    df_delivery['Total ($)'] = df_delivery['Total ($)'].fillna(0)

    print("\nDelivery dataframe with overtime cost:\n", df_delivery)

    return df_delivery

def general_cost_calculation(route_summary_df, client_df):
    # Cost calculation at route level for shared costs (gas, driving)
    route_summary_df['gas_cost'] = route_summary_df['total_km'] * route_summary_df['gas_cost_per_km']

    # Calculate total route unloading time for reference
    route_unloading = client_df.groupby('name')['unloading_time'].sum().reset_index()
    route_summary_df = route_summary_df.merge(route_unloading, on='name', how='left', suffixes=('', '_total'))
    route_summary_df['unloading_time'] = route_summary_df['unloading_time'].fillna(0)

    # Calculate driving cost at route level (to be split proportionally by delivery points)
    route_summary_df['driver_driving_cost'] = route_summary_df['driver_wage'] * route_summary_df['eta_hours']

    # Store overtime cost
    route_summary_df['overtime_cost'] = route_summary_df.get('overtime_cost', 0).fillna(0)

    # Merge route data into client breakdown
    merged = pd.merge(client_df, route_summary_df, on='name', how='left')

    # Step 1: get all unique route-date combinations from route_summary_df
    route_dates = route_summary_df[['route_id', 'date']].drop_duplicates()

    # Step 2: find which of those are NOT in overtime_summary
    merged_check = route_dates.merge(merged, on=['route_id', 'date'], how='left', indicator=True)

    # Step 3: keep only the ones that didn't match
    missing = merged_check[merged_check['_merge'] == 'left_only']
    print("Missing overtime rows:\n", missing[['route_id', 'date']])

    # Apply proportional sharing for shared costs (gas, driving time)
    merged['Total gas cost'] = merged['gas_cost'] * merged['client_proportion']
    merged['Total driver driving cost'] = merged['driver_driving_cost'] * merged['client_proportion']
    merged['Total overtime cost'] = merged['overtime_cost'] * merged['client_proportion']

    # Calculate wage costs based on ACTUAL per-client unloading time (not proportional split)
    # unloading_time_x is the per-client unloading time from client_df
    merged['Total driver unloading cost'] = merged['driver_wage'] * merged['unloading_time_x']
    merged['Total aux cost'] = merged['aux_wage'] * merged['num_aux'] * merged['unloading_time_x']

    # Total driver cost = driving cost (proportional) + unloading cost (actual time)
    merged['Total driver cost'] = merged['Total driver driving cost'] + merged['Total driver unloading cost']

    # Total wage cost = driver + aux
    merged['Total wage cost'] = merged['Total driver cost'] + merged['Total aux cost']

    # Total route cost per client
    merged['Total route cost'] = (
        merged['Total gas cost'] +
        merged['Total wage cost'] +
        merged['Total overtime cost']
    )

    # Clean up - use the per-client unloading time
    merged['Unloading time (Total)'] = merged['unloading_time_x']

    # Rename and reorder
    merged = merged.rename(columns={
        'eta_hours': 'Driving time (Total)',
        'total_km': 'Total Km traveled',
        'num_aux': 'Number of aux personnel'
    })

    print("merged dataframe with overtime cost:\n", merged)

    return merged[[
        'date', 'name', 'Empresa', 'Unloading time (Total)', 'Driving time (Total)', 'total_delivery_points',
        'Total Km traveled', 'driver_wage', 'aux_wage', 'Number of aux personnel',
        'Total gas cost', 'Total driver cost', 'Total aux cost', 'Total wage cost', 'Total overtime cost', 'Total route cost', 'client_proportion'
    ]]

def summarize_overtime_by_route(df_delivery_overtime):
    df = df_delivery_overtime.copy()
    df['Ruta'] = df['Ruta'].fillna("Especial").astype(str)
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    overtime_summary = df.groupby(['Ruta', 'Fecha'])['Total ($)'].sum().reset_index()
    overtime_summary.rename(columns={'Ruta': 'route_id', 'Total ($)': 'overtime_cost', 'Fecha':'date'}, inplace=True)
    print("\nOvertime dataframe with overtime cost:\n", overtime_summary)
    return overtime_summary



def calculate_overtime_from_departure_time(df_control, df_salary_overtime, normal_start_hour=8):
    """
    Calculate overtime costs for quoting based on departure time from df_control.

    If departure time ('Hora de salida') is before normal_start_hour, calculate overtime
    cost for the driver and auxiliaries.

    Args:
        df_control (pd.DataFrame): Control dataframe with route information including 'Hora de salida'
        df_salary_overtime (pd.DataFrame): Overtime rates (Hora diurna, Hora nocturna, Hora domingo)
        normal_start_hour (int): Normal start hour (default 8 for 8:00 AM)

    Returns:
        pd.DataFrame: Overtime summary by route with columns: route_id, date, overtime_cost
    """
    df = df_control.copy()

    # Ensure date is datetime
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Ensure Hora de salida is cleaned
    if 'Hora de salida' in df.columns:
        df = clean_time_format(df, 'Hora de salida')
    else:
        print("Warning: 'Hora de salida' column not found in df_control")
        # Return empty overtime summary
        return pd.DataFrame(columns=['route_id', 'date', 'overtime_cost'])

    # Get average overtime rates for driver and aux
    # We'll use nighttime rates since early morning (before 8am) is typically night rate
    driver_overtime_rate = df_salary_overtime[df_salary_overtime['Cargo'] == 'MOTORISTA']['Hora nocturna'].iloc[0] if not df_salary_overtime[df_salary_overtime['Cargo'] == 'MOTORISTA'].empty else 0
    aux_overtime_rate = df_salary_overtime[df_salary_overtime['Cargo'] == 'DESPACHADOR']['Hora nocturna'].iloc[0] if not df_salary_overtime[df_salary_overtime['Cargo'] == 'DESPACHADOR'].empty else 0

    print(f"Using overtime rates - Driver: ${driver_overtime_rate:.2f}/hr, Aux: ${aux_overtime_rate:.2f}/hr")

    # Calculate overtime for each route
    overtime_records = []

    grouped = df.groupby(['Ruta (si fue agregado a una ruta)', 'Fecha'])

    for (route_id, date), group in grouped:
        # Get the departure time (should be same for all rows in a route)
        departure_times = group['Hora de salida'].dropna()

        if departure_times.empty:
            continue

        departure_time = departure_times.iloc[0]

        # Check if departure is before normal start hour
        if isinstance(departure_time, time):
            departure_hour = departure_time.hour + departure_time.minute / 60.0

            if departure_hour < normal_start_hour:
                # Calculate overtime hours
                overtime_hours = normal_start_hour - departure_hour

                # Get number of auxiliaries
                num_aux = group['Num Auxiliares'].max() if not group['Num Auxiliares'].isna().all() else 2

                # Calculate overtime cost: driver + auxiliaries
                overtime_cost = (driver_overtime_rate * overtime_hours) + (aux_overtime_rate * overtime_hours * num_aux)

                overtime_records.append({
                    'route_id': route_id,
                    'date': date,
                    'overtime_cost': overtime_cost,
                    'overtime_hours': overtime_hours,
                    'departure_time': str(departure_time)
                })

    if overtime_records:
        overtime_summary = pd.DataFrame(overtime_records)
        print("\nOvertime calculated from departure times:\n", overtime_summary)
    else:
        overtime_summary = pd.DataFrame(columns=['route_id', 'date', 'overtime_cost'])
        print("\nNo overtime detected from departure times")

    return overtime_summary[['route_id', 'date', 'overtime_cost']]



def pricing(routes_calc_df, margin, method):
    """
    Calculate price based on cost and margin, using either cost-based or selling price-based margin method.

    Args:
        routes_calc_df (pd.DataFrame): DataFrame containing at least a 'Total route cost' column.
        margin (float): Margin value between 0 and 1 (exclusive).
        method (str): 'cost' for cost-based margin or 'price' for selling price-based margin.

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'Price' column.
    """
    if not 0 <= margin < 1:
        raise ValueError(f"Margin must be between 0 (inclusive) and 1 (exclusive). Got: {margin}")

    if method not in ['cost', 'price']:
        raise ValueError(f"Method must be either 'cost' or 'price'. Got: {method}")

    if 'Total route cost' not in routes_calc_df.columns:
        raise KeyError("'Total route cost' column not found in the DataFrame.")

    routes_calc_df = routes_calc_df.copy()

    if method == 'cost':
        routes_calc_df['Price'] = routes_calc_df['Total route cost'] * (1 + margin)
    elif method == 'price':
        routes_calc_df['Price'] = routes_calc_df['Total route cost'] / (1 - margin)

    return routes_calc_df



def write_to_excel_with_individual_formatting(output_file, cost_df):
    # Define sheet names and corresponding DataFrames
    sheet_mapping = {
        'Cotizacion de ruteo': cost_df,
    }
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1})

        # Define simplified custom formats compatible with xlsxwriter
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})  # Simple date format
        regular_number_format = workbook.add_format({'num_format': '#,##0.00'})  # Number format with two decimals
        accounting_format = workbook.add_format(
            {'num_format': '$#,##0.00'})  # Simple accounting format with two decimals

        # Write each DataFrame to its sheet and format individually
        for sheet_name, dataframe in sheet_mapping.items():
            # Write the DataFrame to the sheet
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            # Apply header formatting for each column header
            for col_num, header in enumerate(dataframe.columns):
                worksheet.write(0, col_num, header, header_format)

                # Set the column width to fit the header length with a bit of padding
                header_length = len(header) + 2
                worksheet.set_column(col_num, col_num, header_length)

            # Apply specific formatting for "Horas en bodega" sheet (Sheet 1)
            if sheet_name == 'Cotizacion de ruteo':
                # Column 1 (Date) formatting
                worksheet.set_column(0, 0, 20, date_format)  # Date
                worksheet.set_column(1, 1, 20)  # Route name
                worksheet.set_column(2, 2, 18)   # Client
                worksheet.set_column(3, 3, 18, regular_number_format)   # Unloading time
                worksheet.set_column(4, 4, 20, regular_number_format)   # Driving time
                worksheet.set_column(5, 5, 18, regular_number_format)   # Total delivery points
                worksheet.set_column(6, 6, 22, regular_number_format)   # Total km traveled
                worksheet.set_column(7, 7, 18, accounting_format)   # Driver wage
                worksheet.set_column(8, 8, 18, accounting_format)   # Aux wage
                worksheet.set_column(9, 9, 18, regular_number_format)  # Number of aux operators
                worksheet.set_column(10, 10, 18, accounting_format)  # Total gas cost
                worksheet.set_column(11, 11, 18, accounting_format)  # Total driver cost
                worksheet.set_column(12, 12, 18, accounting_format)  # Total aux cost
                worksheet.set_column(13, 13, 18, accounting_format)  # Total wage cost
                worksheet.set_column(14, 14, 18, accounting_format)  # Total overtime cost
                worksheet.set_column(15, 15, 18, accounting_format)  # Total route cost
                worksheet.set_column(16, 16, 18, regular_number_format)  # Client proportion
                worksheet.set_column(17, 17, 18, accounting_format)  # Price

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


def check_join_coverage(route_summary_df, overtime_summary):
    """
    Verifies that (route_id, date) keys match between route_summary_df and overtime_summary.
    Reports:
      1) Overtime keys that don't exist in routes (OT without Route)
      2) Route keys that don't exist in overtime (Route without OT)
      3) Duplicate keys on either side (can cause row-multiplication on merge)
    Returns: (ot_without_rt, rt_without_ot, dup_rt_keys, dup_ot_keys) as DataFrames
    """

    # 0) Canonicalize keys (idempotent; safe if you already normalized)
    def fmt_route(x):
        s = str(x).strip()
        return s[:-2] if s.endswith('.0') else s

    for df in (route_summary_df, overtime_summary):
        df['route_id'] = df['route_id'].map(fmt_route)
        df['date']     = pd.to_datetime(df['date']).dt.normalize()

    # 1) Unique key sets
    rt_keys = route_summary_df[['route_id','date']].drop_duplicates()
    ot_keys = overtime_summary[['route_id','date']].drop_duplicates()

    # 2) Anti-joins (missing on each side)
    ot_without_rt = (
        ot_keys.merge(rt_keys, on=['route_id','date'], how='left', indicator=True)
               .query('_merge=="left_only"')
               .drop(columns=['_merge'])
               .sort_values(['route_id','date'])
    )

    rt_without_ot = (
        rt_keys.merge(ot_keys, on=['route_id','date'], how='left', indicator=True)
               .query('_merge=="left_only"')
               .drop(columns=['_merge'])
               .sort_values(['route_id','date'])
    )

    # 3) Duplicate keys (can multiply rows on merge)
    dup_rt_mask = route_summary_df.duplicated(['route_id','date'], keep=False)
    dup_ot_mask = overtime_summary.duplicated(['route_id','date'], keep=False)

    dup_rt_keys = (route_summary_df.loc[dup_rt_mask, ['route_id','date']]
                   .drop_duplicates()
                   .sort_values(['route_id','date']))
    dup_ot_keys = (overtime_summary.loc[dup_ot_mask, ['route_id','date']]
                   .drop_duplicates()
                   .sort_values(['route_id','date']))

    # 4) Summary prints
    print(f"OT without Route: {len(ot_without_rt)}")
    if not ot_without_rt.empty:
        print(ot_without_rt.head(20))

    print(f"Route without OT: {len(rt_without_ot)}")
    if not rt_without_ot.empty:
        print(rt_without_ot.head(20))

    print(f"Duplicate keys in routes: {len(dup_rt_keys)}")
    if not dup_rt_keys.empty:
        print(dup_rt_keys.head(20))

    print(f"Duplicate keys in overtime: {len(dup_ot_keys)}")
    if not dup_ot_keys.empty:
        print(dup_ot_keys.head(20))

    return ot_without_rt, rt_without_ot, dup_rt_keys, dup_ot_keys


def prompt_user_to_continue(overtime_summary, route_summary_df, ov_miss, rt_miss,
                            allow_non_interactive=False, default_continue=False):
    """
    Print anomaly details and prompt the user to continue.
    - allow_non_interactive: if True, no prompt is shown; proceed according to default_continue.
    - default_continue: what to do when non-interactive.
    Returns True (continue) or False (abort).
    """
    # Build detailed views
    ot_orphans = (
        overtime_summary.merge(ov_miss, on=['route_id','date'], how='inner')
        .sort_values(['route_id','date'])
    ) if not ov_miss.empty else overtime_summary.iloc[0:0].copy()

    routes_without_ot = (
        route_summary_df.merge(rt_miss, on=['route_id','date'], how='inner')
        [['route_id','date','name']]
        .sort_values(['route_id','date'])
    ) if not rt_miss.empty else route_summary_df.iloc[0:0].copy()

    # Pretty print compact summary
    print("\n=== DATA CHECK: Join coverage anomalies ===")
    print(f"OT without Route: {len(ov_miss)}")
    if not ot_orphans.empty:
        # Show just a few lines, include amounts
        to_show = min(10, len(ot_orphans))
        print(ot_orphans[['route_id','date','overtime_cost']].head(to_show).to_string(index=False))
        if len(ot_orphans) > to_show:
            print(f"... ({len(ot_orphans) - to_show} more)")

    print(f"\nRoute without OT: {len(rt_miss)}")
    if not routes_without_ot.empty:
        to_show = min(10, len(routes_without_ot))
        print(routes_without_ot.head(to_show).to_string(index=False))
        if len(routes_without_ot) > to_show:
            print(f"... ({len(routes_without_ot) - to_show} more)")

    if allow_non_interactive:
        print(f"\n[NON-INTERACTIVE MODE] Proceeding = {default_continue}")
        return bool(default_continue)

    # Interactive prompt
    while True:
        ans = input("\nProceed with current data? [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")

def main():
    pd.set_option(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.expand_frame_repr", False
    )

    # === Configuration ===
    # Set mode: 'billing' or 'quoting'
    mode = 'billing'
    # mode = 'quoting'

    start_date_str = "26/10/25"
    end_date_str = "25/11/25"

    # margin = input("Enter margin as decimal points (e.g. 0.3 for 30%): ")
    # margin = float(margin)
    # method = input("Select 'cost' or 'price': ")

    margin = 0.35
    margin = float(margin)
    method = 'price'

    # Convert dates
    try:
        start_date = pd.Timestamp(parse_date(start_date_str))
        end_date = pd.Timestamp(parse_date(end_date_str))
        if start_date > end_date:
            print("Error: Start date must be before or equal to end date.")
            return
    except ValueError as e:
        print(e)
        return

    print(f"Main: Running in {mode.upper()} mode")
    print("Main: Loading data...")

    # === Load and prepare data ===
    df_delivery_overtime, df_salary, df_control, df_rutas, df_camiones, df_precios, df_salary_overtime = load_data(mode=mode)

    df_control = filter_data_by_date(df_control, 'Fecha', start_date, end_date)
    df_delivery_overtime = filter_data_by_date(df_delivery_overtime, 'Fecha', start_date, end_date)

    df_control = clean_time_format(df_control, 'Hora Inicio')
    df_control = clean_time_format(df_control, 'Hora Fin')
    df_control = assign_special_route(df_control)

    # === Calculate overtime costs based on mode ===
    if mode == 'billing':
        # Billing mode: use overtime from separate overtime document
        print("\n=== BILLING MODE: Using overtime from overtime document ===")
        df_delivery_overtime = cost_calculator(df_delivery_overtime, df_salary_overtime, start_date, end_date)
        overtime_summary = summarize_overtime_by_route(df_delivery_overtime)
    else:  # quoting
        # Quoting mode: calculate overtime from departure time in df_control
        print("\n=== QUOTING MODE: Calculating overtime from departure times ===")
        overtime_summary = calculate_overtime_from_departure_time(df_control, df_salary_overtime, normal_start_hour=8)

    # === Process route and client data ===
    route_summary_df, client_df = process_control_df(df_control, df_salary, df_camiones, df_precios)
    route_summary_df = calculate_distances_and_times(route_summary_df)

    print("route_summary_df: \n", route_summary_df)
    print("overtime_summary: \n", overtime_summary)

    # Check if there are any routes to process
    if route_summary_df.empty:
        control_sheet = 'Control de Rutas y Fletes' if mode == 'billing' else 'Costeo'
        print(f"\n{'='*60}")
        print(f"WARNING: No routes found for the date range {start_date.date()} to {end_date.date()}")
        print(f"{'='*60}")
        print("\nPlease check:")
        print("  1. The date range is correct")
        print("  2. Routes exist in the control document for these dates")
        print(f"  3. The sheet '{control_sheet}' has data for this period")
        return

    route_summary_df = route_summary_df.copy()
    overtime_summary = overtime_summary.copy()

    # Normalize keys for merging
    route_summary_df['route_id'] = route_summary_df['route_id'].astype(str).str.strip()
    route_summary_df['date'] = pd.to_datetime(route_summary_df['date']).dt.normalize()

    if not overtime_summary.empty:
        overtime_summary['route_id'] = overtime_summary['route_id'].astype(str).str.strip()
        overtime_summary['date'] = pd.to_datetime(overtime_summary['date']).dt.normalize()

    # === JOIN COVERAGE CHECK: Only for billing mode ===
    if mode == 'billing':
        print("\n=== BILLING MODE: Checking data consistency between routes and overtime ===")
        ot_missing, rt_missing, dup_rt, dup_ot = check_join_coverage(route_summary_df, overtime_summary)

        # Bring some context
        if not ot_missing.empty:
            print("\n>>> Overtime without Route — with OT amounts")
            print(
                overtime_summary
                .merge(ot_missing, on=['route_id', 'date'], how='inner')
                .sort_values(['route_id', 'date'])
            )

        if not rt_missing.empty:
            print("\n>>> Routes without Overtime — with route names")
            print(
                route_summary_df
                .merge(rt_missing, on=['route_id', 'date'], how='inner')
                [['route_id', 'date', 'name']]
                .sort_values(['route_id', 'date'])
            )

        # Interactively decide whether to proceed when there are gaps
        if len(ot_missing) or len(rt_missing):
            proceed = prompt_user_to_continue(
                overtime_summary=overtime_summary,
                route_summary_df=route_summary_df,
                ov_miss=ot_missing,
                rt_miss=rt_missing,
                allow_non_interactive=False,  # set True on servers/cron
                default_continue=False  # default action if non-interactive
            )
            if not proceed:
                print("Aborting by user choice due to data mismatches.")
                return
    else:
        print("\n=== QUOTING MODE: Skipping join coverage check (not applicable) ===")

    # Merge overtime with routes
    route_summary_df = route_summary_df.merge(
        overtime_summary, on=['route_id', 'date'], how='left'
    )

    route_summary_df['overtime_cost'] = route_summary_df['overtime_cost'].fillna(0)

    # === Merge route cost into client shares ===
    routing_cost = general_cost_calculation(route_summary_df, client_df)

    # === Apply pricing logic ===
    routing_price = pricing(routing_cost, margin, method)

    print("Final output:\n", routing_price)

    # === Export to Excel ===
    # Create dynamic filename using start and end dates
    start_date_formatted = start_date.strftime('%d%m%y')
    end_date_formatted = end_date.strftime('%d%m%y')
    filename = f'Precios de ruteo cotizacion {start_date_formatted}-{end_date_formatted}.xlsx'
    output_file = os.path.join(get_base_output_path(), filename)
    write_to_excel_with_individual_formatting(output_file, routing_price)
    print(f"\nDataFrames have been successfully written to {output_file}\n")


if __name__ == "__main__":
    main()
