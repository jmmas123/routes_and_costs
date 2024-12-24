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
    df_delivery_overtime = pd.read_excel(overtime_file_path, sheet_name='Horas en ruta', header=0,
                                         dtype={'Codigo': str})
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
                return pd.to_datetime(cleaned_time, format="%H:%M:%S").time()
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

    diesel_price_per_gallon = latest_prices['Superior'].iloc[0]

    return diesel_price_per_gallon


def assign_special_route(df_control):
    # Assign "Ruta Especial" to rows with no specific route
    df_control['Ruta (si fue agregado a una ruta)'] = (df_control['Ruta (si fue agregado a una ruta)'].fillna
                                                      ("Especial"))
    return df_control

def process_control_df(df_control, df_salary, df_camiones, df_precios):
    routes = []

    # Create a copy of the DataFrame to avoid modifying the original
    df = df_control.copy()

    # Ensure 'Fecha' is in datetime format
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Group by 'Ruta' and 'Fecha'
    grouped = df.groupby(['Ruta (si fue agregado a una ruta)', 'Fecha'])

    for (route_id, date), group in grouped:
        # Skip rows without a valid route identifier
        if pd.isna(route_id):
            continue

        # Identify all distinct clients on this route
        clients = group['Empresa'].unique()

        # Total delivery points in the route
        total_delivery_points = len(group)

        for client in clients:
            # Filter the group for the specific client
            client_group = group[group['Empresa'] == client]

            # Client-specific delivery points
            client_delivery_points = len(client_group)

            # Calculate client proportion
            client_proportion = client_delivery_points / total_delivery_points if total_delivery_points > 0 else 0

            # Extract route details
            empresa_name = client
            points = {
                "PLISA": (13.814771381058584, -89.40960526517033)  # Add the starting point if needed
            }

            # Add each client's delivery points
            for idx, row in client_group.iterrows():
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

            # Calculate unloading time per store for the client
            unloading_times = []
            for idx, row in client_group.iterrows():
                hora_inicio = row['Hora Inicio']
                hora_fin = row['Hora Fin']
                if pd.notna(hora_inicio) and pd.notna(hora_fin):
                    try:
                        time_format = "%H:%M:%S"  # Handle AM/PM format
                        t_inicio = pd.to_datetime(hora_inicio, format=time_format)
                        t_fin = pd.to_datetime(hora_fin, format=time_format)
                        unloading_time = (t_fin - t_inicio).total_seconds() / 3600  # in hours
                        unloading_times.append(unloading_time)
                    except Exception as e:
                        print(f"Error parsing times at index {idx}: {e}")
                else:
                    # Default unloading time per store
                    unloading_times.append(1.0)

            # Calculate total unloading time for the client
            unloading_time_h_per_store = sum(unloading_times) if unloading_times else len(client_group)

            # Fetch driver and auxiliary personnel details
            placa_vehiculo = group['Placa Vehiculo'].iloc[0] if not group['Placa Vehiculo'].isna().all() else None
            driver_cargo = 'MOTORISTA'
            efficiency_km_per_gallon = 0.3  # Default value

            if placa_vehiculo:
                camion_info = df_camiones[df_camiones['Placa'] == placa_vehiculo]
                if not camion_info.empty:
                    efficiency_km_per_gallon = camion_info['Eficiencia (km/gal)'].iloc[0]

            driver_salary_info = df_salary[df_salary['Cargo'] == driver_cargo]
            driver_wage_per_hour = driver_salary_info['Salario/Hora'].iloc[0] if not driver_salary_info.empty else 2.5

            aux_cargo = 'DESPACHADOR'
            aux_salary_info = df_salary[df_salary['Cargo'] == aux_cargo]
            aux_personnel_wage_per_hour = aux_salary_info['Salario/Hora'].iloc[0] if not aux_salary_info.empty else 2.0
            num_aux_personnel = group['Num Auxiliares'].max() if not group['Num Auxiliares'].isna().all() else 2

            # Fuel price calculation
            fuel_price_per_gallon = get_fuel_price_on_date(df_precios, date)
            if fuel_price_per_gallon is None:
                fuel_price_per_gallon = 4.0  # Default value

            gas_cost_per_km = fuel_price_per_gallon / efficiency_km_per_gallon if efficiency_km_per_gallon else 0.3

            # Create the route record
            route = {
                "name": f"Ruta {route_id} - {date.date()}",
                "Empresa": empresa_name,
                "unloading_time_h_per_store": unloading_time_h_per_store * client_proportion,
                "driver_wage_per_hour": driver_wage_per_hour,
                "aux_personnel_wage_per_hour": aux_personnel_wage_per_hour,
                "num_aux_personnel": num_aux_personnel,
                "gas_cost_per_km": gas_cost_per_km,
                "client_proportion": client_proportion,
                **{f"{point}_lat": lat for point, (lat, lon) in points.items()},
                **{f"{point}_lon": lon for point, (lat, lon) in points.items()}
            }

            routes.append(route)

    # Convert routes to a DataFrame
    routes_df = pd.DataFrame(routes)

    # Verify that proportions for each route sum to 1
    route_sums = routes_df.groupby('name')['client_proportion'].sum()
    if not all(np.isclose(route_sums, 1.0)):
        print("Warning: Proportions do not sum to 1 for all routes!")
        print(route_sums[~np.isclose(route_sums, 1.0)])

    print("Updated Route data:\n", routes_df)

    return routes_df

def calculate_distances_and_times(routes_df):
    """
    Calculate total distances and travel times for routes, including round trips.
    """
    # Initialize lists for distance, duration, and delivery points
    total_distances = []
    eta_hours_list = []
    delivery_points_count = []

    for _, row in routes_df.iterrows():
        # Extract latitude and longitude pairs
        points = {
            col.replace("_lat", ""): (row[f"{col}"], row[f"{col.replace('_lat', '_lon')}"])
            for col in routes_df.columns if "_lat" in col and f"{col.replace('_lat', '_lon')}" in routes_df.columns
        }

        # Remove invalid points where either latitude or longitude is NaN
        points = {name: coords for name, coords in points.items() if not pd.isna(coords[0]) and not pd.isna(coords[1])}

        # Count valid delivery points
        delivery_points_count.append(len(points))

        # Skip if there are less than two valid points (cannot calculate distances)
        if len(points) < 2:
            total_distances.append(0)
            eta_hours_list.append(0)
            continue

        # Prepare waypoints as a list of coordinates
        waypoints = list(points.values())
        waypoints_str = [f"{lat},{lon}" for lat, lon in waypoints]

        total_distance = 0
        total_duration = 0

        # Calculate distance and ETA for consecutive points
        try:
            for i in range(len(waypoints_str) - 1):
                origin = waypoints_str[i]
                destination = waypoints_str[i + 1]

                # Request distance and duration from Google Maps API
                result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
                element = result["rows"][0]["elements"][0]

                # Check if the response has valid distance and duration
                if "distance" in element and "duration" in element:
                    distance = element["distance"]["value"]  # meters
                    duration = element["duration"]["value"]  # seconds

                    total_distance += distance / 1000  # Convert meters to kilometers
                    total_duration += duration / 3600  # Convert seconds to hours

            # Add the return leg to make it a round trip
            origin = waypoints_str[-1]
            destination = waypoints_str[0]
            result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
            element = result["rows"][0]["elements"][0]

            if "distance" in element and "duration" in element:
                return_distance = element["distance"]["value"]  # meters
                return_duration = element["duration"]["value"]  # seconds

                total_distance += return_distance / 1000  # Convert meters to kilometers
                total_duration += return_duration / 3600  # Convert seconds to hours

        except Exception as e:
            print(f"Error calculating distance or duration: {e}")
            total_distance = 0
            total_duration = 0

        # Append results for the current route
        total_distances.append(total_distance)
        eta_hours_list.append(total_duration)

    # Add results as new columns in the DataFrame
    routes_df["total_km"] = total_distances
    routes_df["eta_hours"] = eta_hours_list
    routes_df["total_delivery_points"] = delivery_points_count


    print("Route data:\n", routes_df)

    return routes_df


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


def general_cost_calculation(routes_df):
    # Ensure the date column is properly extracted
    if 'Date' not in routes_df.columns:
        routes_df['Date'] = routes_df['Name'].str.extract(r'- (\d{4}-\d{2}-\d{2})')[0]

    # Assign "Ruta Especial" to rows without a route
    routes_df['Name'] = routes_df['Name'].fillna("Ruta Especial")

    # Calculate total gas cost per route
    routes_df['Total gas cost'] = routes_df['Total Km traveled'] * routes_df['Gas cost/km']

    # Calculate total wage cost
    routes_df['Total wage cost'] = (
        (routes_df['Driver wage'] + (routes_df['Aux personnel wage'] * routes_df['Number of aux personnel'])) *
        (routes_df['Unloading time (Total)'] + routes_df['Driving time (Total)'])
    )

    # Calculate total cost of the route
    routes_df['Total route cost'] = routes_df['Total gas cost'] + routes_df['Total wage cost']

    # Select columns to display the cost breakdown
    cost_columns = [
        'Date', 'Name', 'Empresa', 'Total delivery points', 'Total Km traveled', 'Total gas cost',
        'Unloading time (Total)', 'Driving time (Total)', 'Number of aux personnel',
        'Total wage cost', 'Total route cost'
    ]
    cost_df = routes_df[cost_columns]

    print("Cost Breakdown:\n", cost_df)

    return cost_df



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
                worksheet.set_column(3, 3, 18, regular_number_format)   # Total delivery points
                worksheet.set_column(4, 4, 20, regular_number_format)   # Total km traveled
                worksheet.set_column(5, 5, 18, accounting_format)   # Total gas cost
                worksheet.set_column(6, 6, 22, regular_number_format)   # Unloading time
                worksheet.set_column(7, 7, 18, regular_number_format)   # Driving time
                worksheet.set_column(8, 8, 18, regular_number_format)   # Number of aux personnel
                worksheet.set_column(9, 9, 18, accounting_format)  # Total wage Cost
                worksheet.set_column(10, 10, 18, accounting_format)  # Total route cost


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

    df_control = assign_special_route(df_control)

    # Process Control df to create routes
    routes_df = process_control_df(df_control, df_salary, df_camiones, df_precios)

    routes_calc = calculate_distances_and_times(routes_df)

    routes_calc_df = clean_dataframe(routes_calc)

    routing_cost = general_cost_calculation(routes_calc_df)

    # Define the output path and file name
    output_file = os.path.join(get_base_output_path(), 'Costos de ruteo.xlsx')

    write_to_excel_with_individual_formatting(output_file, routing_cost)
    print(f"\nDataFrames have been successfully written to {output_file}\n")

if __name__ == "__main__":
    main()
