import numpy as np
import pandas as pd
import os

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

def process_control_df(control_df, df_salary, df_camiones, precios_df):
    routes = []

    # Create a copy of the DataFrame to avoid modifying the original
    df = control_df.copy()

    # Create a 'Ruta' column by converting 'Ruta (si fue agregado a una ruta)' to string
    df['Ruta'] = df['Ruta (si fue agregado a una ruta)'].astype(str)

    # Ensure 'Fecha' is in datetime format
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Create a unique route identifier by combining 'Ruta' and 'Fecha'
    # For routes without a number ('nan'), create a unique identifier using the date and index
    def generate_route_id(row):
        if row['Ruta'] == 'nan' or pd.isna(row['Ruta']):
            return f"No_Ruta_{row['Fecha'].date()}_{row.name}"
        else:
            return f"Ruta_{row['Ruta']}_{row['Fecha'].date()}"

    df['Route_ID'] = df.apply(generate_route_id, axis=1)

    # Now group by 'Route_ID'
    grouped = df.groupby('Route_ID')

    for route_id, group in grouped:
        # For each group (route), create a route dict
        route_name = f"{route_id}"
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
            unloading_time_h_per_store = np.mean(unloading_times)
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

            # Get driver wage per hour from Salaries df
            print(f"Looking up salary for driver cargo: '{driver_cargo}'")
            driver_salary_info = df_salary[df_salary['Cargo'] == driver_cargo]
            print(f"Driver salary info:\n{driver_salary_info}")

            if not driver_salary_info.empty:
                try:
                    driver_wage_per_hour = float(driver_salary_info['Salario/Hora'].iloc[0])
                    print(f"Driver wage per hour: {driver_wage_per_hour}")
                except (IndexError, KeyError, ValueError) as e:
                    print(f"Error retrieving 'Salario/Hora' for cargo '{driver_cargo}': {e}")
                    print(f"Using default driver wage per hour: 2.5")
                    driver_wage_per_hour = 2.5  # Default value
            else:
                print(f"No salary information found for cargo '{driver_cargo}'. Using default value 2.5.")
                driver_wage_per_hour = 2.5  # Default value

            # Determine auxiliary personnel wage per hour
            aux_cargo = 'DESPACHADOR'  # Assuming 'DESPACHADOR' as the role for auxiliary personnel
            aux_salary_info = df_salary[df_salary['Cargo'] == aux_cargo]
            if not aux_salary_info.empty:
                aux_personnel_wage_per_hour = aux_salary_info['Salario/Hora'].iloc[0]
            else:
                print(f"No salary information found for cargo '{aux_cargo}'. Using default value 2.0.")
                aux_personnel_wage_per_hour = 2.0  # Default value

            # Get the number of auxiliary personnel (take max value or default)
            num_aux_personnel = group['Num Auxiliares'].max()
            if pd.isna(num_aux_personnel):
                num_aux_personnel = 2  # Default value

        # Assign average speed
        average_speed_kmh = 60  # Default value

        # Get the date of the route
        route_date = group['Fecha'].iloc[0]

        # Find the fuel price for the date and zone
        fuel_price_per_gallon = get_fuel_price_on_date(precios_df, route_date)

        if fuel_price_per_gallon is None:
            print(f"No fuel price found for date {route_date}. Using default value $4.00 per gallon.")
            fuel_price_per_gallon = 4.00  # Default value

            # Calculate gas cost per km
            if efficiency_km_per_gallon is not None:
                gas_cost_per_km = fuel_price_per_gallon / efficiency_km_per_gallon
            else:
                print(
                    f"Efficiency per gallon is not available for route '{route_name}'. Cannot calculate gas cost per km.")
                gas_cost_per_km = None  # Or assign a default value if appropriate

        route = {
            "name": route_name,
            "points": points,
            "unloading_time_h_per_store": unloading_time_h_per_store,
            "driver_wage_per_hour": driver_wage_per_hour,
            "aux_personnel_wage_per_hour": aux_personnel_wage_per_hour,
            "num_aux_personnel": int(num_aux_personnel),
            "average_speed_kmh": average_speed_kmh,
            "gas_cost_per_km": gas_cost_per_km,
            # Add other fields as needed
        }

        routes.append(route)

    return routes


def main():
    pd.set_option(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.expand_frame_repr", False
    )

    print("Main: Loading data...")

    df_delivery_overtime, df_salary, df_control, df_rutas, df_camiones, df_precios = load_data()

    # Process Control df to create routes
    routes = process_control_df(df_control, df_camiones, df_salary, df_precios)

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
