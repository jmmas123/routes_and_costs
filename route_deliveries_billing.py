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
            if file_type == 'overtime_t':
                return '/Users/j.m./Library/Mobile Documents/com~apple~CloudDocs/GM/MOBU - OPL/Horas extra'
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
    df_control = pd.read_excel(income_overtime_client_path, sheet_name='Control de Rutas y fletes')
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

def process_control_df(control_df, df_salary, df_camiones):
    routes = []

    # Identify rows where 'Ruta' is NaN (after conversion, these will be 'nan' strings)
    nan_route_mask = control_df['Ruta'] == 'nan'

    # Assign unique route identifiers to NaN 'Ruta' rows
    control_df.loc[nan_route_mask, 'Ruta'] = 'Special Delivery' + control_df[nan_route_mask].index.astype(str)

    # Ensure that 'Ruta' is treated as a string
    control_df['Ruta'] = control_df['Ruta (si fue agregado a una ruta)'].astype(str)

    # Group by 'Ruta'
    grouped = control_df.groupby('Ruta')

    for route_number, group in grouped:
        # For each group (route), create a route dict
        route_name = f"Ruta {route_number}"
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
            else:
                # Match the placa with Camiones df to get capacity
                camion_info = df_camiones[df_camiones['Placa'] == placa_vehiculo]
                if not camion_info.empty:
                    capacidad_ton = camion_info['Capacidad (Ton)'].iloc[0]
                    if capacidad_ton > 10:
                        driver_cargo = 'MOTORISTA LICENCIA PESADA'
                    else:
                        driver_cargo = 'MOTORISTA'
                else:
                    print(f"No truck information found for placa '{placa_vehiculo}'. Defaulting to 'MOTORISTA'.")
                    driver_cargo = 'MOTORISTA'

            # Get driver wage per hour from Salaries df
            driver_salary_info = df_salary[df_salary['Cargo'] == driver_cargo]
            if not driver_salary_info.empty:
                driver_wage_per_hour = driver_salary_info['Salario/Hora'].iloc[0]
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

        route = {
            "name": route_name,
            "points": points,
            "unloading_time_h_per_store": unloading_time_h_per_store,
            "driver_wage_per_hour": driver_wage_per_hour,
            "aux_personnel_wage_per_hour": aux_personnel_wage_per_hour,
            "num_aux_personnel": int(num_aux_personnel),
            "average_speed_kmh": average_speed_kmh,
            # 'gas_cost_per_km' will be computed later
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

    pd.set_option(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.expand_frame_repr", False
    )

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
