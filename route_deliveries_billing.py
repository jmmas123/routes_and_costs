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
    df_salary = pd.read_excel(workforce_and_salaries_path, sheet_name='Empleados', header=0)
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
