
import re
import numpy as np
import pandas as pd
import time
from sqlalchemy import create_engine
from asammdf import MDF
from config_utils import cache_data

@cache_data
def extract_temperatures_from_mf4(mf4_path):
    print(f"Loading MF4 file: {mf4_path}")
    mdf = MDF(mf4_path)
    pattern = re.compile(r"moduletemperature(\d+)_bms(\d+)", re.I)

    data_list, sensor_ids = [], []
    for ch in mdf.iter_channels():
        m = pattern.fullmatch(ch.name.lower())
        if not m:
            continue
        samples = ch.samples
        if samples.dtype.kind not in "fiu":
            print(f"Ignored non-numeric signal: {ch.name}")
            continue
        num = int(m.group(1))
        bms = m.group(2).zfill(2)
        sensor_ids.append(((num, bms), ch.name))
        data_list.append(samples.astype(float))

    if not data_list:
        raise RuntimeError("Keine gültigen Temperaturkanäle gefunden!")

    min_len = min(len(a) for a in data_list)
    temps = np.vstack([a[:min_len] for a in data_list])
    print(f"{temps.shape[0]} gültige Temperatur-Kanäle · {min_len} Samples")
    return temps, sensor_ids

@cache_data
def extract_inlet_outlet_flow(db_path, file_id_value, lookup_table, cache_filename=None, force_refresh=False):
    start_time = time.time()
    engine = create_engine(f'sqlite:///{db_path}')

    inlet_temperature = []
    outlet_temperature = []
    coolant_flow = []

    signal_info = lookup_table[lookup_table['File.ID'] == file_id_value]
    signals = {
        'inlet': {'SensorNumber': 101, 'data': None},
        'outlet': {'SensorNumber': 102, 'data': None},
        'flow': {'SensorNumber': 103, 'data': None},
    }

    for key in signals.keys():
        signal_entries = signal_info[signal_info['SensorNumber'] == signals[key]['SensorNumber']]
        for _, signal_entry in signal_entries.iterrows():
            table_name = signal_entry['Table.Name']
            column_name = signal_entry['Channel.Name']
            query = f"SELECT {column_name} FROM {table_name} WHERE file_id = ?"
            try:
                df = pd.read_sql_query(query, engine, params=(file_id_value,))
                df.dropna(inplace=True)
                if not df.empty:
                    signals[key]['data'] = df[column_name].values
                    print(f"Found {key} data in table {table_name}")
                    break
            except Exception as e:
                print(f"Error processing {key} data from table {table_name}: {e}")

    inlet_temperature = signals['inlet']['data'] or []
    outlet_temperature = signals['outlet']['data'] or []
    coolant_flow = signals['flow']['data'] or []

    end_time = time.time()
    print(f"Inlet/Outlet/Flow data extraction took {end_time - start_time:.2f} seconds")

    return inlet_temperature, outlet_temperature, coolant_flow

def calculation_heat_flux(volumenstrom, temp_inlet, temp_outlet):
    cw, cg = 4186, 3350
    pw, pg = 0.5, 0.5
    rho_w, rho_g = 1000, 1070
    delta_t = temp_outlet - temp_inlet
    if volumenstrom == 0 or delta_t == 0:
        return 0
    return volumenstrom * delta_t * (pw * cw * rho_w + pg * cg * rho_g)
