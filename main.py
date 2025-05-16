import os
import json
import numpy as np
from config_utils import load_config
from data_extraction import extract_temperatures_from_mf4
from interface import interactive_battery_layout  # neue Version mit Radiobuttons & Sprungbutton

def load_sensor_order(name):
    layout_dir = os.path.join(os.path.dirname(__file__), "layouts")
    layout_path = os.path.join(layout_dir, f"{name}.json")
    if not os.path.exists(layout_path):
        print(f"Error: Layout file {layout_path} not found.")
        return []
    with open(layout_path, "r") as f:
        config = json.load(f)
    return [tuple(x) for x in config["sensor_order"]]

def main():
    # Eingabedaten
    mf4_path = "raw_data/Trigger1Converted.mf4"
    vmin, vmax = 15.0, 40.0
    sensors_per_module_list = [2] * 6
    strings_count = 6

    # Lade Temperaturdaten
    temperatures, sensor_ids = extract_temperatures_from_mf4(mf4_path)

    # Dummywerte für Flow-Daten
    inlet_temp = outlet_temp = flow = []

    custom_sensor_order = load_sensor_order("HVB_340_800_L")

    interactive_battery_layout(
        temperatures,
        sensor_ids,
        sensors_per_module_list,
        strings_count,
        custom_sensor_order,
        inlet_temp=inlet_temp,
        outlet_temp=outlet_temp,
        flow=flow,
        vmin=vmin,
        vmax=vmax,
        mf4_path=mf4_path
    )


if __name__ == "__main__":
    main()
