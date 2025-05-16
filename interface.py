
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
import os
import json
import pandas as pd
import matplotlib.gridspec as gridspec
from data_extraction import extract_temperatures_from_mf4, extract_inlet_outlet_flow, calculation_heat_flux
from visualization import plot_battery_layout

def interactive_battery_layout(
    data, sensor_identifiers, sensors_per_module_list, strings_count,
    custom_sensor_order, inlet_temp, outlet_temp, flow, vmin, vmax, mf4_path=None
):
    data_length = data.shape[1]
    if not inlet_temp: inlet_temp = [np.nan] * data_length
    if not outlet_temp: outlet_temp = [np.nan] * data_length
    if not flow: flow = [np.nan] * data_length

    min_length = min(data_length, len(inlet_temp), len(outlet_temp), len(flow))
    data = data[:, :min_length]
    inlet_temp = inlet_temp[:min_length]
    outlet_temp = outlet_temp[:min_length]
    flow = flow[:min_length]
    total_frames = min_length

    fig = plt.figure(figsize=(15, 10))

    # Layout-Optionsbereich (ersetzt ehemalige Inlet/Outlet/Flow Anzeige)
    layout_options = ["HVB_340_800_L", "HVB_170_400_L", "HVB_065_400_T"]
    selected_layout = [layout_options[0]]
    import json
    from config_utils import load_config

    # Radiobuttons zur Auswahl des Layouts
    ax_radio = plt.axes([0.65, 0.85, 0.1, 0.1], facecolor='lightgray')
    radio_layout = RadioButtons(ax_radio, layout_options)
    radio_layout.set_active(0)

    def update_layout(label):
        print(f"🔄 Lade neues Layout: {label}.json")
        try:
            layout_path = os.path.join("layouts", f"{label}.json")
            with open(layout_path, "r") as f:
                layout_data = json.load(f)
            new_order = [tuple(x) for x in layout_data["sensor_order"]]
            custom_sensor_order[:] = new_order
            update(int(slider.val))
        except Exception as e:
            print(f"Fehler beim Laden des Layouts: {e}")

    radio_layout.on_clicked(update_layout)

    # Button: Gehe zu Frame mit höchster Temperatur
    ax_max_button = plt.axes([0.77, 0.85, 0.15, 0.05])
    button_max_temp = Button(ax_max_button, 'Go to Max Temp')

    def jump_to_max_temp(event):
        max_idx = np.nanargmax(np.nanmax(data, axis=0))
        slider.set_val(max_idx)

    button_max_temp.on_clicked(jump_to_max_temp)
    fig.text(0.95, 0.9, f"Source File: {os.path.basename(mf4_path)}", ha='right', va='top', fontsize=10, color='gray')
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.5])

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_additional = fig.add_subplot(gs[2, :])
    plt.subplots_adjust(hspace=0.01, wspace=0.3, top=0.80)
    cbar_list = [None] * strings_count

    suptitle_text_obj, subtitle_text_middle_obj = None, None

    ax_slider = plt.axes([0.20, 0.02, 0.50, 0.04])
    slider = Slider(ax_slider, 'Time [s]', 0, total_frames - 1, valinit=0, valstep=1)

    ax_button_play = plt.axes([0.05, 0.02, 0.1, 0.04])
    button_play = Button(ax_button_play, 'Play/Pause')
    ax_button_rw = plt.axes([0.78, 0.02, 0.1, 0.04])
    button_rw = Button(ax_button_rw, 'Rewind')
    ax_button_ff = plt.axes([0.89, 0.02, 0.1, 0.04])
    button_ff = Button(ax_button_ff, 'Fast Forward')

    # Auswahlfeld für Layout-Konfigurationen
    ax_radio = plt.axes([0.01, 0.65, 0.15, 0.2], facecolor='lightgray')
    radio = RadioButtons(ax_radio, ("HVB_340_800_L", "HVB_170_400_L", "HVB_065_400_T"), active=0)

    playing = [False]
    time_axis = np.arange(total_frames)
    overall_max_temps = np.nanmax(data, axis=0)
    overall_min_temps = np.nanmin(data, axis=0)
    overall_temp_range_over_time = overall_max_temps - overall_min_temps

    layer_mean_temps = np.zeros((strings_count, total_frames))
    for layer in range(strings_count):
        start_idx = sum([sensors_per_module_list[i] * 4 * 4 for i in range(layer)])
        end_idx = start_idx + sensors_per_module_list[layer] * 4 * 4
        layer_mean_temps[layer, :] = np.nanmean(data[start_idx:end_idx, :], axis=0)

    max_mean_layer_temps = np.nanmax(layer_mean_temps, axis=0)
    min_mean_layer_temps = np.nanmin(layer_mean_temps, axis=0)
    range_mean_layer_temps = max_mean_layer_temps - min_mean_layer_temps

    line_overall, = ax_additional.plot([], [], label='Cell Temp Range', color='black')
    line_layer_mean_range, = ax_additional.plot([], [], label='Layer Mean Temp Range', color='red')
    ax_additional.set_xlabel('Time [s]')
    ax_additional.set_ylabel('Temperature Range [°C]')
    ax_additional.set_title('Temperature Ranges Over Time')
    ax_additional.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax_additional.set_xlim(time_axis[0], time_axis[-1])
    ax_additional.set_ylim(0, max(np.nanmax(overall_temp_range_over_time), np.nanmax(range_mean_layer_temps)) * 1.1)

    def update(val):
        nonlocal suptitle_text_obj, subtitle_text_middle_obj
        t_index = int(slider.val)
        heatmap = plot_battery_layout(data, sensor_identifiers, sensors_per_module_list, strings_count, t_index, total_frames, axes, cbar_list, custom_sensor_order, vmin, vmax, fig)

        overall_mean_temp = np.nanmean(data[:, t_index])
        overall_max_temp = np.nanmax(data[:, t_index])
        overall_min_temp = np.nanmin(data[:, t_index])
        overall_temp_range = overall_max_temp - overall_min_temp
        overall_std_dev = np.nanstd(data[:, t_index])

        inlet_display = f"{inlet_temp[t_index]:.2f} °C" if not np.isnan(inlet_temp[t_index]) else 'N/A'
        outlet_display = f"{outlet_temp[t_index]:.2f} °C" if not np.isnan(outlet_temp[t_index]) else 'N/A'
        flow_value = flow[t_index] if not np.isnan(flow[t_index]) else 0
        flow_display = f"{flow_value:.2f} L/min" if flow_value else 'N/A'
        flow_m3_s = flow_value / 60000 if flow_value else 0

        heat_flux = calculation_heat_flux(flow_m3_s, inlet_temp[t_index], outlet_temp[t_index]) if flow_value and inlet_display != 'N/A' and outlet_display != 'N/A' else 0
        heat_flow_display = f"Q_HVB: {heat_flux:.2f} W" if heat_flux else "Q_HVB: N/A"

        line_overall.set_data(time_axis[:t_index + 1], overall_temp_range_over_time[:t_index + 1])
        line_layer_mean_range.set_data(time_axis[:t_index + 1], range_mean_layer_temps[:t_index + 1])
        ax_additional.set_xlim(time_axis[0], time_axis[-1])

        suptitle_text = (
            f"Module Mean Temp: {overall_mean_temp:.2f}°C\n"
            f"Module Max Temp: {overall_max_temp:.2f}°C\n"
            f"Module Min Temp: {overall_min_temp:.2f}°C\n"
            f"Cell Range: {overall_temp_range:.2f}°C\n"
            f"Layer Range: {range_mean_layer_temps[t_index]:.2f}°C\n"
            f"Std Dev: {overall_std_dev:.2f}°C\n"
        )

        if suptitle_text_obj is None:
            suptitle_text_obj = fig.text(0.03, 0.80, suptitle_text, fontsize=12, fontweight='bold', ha='left')
        else:
            suptitle_text_obj.set_text(suptitle_text)

        subtitle_text_middle = (
            f"Inlet Temp: {inlet_display}\n"
            f"Outlet Temp: {outlet_display}\n"
            f"Coolant Flow: {flow_display}\n"
            f"{heat_flow_display}"
        )


        if subtitle_text_middle_obj is None:
            subtitle_text_middle_obj = fig.text(0.5, 0.86, subtitle_text_middle, fontsize=12, fontweight='bold', ha='center')
        else:
            subtitle_text_middle_obj.set_text(subtitle_text_middle)

        fig.canvas.draw_idle()
        if cbar_list[0] is None:
            cbar_ax = fig.add_axes([0.92, 0.33, 0.02, 0.4])
            colorbar = fig.colorbar(heatmap, cax=cbar_ax)
            colorbar.set_label("Temperature [°C]", fontsize=12)
            cbar_list[0] = True

    def load_sensor_order_from_json(layout_name):
        json_files = {
            "HVB_340_800_L": "layouts/HVB_340_800_L.json",
            "HVB_170_400_L": "layouts/HVB_170_400_L.json",
            "HVB_065_400_T": "layouts/HVB_065_400_T.json"
        }
        selected_file = json_files.get(layout_name)
        if selected_file and os.path.exists(selected_file):
            with open(selected_file, 'r') as f:
                config = json.load(f)
            return config["sensor_order"]
        else:
            print(f"⚠️ Layout file not found: {selected_file}")
            return custom_sensor_order

    def on_layout_change(label):
        nonlocal custom_sensor_order
        custom_sensor_order = load_sensor_order_from_json(label)
        update(slider.val)

    radio.on_clicked(on_layout_change)
    slider.on_changed(update)
    button_play.on_clicked(lambda event: playing.__setitem__(0, not playing[0]))
    button_ff.on_clicked(lambda event: slider.set_val(min(total_frames - 1, slider.val + 5)))
    button_rw.on_clicked(lambda event: slider.set_val(max(0, slider.val - 5)))

    def animate(i):
        if playing[0]:
            if slider.val < total_frames - 1:
                slider.set_val(slider.val + 1)
            else:
                slider.set_val(0)

    anim = FuncAnimation(fig, animate, interval=200)
    update(0)
    plt.show()
