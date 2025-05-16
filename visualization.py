
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_battery_layout(data, sensor_identifiers, sensors_per_module_list, strings_count, t_index, total_frames, axes, cbar_list, custom_sensor_order, vmin=15, vmax=40, title="Battery Temperature Layout", fig=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    background_image_path = os.path.join(current_dir, "coolingplate_edited.png")
    try:
        background_img = plt.imread(background_image_path)
    except FileNotFoundError:
        print(f"Error: Background image not found at {background_image_path}")
        return

    image_height, image_width = background_img.shape[:2]
    white_area_width, white_area_height = 486, 212
    x_start = (image_width - white_area_width + 40) / 2
    x_end = x_start + white_area_width
    y_start = (image_height - white_area_height) / 2
    y_end = y_start + white_area_height
    heatmap_extent = [x_start, x_end, y_start, y_end]

    data_at_timestamp = data[:, t_index]
    reordered_layers = []
    reordered_sensor_numbers_layers = []
    mean_temperatures, max_temperatures, min_temperatures = [], [], []
    temperature_ranges, std_devs = [], []

    for layer in range(strings_count):
        sensors_per_module = sensors_per_module_list[layer]
        total_sensors_per_layer = 4 * sensors_per_module * 4
        reordered_data_layer = np.full((4, 4 * sensors_per_module), np.nan)
        reordered_sensor_numbers_layer = np.full((4, 4 * sensors_per_module), None, dtype=object)
        start_index = sum([4 * sensors_per_module_list[i] * 4 for i in range(layer)])
        end_index = start_index + total_sensors_per_layer
        layer_sensor_indices = custom_sensor_order[start_index:end_index]

        for i in range(4):
            for j in range(4 * sensors_per_module):
                local_index = i * (4 * sensors_per_module) + j
                if local_index < len(layer_sensor_indices):
                    wanted_id = layer_sensor_indices[local_index]
                    try:
                        idx = next(k for k, (id_pair, _) in enumerate(sensor_identifiers) if id_pair == wanted_id)
                        reordered_data_layer[i, j] = data_at_timestamp[idx]
                        reordered_sensor_numbers_layer[i, j] = sensor_identifiers[idx]
                    except StopIteration:
                        pass

        reordered_layers.append(reordered_data_layer)
        reordered_sensor_numbers_layers.append(reordered_sensor_numbers_layer)

        mean_temperatures.append(np.nanmean(reordered_data_layer))
        max_temperatures.append(np.nanmax(reordered_data_layer))
        min_temperatures.append(np.nanmin(reordered_data_layer))
        temperature_ranges.append(max_temperatures[-1] - min_temperatures[-1])
        std_devs.append(np.nanstd(reordered_data_layer))

    for string_index in range(strings_count):
        ax = axes[string_index]
        ax.clear()
        ax.imshow(background_img, extent=[0, image_width, 0, image_height], aspect='auto', origin='lower', zorder=0)
        heatmap = ax.imshow(reordered_layers[string_index], cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax, extent=heatmap_extent, alpha=1, origin='lower', zorder=1)
        ax.set_title(f'Layer {string_index + 1}\nMean: {mean_temperatures[string_index]:.2f}°C | Max: {max_temperatures[string_index]:.2f}°C | Min: {min_temperatures[string_index]:.2f}°C\nRange: {temperature_ranges[string_index]:.2f}°C | Std Dev: {std_devs[string_index]:.2f}°C', fontsize=10, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_aspect('equal')

        cell_width = (x_end - x_start) / (4 * sensors_per_module_list[string_index])
        cell_height = (y_end - y_start) / 4

        for i in range(4):
            for j in range(4 * sensors_per_module_list[string_index]):
                temp = reordered_layers[string_index][i, j]
                sensor_info = reordered_sensor_numbers_layers[string_index][i, j]
                if sensor_info is None or np.isnan(temp):
                    continue
                annotation_x = x_start + (j + 0.5) * cell_width
                annotation_y = y_start + (i + 0.5) * cell_height
                (id_pair, _) = sensor_info
                sensor_num, bms_id = id_pair
                ax.text(annotation_x, annotation_y, f"Temp {sensor_num}\nBMS {bms_id}\n{temp:.1f}°C", ha="center", va="center", fontsize=7, color="black", zorder=2)

    return heatmap
