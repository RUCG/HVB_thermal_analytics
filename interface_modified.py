"""interface.py – Interaktive Batterie‑Visualisierung
----------------------------------------------------
Überarbeitet: • UnboundLocalError behoben • Pathlib • klarere Defaults

Abhängigkeiten: matplotlib, numpy, visualization.py, data_extraction.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

from data_extraction import calculation_heat_flux
from visualization import plot_battery_layout

# ---------------------------------------------------------------------------
# Hilfsfunktion -------------------------------------------------------------
# ---------------------------------------------------------------------------

LAYOUT_DIR = Path(__file__).with_suffix("").parent / "layouts"


def load_sensor_order_from_json(layout_name: str, fallback: Sequence[int] | None = None) -> List[int]:
    """Lädt die Sensor‑Reihenfolge für *layout_name*.

    Gibt *fallback* zurück, falls die JSON‑Datei fehlt oder defekt ist.
    """
    file_map = {
        "HVB_340_800_L": LAYOUT_DIR / "HVB_340_800_L.json",
        "HVB_170_400_L": LAYOUT_DIR / "HVB_170_400_L.json",
        "HVB_065_400_T": LAYOUT_DIR / "HVB_065_400_T.json",
    }
    path = file_map.get(layout_name)

    if path and path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)["sensor_order"]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"⚠️ Fehler beim Einlesen von {path.name}: {exc}")
    else:
        print(f"⚠️ Layout‑Datei nicht gefunden: {path}")

    return list(fallback) if fallback is not None else []


# ---------------------------------------------------------------------------
# Hauptfunktion -------------------------------------------------------------
# ---------------------------------------------------------------------------

def interactive_battery_layout(
    data: np.ndarray,
    sensor_identifiers: Sequence[str],
    sensors_per_module_list: Sequence[int],
    strings_count: int,
    custom_sensor_order: Sequence[int] | None = None,
    inlet_temp: Sequence[float] | None = None,
    outlet_temp: Sequence[float] | None = None,
    flow: Sequence[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    mf4_path: str | os.PathLike | None = None,
) -> None:
    """Zeigt interaktive Heatmaps & Kennzahlen des Batterie‑Layouts an."""

    # ------------------------- Daten vorbereiten --------------------------
    samples = data.shape[1]

    inlet_temp = list(inlet_temp or [np.nan] * samples)
    outlet_temp = list(outlet_temp or [np.nan] * samples)
    flow = list(flow or [np.nan] * samples)

    min_len = min(samples, len(inlet_temp), len(outlet_temp), len(flow))
    if min_len < samples:
        data = data[:, :min_len]
        inlet_temp = inlet_temp[:min_len]
        outlet_temp = outlet_temp[:min_len]
        flow = flow[:min_len]
        samples = min_len

    # Sensor‑Layout laden --------------------------------------------------
    if custom_sensor_order is None:
        custom_sensor_order = load_sensor_order_from_json("HVB_340_800_L")

    # ---------------------------- Figure ----------------------------------
    fig = plt.figure(figsize=(15, 10))
    if mf4_path:
        fig.text(0.965, 0.92, f"Source: {Path(mf4_path).name}", ha="right", va="top", fontsize=9, color="gray")

    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    axes_heat = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_stats = fig.add_subplot(gs[2, :])
    plt.subplots_adjust(hspace=0.03, wspace=0.3, top=0.82)

    # ----------------------- Widgets & Controls ---------------------------
    ax_slider = plt.axes([0.22, 0.02, 0.56, 0.04])
    slider = Slider(ax_slider, "Time [s]", 0, samples - 1, valinit=0, valstep=1)

    # Buttons
    button_cfg = {"color": "lightgray", "hovercolor": "0.85"}
    but_play = Button(plt.axes([0.05, 0.02, 0.1, 0.04]), "Play/Stop", **button_cfg)
    but_rw = Button(plt.axes([0.80, 0.02, 0.08, 0.04]), "⟲ -5", **button_cfg)
    but_ff = Button(plt.axes([0.89, 0.02, 0.08, 0.04]), "+5 ⟳", **button_cfg)

    # Radio‑Buttons für Layout
    r_ax = plt.axes([0.02, 0.45, 0.12, 0.2])
    radio = RadioButtons(r_ax, ["HVB_340_800_L", "HVB_170_400_L", "HVB_065_400_T"])

    # ----------------------- Pre‑Calculations -----------------------------
    t_axis = np.arange(samples)
    cell_max = np.nanmax(data, axis=0)
    cell_min = np.nanmin(data, axis=0)
    cell_range = cell_max - cell_min

    # Layer‑Durchschnittstemperaturen
    layer_means = np.empty((strings_count, samples))
    for layer in range(strings_count):
        start = sum(sensors_per_module_list[:layer]) * 16  # 4×4 Sensoren pro Modul
        end = start + sensors_per_module_list[layer] * 16
        layer_means[layer] = np.nanmean(data[start:end], axis=0)

    layer_range = np.nanmax(layer_means, axis=0) - np.nanmin(layer_means, axis=0)

    # Linien in ax_stats
    (ln_cell_range,) = ax_stats.plot([], [], label="Cell Temp Range")
    (ln_layer_range,) = ax_stats.plot([], [], label="Layer Mean Range")

    ax_stats.set_title("Temperature Ranges Over Time")
    ax_stats.set_xlabel("Time [s]")
    ax_stats.set_ylabel("ΔT [°C]")
    ax_stats.set_xlim(t_axis[0], t_axis[-1])
    ax_stats.set_ylim(0, max(np.nanmax(cell_range), np.nanmax(layer_range)) * 1.1)
    ax_stats.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # ------------------- State vars in Closure ----------------------------
    playing: list[bool] = [False]
    cbar_drawn: list[bool] = [False]
    txt_left = txt_mid = None  # type: ignore

    # ------------------------- Update‑Funktion ----------------------------
    def update(frame: int | float) -> None:
        nonlocal txt_left, txt_mid
        i = int(frame)

        # Heatmaps zeichnen
        hm = plot_battery_layout(
            data,
            sensor_identifiers,
            sensors_per_module_list,
            strings_count,
            i,
            samples,
            axes_heat,
            [None] * strings_count,  # Dummy – Farbbalken global unten
            custom_sensor_order,
            vmin,
            vmax,
            fig,
        )

        # Textlinks Kennwerte
        mean_t = np.nanmean(data[:, i])
        max_t = cell_max[i]
        min_t = cell_min[i]
        std_t = np.nanstd(data[:, i])
        left_txt = (
            f"Mean: {mean_t:.2f}°C\nMax: {max_t:.2f}°C\nMin: {min_t:.2f}°C\n"
            f"Cell ΔT: {cell_range[i]:.2f}°C\nLayer ΔT: {layer_range[i]:.2f}°C\nσ: {std_t:.2f}°C"
        )
        if txt_left is None:
            txt_left = fig.text(0.03, 0.82, left_txt, fontsize=11, fontweight="bold", va="top")
        else:
            txt_left.set_text(left_txt)

        # Textmitte Kühlmittel‑Infos
        inlet = inlet_temp[i]
        outlet = outlet_temp[i]
        flow_l_min = flow[i]
        flow_m3_s = flow_l_min / 60000 if not np.isnan(flow_l_min) else np.nan
        q_hvb = calculation_heat_flux(flow_m3_s, inlet, outlet) if not np.isnan(flow_m3_s) else np.nan
        mid_txt = (
            f"Inlet: {inlet if not np.isnan(inlet) else 'N/A':.2f}°C\n"
            f"Outlet: {outlet if not np.isnan(outlet) else 'N/A':.2f}°C\n"
            f"Flow: {flow_l_min if not np.isnan(flow_l_min) else 'N/A':.2f} L/min\n"
            f"Qₕₐₕ: {q_hvb if not np.isnan(q_hvb) else 'N/A':.2f} W"
        )
        if txt_mid is None:
            txt_mid = fig.text(0.5, 0.86, mid_txt, fontsize=11, fontweight="bold", ha="center", va="top")
        else:
            txt_mid.set_text(mid_txt)

        # Linien aktualisieren
        ln_cell_range.set_data(t_axis[: i + 1], cell_range[: i + 1])
        ln_layer_range.set_data(t_axis[: i + 1], layer_range[: i + 1])

        # Farbskala einmalig
        if not cbar_drawn[0]:
            cbar_ax = fig.add_axes([0.93, 0.26, 0.02, 0.46])
            cb = fig.colorbar(hm, cax=cbar_ax)
            cb.set_label("Temperature [°C]")
            cbar_drawn[0] = True

        fig.canvas.draw_idle()

    # ------------------------- Callbacks ----------------------------------
    def _toggle_play(_):
        playing[0] = not playing[0]

    def _ff(_):
        slider.set_val(min(samples - 1, slider.val + 5))

    def _rw(_):
        slider.set_val(max(0, slider.val - 5))

    def _layout_changed(label: str):
        nonlocal custom_sensor_order
        custom_sensor_order = load_sensor_order_from_json(label, custom_sensor_order)
        update(slider.val)

    # Connect
    but_play.on_clicked(_toggle_play)
    but_ff.on_clicked(_ff)
    but_rw.on_clicked(_rw)
    radio.on_clicked(_layout_changed)
    slider.on_changed(update)

    # Animation -----------------------------------------------------------
    def _animate(_):
        if playing[0]:
            slider.set_val((slider.val + 1) % samples)

    FuncAnimation(fig, _animate, interval=200)
    update(0)
    plt.show()
