"""interface.py
================
Interaktive Temperatur‑Visualisierung für HV‑Batterie‑Messdaten
----------------------------------------------------------------
Diese Version ignoriert die ehemaligen Inlet/Outlet‑Temperatur‑ sowie
Volumenstrom‑Parameter, behält sie aber in der Funktionssignatur, damit
bestehende Aufrufe (z. B. aus *main.py*) unverändert funktionieren.
Außerdem: Pathlib, persistentes Animation‑Handle und `cache_frame_data=False`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

from visualization import plot_battery_layout

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

LAYOUT_JSONS = {
    "HVB_340_800_L": Path("layouts/HVB_340_800_L.json"),
    "HVB_170_400_L": Path("layouts/HVB_170_400_L.json"),
    "HVB_065_400_T": Path("layouts/HVB_065_400_T.json"),
}


def load_sensor_order_from_json(layout_name: str) -> List[Sequence[int]]:
    """Lädt die Sensor‑Reihenfolge aus der entsprechenden JSON‑Layoutdatei."""
    path = LAYOUT_JSONS.get(layout_name)
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return [tuple(seq) for seq in json.load(f)["sensor_order"]]
    print(f"⚠️  Layout‑JSON nicht gefunden: {path}")
    return []


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def interactive_battery_layout(
    data: np.ndarray,
    sensor_identifiers: Sequence[str],
    sensors_per_module_list: Sequence[int],
    strings_count: int,
    custom_sensor_order: List[Sequence[int]] | None,
    # legacy‑Parameter – werden ignoriert, damit alter Code weiterhin läuft
    inlet_temp=None,
    outlet_temp=None,
    flow=None,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    mf4_path: str | Path | None = None,
) -> None:
    """Interaktive Heatmap‑Animation der Zelltemperaturen pro Zeitstufe."""

    # ---------------------------------------------------------------------
    # Grunddaten / Defaults
    # ---------------------------------------------------------------------
    data = np.asarray(data)
    total_frames = data.shape[1]
    vmin = float(np.nanmin(data)) if vmin is None else vmin
    vmax = float(np.nanmax(data)) if vmax is None else vmax

    if custom_sensor_order is None:
        custom_sensor_order = load_sensor_order_from_json("HVB_340_800_L")

    # ---------------------------------------------------------------------
    # Figure & Layout
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))
    if mf4_path is not None:
        fig.text(
            0.99,
            0.97,
            f"Source File: {Path(mf4_path).name}",
            ha="right",
            va="top",
            fontsize=9,
            color="gray",
        )

    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_stats = fig.add_subplot(gs[2, :])
    plt.subplots_adjust(hspace=0.02, wspace=0.3, top=0.80)

    # ---------------------------------------------------------------------
    # Widgets
    # ---------------------------------------------------------------------
    ax_slider = plt.axes([0.20, 0.02, 0.55, 0.04])
    slider = Slider(ax_slider, "Time [s]", 0, total_frames - 1, valinit=0, valstep=1)

    ax_play = plt.axes([0.05, 0.02, 0.1, 0.04])
    btn_play = Button(ax_play, "Play/Pause")

    ax_rw = plt.axes([0.82, 0.02, 0.08, 0.04])
    btn_rw = Button(ax_rw, "<- 5s")

    ax_ff = plt.axes([0.91, 0.02, 0.08, 0.04])
    btn_ff = Button(ax_ff, "-> 5s")

    ax_radio = plt.axes([0.65, 0.85, 0.1, 0.1], facecolor='lightgray')
    radio_layout = RadioButtons(ax_radio, list(LAYOUT_JSONS.keys()), active=0)

    ax_max = plt.axes([0.77, 0.85, 0.18, 0.05])
    btn_max = Button(ax_max, "Go to Max Temp")

    # ---------------------------------------------------------------------
    # Zusatz‑Statistik‑Plot
    # ---------------------------------------------------------------------
    time_axis = np.arange(total_frames)
    overall_max = np.nanmax(data, axis=0)
    overall_min = np.nanmin(data, axis=0)
    overall_range = overall_max - overall_min

    layer_mean = np.zeros((strings_count, total_frames))
    for layer in range(strings_count):
        start = sum(sensors_per_module_list[:layer]) * 16
        end = start + sensors_per_module_list[layer] * 16
        layer_mean[layer] = np.nanmean(data[start:end], axis=0)

    layer_range = np.nanmax(layer_mean, axis=0) - np.nanmin(layer_mean, axis=0)

    ln_cell_rng, = ax_stats.plot([], [], label="Cell ΔT", lw=1)
    ln_layer_rng, = ax_stats.plot([], [], label="Layer ΔT", lw=1)
    ax_stats.set_xlabel("Time [s]")
    ax_stats.set_ylabel("ΔT [°C]")
    ax_stats.set_title("Temperature Ranges Over Time")
    ax_stats.set_xlim(0, total_frames - 1)
    ax_stats.set_ylim(0, max(np.nanmax(overall_range), np.nanmax(layer_range)) * 1.1)
    ax_stats.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # ---------------------------------------------------------------------
    # State Variablen
    # ---------------------------------------------------------------------
    playing = False
    cbar_drawn = False
    suptxt = None

    # ---------------------------------------------------------------------
    # Update‑Funktion
    # ---------------------------------------------------------------------

    def _update(frame):  # Slider liefert float
        nonlocal suptxt, cbar_drawn
        idx = int(frame)

        hm = plot_battery_layout(
            data,
            sensor_identifiers,
            sensors_per_module_list,
            strings_count,
            idx,
            total_frames,
            axes,
            [None] * strings_count,
            custom_sensor_order,
            vmin,
            vmax,
            fig,
        )

        ln_cell_rng.set_data(time_axis[: idx + 1], overall_range[: idx + 1])
        ln_layer_rng.set_data(time_axis[: idx + 1], layer_range[: idx + 1])

        mean = np.nanmean(data[:, idx])
        t_max = overall_max[idx]
        t_min = overall_min[idx]
        rng = overall_range[idx]
        rng_layer = layer_range[idx]
        std = np.nanstd(data[:, idx])

        stats = (
            f"Mean: {mean:.2f}°C\n"
            f"Max:  {t_max:.2f}°C\n"
            f"Min:  {t_min:.2f}°C\n"
            f"Cell ΔT:  {rng:.2f}°C\n"
            f"Layer ΔT: {rng_layer:.2f}°C\n"
            f"Std Dev: {std:.2f}°C"
        )
        if suptxt is None:
            suptxt = fig.text(0.03, 0.80, stats, fontsize=12, fontweight='bold', ha='left')
        else:
            suptxt.set_text(stats)

        if not cbar_drawn:
            cax = fig.add_axes([0.92, 0.33, 0.02, 0.4])
            cb = fig.colorbar(hm, cax=cax)
            cb.set_label("Temperature [°C]")
            cbar_drawn = True

        fig.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Widgets verbinden
    # ---------------------------------------------------------------------

    slider.on_changed(_update)

    def _toggle_play(event):
        nonlocal playing
        playing = not playing

    btn_play.on_clicked(_toggle_play)
    btn_rw.on_clicked(lambda _: slider.set_val(max(0, slider.val - 5)))
    btn_ff.on_clicked(lambda _: slider.set_val(min(total_frames - 1, slider.val + 5)))
    btn_max.on_clicked(lambda _: slider.set_val(int(np.nanargmax(overall_max))))

    def _layout_change(label):
        nonlocal custom_sensor_order
        new_order = load_sensor_order_from_json(label)
        if new_order:
            custom_sensor_order = new_order
        _update(slider.val)

    radio_layout.on_clicked(_layout_change)

    # ---------------------------------------------------------------------
    # Animation‑Loop
    # ---------------------------------------------------------------------

    def _animate(_):
        if playing:
            nxt = slider.val + 1
            if nxt > total_frames - 1:
                nxt = 0
            slider.set_val(nxt)

    anim = FuncAnimation(fig, _animate, interval=200, cache_frame_data=False)

    # Initialer Draw --------------------------------------------------------------
    _update(0)
    plt.show()