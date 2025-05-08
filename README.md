# HVB Thermal Analytics

Ein Python-Tool zur Analyse und Visualisierung von Temperaturverteilungen in Hochvoltbatterien auf Basis von MF4-Dateien (ASAM MDF v4) und optionalen Sensordaten (Inlet, Outlet, Volumenstrom) zur Berechnung des Wärmestroms

## 🚀 Features

- Liest `moduleTemperatureXX_BMSYY`-Kanäle aus `.mf4`-Dateien
- Visualisiert Zelltemperaturen auf Coolingplate-Hintergrund
- Animierte Zeitnavigation mit Heatmap und Sensorbeschriftung
- Berechnung und Anzeige von Temperaturstatistiken

## 🔧 Setup

### 1. Repository klonen

```bash
git clone https://github.com/deinbenutzername/HVB_thermal_analytics.git
cd HVB_thermal_analytics
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```


## ▶️ Ausführung
```bash
python thermal_dynamics_HVB.py
```

## 🗂️ Struktur
```bash
HVB_thermal_analytics
│   .gitignore
│   coolingplate_edited.png
│   README.md
│   requirements.txt
│   thermal_dynamics_HVB.py
│
├───helper_scripts
│       mf4.py
│       signal_analyser.py
│
└───raw_data
        PutHereYourMF4.txt
```
