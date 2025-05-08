import re
from collections import defaultdict
from asammdf import MDF

# Datei laden
file_path = "raw_data/Trigger1Converted.mf4"
mdf = MDF(file_path)

# Signalnamen extrahieren
signal_names = [ch.name for ch in mdf.iter_channels()]

# Nur Signale mit "temp" (Groß-/Kleinschreibung egal)
temp_signals = [name for name in signal_names if "temp" in name.lower()]

# Gruppierung anhand Basisname (z. B. tempSensor_01, tempSensor_02 → tempSensor_)
grouped = defaultdict(list)

for name in temp_signals:
    # Extrahiere alles bis zur letzten Nummer (z. B. tempSensor_01 → tempSensor_)
    match = re.match(r"(.*?)(\d+)?$", name)
    if match:
        base_name = match.group(1).rstrip("_")  # Basisname ohne Ziffern
        grouped[base_name].append(name)

# Ausgabe
print(f"\n📈 Gefundene Temperatur-Signale: {len(temp_signals)}")
print(f"🧩 Gruppierte Sensorfamilien:\n")

for group, members in grouped.items():
    print(f"- {group} → {len(members)} Signale")
    for m in sorted(members):
        print(f"    {m}")
