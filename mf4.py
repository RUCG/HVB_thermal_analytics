from asammdf import MDF

file_path = "raw_data/Trigger1Converted.mf4"
mdf = MDF(file_path)

# Liste für alle Signalnamen
signal_names = []

# Iteration über alle Signale
for channel in mdf.iter_channels():
    signal_names.append(channel.name)

# Duplikate entfernen, alphabetisch sortieren
unique_signals = sorted(set(signal_names))


with open("unique_signal_names.txt", "w") as f:
    for name in unique_signals:
        print(name)
        f.write(f"{name}\n")
