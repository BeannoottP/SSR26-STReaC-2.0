from pathlib import Path
import neruon as nrn

def subfolders_from_folder(root):
    matches = []

    for directory in Path(root).rglob("*"):
        if not directory.is_dir():
            continue
        names = {f.name for f in directory.iterdir() if f.is_file()}

        if {"light_on.txt", "spikes.txt", "meta_data.txt"}.issubset(names):
            matches.append(str(directory))
    
    return matches


def load_neurons_from_path(data_path):
    neuron_paths = subfolders_from_folder(data_path)
    neurons = []
    for path in neuron_paths:
        neurons.append(nrn.neuron(path))
    print(str(len(neurons)) + " neurons loaded from " + data_path)
    return neurons