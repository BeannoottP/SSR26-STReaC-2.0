"""
file_systems.py

Provides utility related to navigating file systems.

Author: Bennett Ptak (2026)
"""

from pathlib import Path
import neruon as nrn

def subfolders_from_folder(root):
    '''
    Returns string of all sub folders in root file that contain data files
    '''
    matches = []
    #iteractes through recursive list of all directories and files in folder
    for directory in Path(root).rglob("*"):
        #if its a file we dont care
        if not directory.is_dir():
            continue
        #generates list of file names in folder
        names = {f.name for f in directory.iterdir() if f.is_file()}

        #if file names match neccessary files, add directory string to final list
        if {"light_on.txt", "spikes.txt", "meta_data.txt"}.issubset(names):
            matches.append(str(directory))
    
    return matches


def load_neurons_from_path(data_path):
    '''
    Returns all neuron data in data_path folder in list of neurons
    '''
    #helper to find all neuron data folders
    neuron_paths = subfolders_from_folder(data_path)
    neurons = []
    #iterates neuron paths and creates a neruon instance for each
    for path in neuron_paths:
        neurons.append(nrn.neuron(path))
    #logs for sanity
    print(str(len(neurons)) + " neurons loaded from " + data_path)
    return neurons