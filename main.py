import neruon as nrn
import feature_space as ft_sp
from matplotlib import pyplot as plt
import poisson_surprise
import numpy as np
import helpers
import file_system as f
import feature as ft


folder_path = "data/d1_msns"
neuron_list = f.load_neurons_from_path(folder_path)
helpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.fr)
folder_path = "data/gpe_pv"
neuron_list = f.load_neurons_from_path(folder_path)
helpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.fr)


input("")