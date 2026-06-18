import neruon as nrn
import feature_space as ft_sp
from matplotlib import pyplot as plt
import poisson_surprise
import numpy as np
import helpers
import file_system as f
import feature as ft
import difference_methods as df


folder_path = "data/d1_msns/pre_processed_data/6-OHDA_D1-MSNs_cell_body_stim"
neuron_list = f.load_neurons_from_path(folder_path)
neuron = neuron_list[0]
neuron.generate_trial_average_baseline()
neuron.generate_trial_average_difference(df.absolute_difference)

print("Baseline " + str(neuron_list[0].trial_average_baseline))
print("Difference " + str(neuron_list[0].trial_average_difference))
#helpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.fr, color="b",)
#helpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.cv, color="b", range=(0,3))
#elpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.percent_spike_bursting, color="b", range= (0,1))
#older_path = "data/d1_msns/pre_processed_data/Naive_D1-MSNs_cell_body_stim"
#euron_list = f.load_neurons_from_path(folder_path)
#elpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.fr, color="red",)
#elpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.cv, color="red", range=(0,3))
#elpers.plot_histogram_from_feature([n.pre_stimulation_spike_train for n in neuron_list], ft.percent_spike_bursting, color="red", range= (0,1))

input("")