import neruon as nrn
import feature_space as ft_sp
from matplotlib import pyplot as plt
import poisson_surprise
import numpy as np
import helpers
import file_system as f
import feature as ft
import difference_methods as df
import analysis


folder_path = "data/Split/Diseased"
d_neuron_list = f.load_neurons_from_path(folder_path)
folder_path = "data/Split/Healthy"
h_neuron_list = f.load_neurons_from_path(folder_path)

for z_score, z_score_label in zip((True, False), ("with Z Score", "")):
    for difference, difference_label in zip((df.absolute_difference, df.modulation_factor, df.squared_difference, df.relative_difference), ("Absolute Difference", "Modulation Factor", "Squared Difference", "Relative Difference")):
        for per_trial, per_trial_label in zip((True, False), ("Per Trial", "Trial Average")):
            title = per_trial_label + " " + difference_label + " " + z_score_label
            helpers.plot_pca(h_neuron_list, d_neuron_list,  difference_method=difference,z_score=z_score, per_trial=per_trial, title = title, color = "orange")

#neurons_df.to_csv("example_data/test.csv", index=False)