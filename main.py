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


folder_path = "data/d1_msns/pre_processed_data/6-OHDA_D1-MSNs_cell_body_stim"
neuron_list = f.load_neurons_from_path(folder_path)
for neuron in neuron_list:
    neuron.generate_trial_average_baseline()
    neuron.generate_trial_average_difference(df.absolute_difference)

neurons_df = analysis.neurons_to_dataframe(neuron_list)
analysis.z_score_difference_cols(neurons_df)

neurons_df.to_csv("example_data/test.csv", index=False)