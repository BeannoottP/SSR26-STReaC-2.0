import neruon as nrn
import feature_space as ft_sp

folder_path = "data/Naive_mice_hsyn-ChR2_in_GPe/Neuron_0105"

example_neuron = nrn.neuron(folder_path)

example_space = ft_sp.feature_space(example_neuron.stimulation_spike_trains[0])
example_space.load_default_features()
example_space.evaluate_features()
print(example_space)