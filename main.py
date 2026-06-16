import neruon as nrn
import feature_space as ft_sp
from matplotlib import pyplot as plt
import poisson_surprise
import numpy as np
folder_path = "data/Naive_mice_hsyn-ChR2_in_GPe/Neuron_0089"

example_neuron = nrn.neuron(folder_path)

"""
#synthetic bursting
spike_time = []
time = 0.0
for i in range(11):
    time += 0.3
    spike_time.append(time)
    for j in range(4):
        time += 0.03
        spike_time.append(time)
    

plt.eventplot(spike_time)
plt.show()
print(poisson_surprise.run_poisson_surprise(np.array(spike_time), surprise_threshold=3.0))
"""
spikeslist = []
burstslist = []
for trial_num in range(5):
    example_space = ft_sp.feature_space(example_neuron.baseline_spike_trains[trial_num])
    example_space.load_default_features()
    example_space.evaluate_features()
    print(example_space)
    spikeslist.append(example_neuron.baseline_spike_trains[trial_num].spike_times)
    bursttimes = []
    for burst in example_neuron.baseline_spike_trains[trial_num].bursts:
        bursttimes.extend(burst[1])
    burstslist.append(bursttimes)
    

plt.eventplot(spikeslist)
plt.eventplot(burstslist, colors="red")
plt.show()