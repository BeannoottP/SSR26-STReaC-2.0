from matplotlib import pyplot as plt
import neruon as nrn
import feature as ft

def plot_bursts(neuron):
    spikeslist = []
    burstslist = []
    for trial_num in range(5):
        spikeslist.append(neuron.baseline_spike_trains[trial_num].spike_times)
        bursttimes = []
        for burst in neuron.baseline_spike_trains[trial_num].bursts:
            bursttimes.extend(burst[1])
        burstslist.append(bursttimes)

    plt.eventplot(spikeslist)
    plt.eventplot(burstslist, colors="red")
    plt.show()

def plot_histogram_from_feature(spike_trains, feature, color = "blue", range = (0,100)):
    values = []
    for train in spike_trains:
        values.append(train.get_feature(feature).value)
    ref_ft = spike_trains[0].get_feature(feature)

    plt.figure()
    plt.hist(values, bins=20, range=range, edgecolor= "black", color= color)
    plt.xlabel(ref_ft.name)
    plt.ylabel("Probability")
    plt.title(ref_ft.name + " Histogram")
    plt.show(block=False)
    
