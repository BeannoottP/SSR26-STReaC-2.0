from matplotlib import pyplot as plt
import neruon as nrn
import feature as ft
import difference_methods as df
import analysis
import pandas as pd

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

def plot_pca(healthy_neuron_list, diseased_neuron_list, difference_method, z_score=True, per_trial = False, color = "blue", title = "The Author of This Plot is Lazy and Did not Provide a Title"):
    for neuron in diseased_neuron_list + healthy_neuron_list:
        if per_trial:
            neuron.generate_per_trial_difference(difference_method)
        else:
            neuron.generate_trial_average_baseline()
            neuron.generate_trial_average_difference(difference_method)
    
    h_neuron_df = analysis.neurons_to_dataframe(healthy_neuron_list)
    d_neuron_df = analysis.neurons_to_dataframe(diseased_neuron_list)
    h_neuron_df["color"] = color
    d_neuron_df["color"] = "blue"
    
    neuron_df = pd.concat([h_neuron_df, d_neuron_df])

    if z_score:
        regex = "^z_score_.*"
        analysis.z_score_difference_cols(neuron_df)
        analysis.remove_extraneous(neuron_df)
    else:
        regex =  "^diff_.*"
    
    pca = analysis.apply_PCA(neuron_df, 3, regex=regex)
    fig = plt.figure(figsize=(12,6), constrained_layout= True)
    gs = fig.add_gridspec(1,2, width_ratios=[4,1])
    
    ax_pca = fig.add_subplot(gs[0], projection='3d')
    ax_pca.scatter(xs = neuron_df["PC_1st"].tolist(), ys = neuron_df["PC_2nd"].tolist(), zs = neuron_df["PC_3rd"].tolist(), marker="^", color = neuron_df["color"]) # type: ignore
    ax_pca.set_title("PCA Features")
    ax_pca.set_box_aspect((1, 1, 1))

    ax_scree = fig.add_subplot(gs[1])
    ax_scree.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, color = color)
    ax_scree.set_xlabel("PCA Feature")
    ax_scree.set_ylabel("Explained Varience")
    ax_scree.set_title("Varience Explained Bar Chart")

    fig.suptitle(title)
    fig.savefig("figures/{title}".format(title=title))
    return fig