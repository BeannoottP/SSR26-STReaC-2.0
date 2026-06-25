from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import neruon as nrn
import feature as ft
import difference_methods as df
import analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

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
        neuron.generate_trial_average_baseline()
        if per_trial:
            neuron.generate_per_trial_difference(difference_method)
        else:
            neuron.generate_trial_average_difference(difference_method)
    
    h_neuron_df = analysis.neurons_to_dataframe(healthy_neuron_list)
    print(len(h_neuron_df))
    d_neuron_df = analysis.neurons_to_dataframe(diseased_neuron_list)



    h_neuron_df["marker"] = "^"
    d_neuron_df["marker"] = "o"
    
    neuron_df = pd.concat([h_neuron_df, d_neuron_df])
    neuron_df = h_neuron_df

    processed_df = pd.concat([
                pd.read_csv("data/Original/d1_msns/processed_data/all_data.csv"),
                pd.read_csv("data/Original/gpe_pv/processed_data/all_data.csv")])
    
    #processed_df.rename(columns={"cell_dir" : "src"}, inplace= True)
    data_dir_list = neuron_df["data_dir"].to_list()
    data_dir_list = ["/".join(x.split("/")[-4:]) for x in data_dir_list]
    neuron_df["data_dir"] = data_dir_list
    
    data_dir_list = processed_df["src"].to_list()
    data_dir_list = ["/".join(x.split("/")[-4:]) for x in data_dir_list]
    processed_df["data_dir"] = data_dir_list
    neuron_df = neuron_df.merge(processed_df[["data_dir", "neural_response_val"]], on="data_dir", how="left")
    print(len(neuron_df))
    #{'complete inhibition': 0, 'adapting inhibition': 1, 'partial inhibition': 2, 'no effect': 3, 'excitation': 4, 'biphasic IE': 5, 'biphasic EI': 6}
    color_list = ["blue", "red", "green", "yellow", "orange", "cyan", "black"]
    color_map = dict(enumerate(color_list))



    neuron_df["color"] = neuron_df["neural_response_val"].map(color_map)
    neuron_df.to_csv("example_data/test.csv", index=False)

    analysis.z_score_difference_cols(neuron_df)
    analysis.z_score_baseline_cols(neuron_df)
    analysis.remove_extraneous(neuron_df, regex = "^filtering_.*")

    if z_score:
        regex = "^z_score_.*"

    else:
        regex =  "^diff_.*"
    
    pca = analysis.apply_PCA(neuron_df, 3, regex=regex)
    neuron_df = neuron_df.dropna()

    fig = plt.figure(figsize=(18,12), constrained_layout= True)
    gs = fig.add_gridspec(2,3, width_ratios=[6,2,4])
    
    ax_pca = fig.add_subplot(gs[0], projection='3d')
    #ax_pca.scatter(xs = neuron_df["PC_1st"].tolist(), ys = neuron_df["PC_2nd"].tolist(), zs = neuron_df["PC_3rd"].tolist(), marker=neuron_df["marker"].to_list(), c= neuron_df["color"].to_list()) # type: ignore
    for marker, group in neuron_df.groupby("marker"):
        ax_pca.scatter(
            group["PC_1st"],
            group["PC_2nd"],
            group["PC_3rd"], #type: ignore
            marker=marker,
            c=group["color"]
        ) 
    ax_pca.set_title("PCA Features")
    ax_pca.set_xlabel("PCA1")
    ax_pca.set_ylabel("PCA2")
    ax_pca.set_zlabel("PCA3")
    ax_pca.set_box_aspect((1, 1, 1))

    ax_scree = fig.add_subplot(gs[1])
    ax_scree.bar([f'PCA{x}' for x in range(1,pca.n_components_+1)], pca.explained_variance_ratio_, color = color)
    ax_scree.plot([f'PCA{x}' for x in range(1,pca.n_components_+1)], np.cumsum(pca.explained_variance_ratio_), color = color)
    ax_scree.set_xlabel("PCA Feature")
    ax_scree.set_ylabel("Explained Varience (%)")
    ax_scree.set_title("Varience Explained Bar Chart")
    ax_scree.grid()

    ax_loadings = fig.add_subplot(gs[2])
    col_names = neuron_df.filter(regex=regex).columns.to_list()
    feature_names = []
    for name in col_names:
        if name.startswith("z_score_diff_"):
            feature_names.append(name[len("z_score_diff_"):])
        elif name.startswith("diff_"):
            feature_names.append(name[len("diff_"):])
        else:
            feature_names.append(name)

    max_label_len = 21
    xticklabels = [
        fname if len(fname) <= max_label_len else f"{fname[:max_label_len-3]}..."
        for fname in feature_names
    ]
    sns.heatmap(
        pca.components_,
        cmap='coolwarm',
        vmax= 0.75,
        vmin = -0.75,
        yticklabels=[f'PCA{x}' for x in range(1, pca.n_components_ + 1)],
        xticklabels=xticklabels,
        linewidths=1,
        annot=True,
        fmt=',.2f',
        cbar_kws={"shrink": 0.8, "orientation": 'vertical'},
        ax=ax_loadings
    )

    ax_elbow = fig.add_subplot(gs[4])
    distortions = analysis.generate_elbow_distortion_vals(neuron_df)
    ax_elbow.plot(range(1,10), distortions, 'bx-')
    ax_elbow.set_xlabel('Number of Clusters (k)')
    ax_elbow.set_ylabel('Distortion')
    ax_elbow.set_title('The Elbow Method using Distortion')
    
    ax_clusters = fig.add_subplot(gs[3], projection="3d")
    analysis.apply_clusters(neuron_df, 4)

    neuron_df["color"] = neuron_df["cluster"].map(color_map)
    for marker, group in neuron_df.groupby("marker"):
        ax_clusters.scatter(
            group["PC_1st"],
            group["PC_2nd"],
            group["PC_3rd"], #type: ignore
            marker=marker,
            c=group["color"]
        ) 


    fig.suptitle(title)
    fig.savefig("figures/{title}".format(title=title))
    return fig