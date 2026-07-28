import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import analysis


def create_iris_pca_figures(output_dir="presentation figures"):
    """Generate a 2x2 PCA demonstration figure from the Iris dataset."""
    os.makedirs(output_dir, exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    loadings = pca.components_.T

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 12
    legend_fontsize = 12
    annotation_fontsize = 10

    # Top-left: raw feature scatter
    ax = axes[0, 0]
    for target_idx, target_name in enumerate(target_names):
        mask = y == target_idx
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            label=target_name,
            alpha=0.8,
            edgecolor="k",
            s=70
        )
    ax.set_xlabel(feature_names[0], fontsize=label_fontsize)
    ax.set_ylabel(feature_names[1], fontsize=label_fontsize)
    ax.set_title("Iris data without PCA", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(title="Species", fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    # Top-right: PCA scatter
    ax = axes[0, 1]
    for target_idx, target_name in enumerate(target_names):
        mask = y == target_idx
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=target_name,
            alpha=0.8,
            edgecolor="k",
            s=70
        )
    ax.set_xlabel("PC 1", fontsize=label_fontsize)
    ax.set_ylabel("PC 2", fontsize=label_fontsize)
    ax.set_title("Iris data with PCA", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(title="Species", fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    # Bottom-left: variance explained
    ax = axes[1, 0]
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained, alpha=0.75, label="Individual", color="#4C72B0")
    ax.plot(x, np.cumsum(explained), marker="o", color="#DD8452", linewidth=2, label="Cumulative")
    ax.set_xlabel("Principal component", fontsize=label_fontsize)
    ax.set_ylabel("Explained variance ratio", fontsize=label_fontsize)
    ax.set_title("Variance explained by PCA components", fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Bottom-right: coefficient matrix heatmap
    ax = axes[1, 1]
    im = ax.imshow(loadings, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(loadings.shape[1]))
    ax.set_xticklabels([f"PC {i+1}" for i in range(loadings.shape[1])], fontsize=tick_fontsize)
    ax.set_yticks(np.arange(loadings.shape[0]))
    ax.set_yticklabels(feature_names, fontsize=tick_fontsize)
    ax.set_title("PCA coefficient matrix (feature loadings)", fontsize=title_fontsize)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label("Loading value", fontsize=label_fontsize)
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center", color="black", fontsize=annotation_fontsize)

    fig.suptitle("Iris PCA demonstration", fontsize=22, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    grid_output_path = os.path.join(output_dir, "iris_pca_presentation_grid.png")
    fig.savefig(grid_output_path, dpi=300)
    plt.close(fig)

    # K-means elbow analysis using analysis helpers
    pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC_0", "PC_1"])
    distortions = analysis.generate_elbow_distortion_vals(pca_df, max_clusters=10, regex="^PC_.*")
    recommended_clusters = analysis.find_num_clusters(pca_df, regex="^PC_.*")
    kmeans_model = analysis.apply_kmeans_clusters(pca_df, num_clusters=recommended_clusters, regex="^PC_.*")

    fig_kmeans, axes_kmeans = plt.subplots(1, 2, figsize=(16, 7))
    fig_kmeans.subplots_adjust(wspace=0.25)

    ax_elbow = axes_kmeans[0]
    k_values = list(range(1, len(distortions) + 1))
    ax_elbow.plot(k_values, distortions, marker="o", linewidth=2, color="#4C72B0")
    ax_elbow.axvline(recommended_clusters, color="#DD8452", linestyle="--", linewidth=2)
    ax_elbow.set_title("K-means elbow plot", fontsize=18)
    ax_elbow.set_xlabel("Number of clusters", fontsize=14)
    ax_elbow.set_ylabel("Distortion", fontsize=14)
    ax_elbow.grid(True, linestyle="--", alpha=0.4)
    ax_elbow.tick_params(labelsize=12)

    ax_scatter = axes_kmeans[1]
    scatter = ax_scatter.scatter(
        pca_df["PC_0"],
        pca_df["PC_1"],
        c=pca_df["cluster"],
        cmap="viridis",
        s=70,
        edgecolor="k"
    )
    ax_scatter.scatter(
        kmeans_model.cluster_centers_[:, 0],
        kmeans_model.cluster_centers_[:, 1],
        marker="X",
        s=250,
        c="red",
        edgecolor="black",
        label="Cluster centers"
    )
    ax_scatter.set_title(f"K-means clusters on first 2 PCs (k={recommended_clusters})", fontsize=18)
    ax_scatter.set_xlabel("PC 1", fontsize=14)
    ax_scatter.set_ylabel("PC 2", fontsize=14)
    ax_scatter.tick_params(labelsize=12)
    ax_scatter.legend(fontsize=12)

    elbow_output_path = os.path.join(output_dir, "iris_kmeans_elbow_and_clusters.png")
    fig_kmeans.tight_layout()
    fig_kmeans.savefig(elbow_output_path, dpi=300)
    plt.close(fig_kmeans)

    return {
        "grid": grid_output_path,
        "kmeans": elbow_output_path,
    }


if __name__ == "__main__":
    saved_files = create_iris_pca_figures()
    print("Saved presentation figures:")
    for path in saved_files.values():
        print(path)

