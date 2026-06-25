import pandas as pd
import numpy as np
import re
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

def _sanitize_col(name: str) -> str:
	'''
	adapted from
	'''
	name = name.strip().lower()
	name = re.sub(r"\s+", "_", name)
	name = re.sub(r"[^0-9a-zA-Z_]+", "", name)
	return name


def neurons_to_dataframe(neuron_list):
	
	rows = []

    #create row for each neuron
	for neuron in neuron_list:
		row = {}
		row["data_dir"] = neuron.src
		# include metadata
		for k, v in neuron.meta_data.items():
			row[k] = v

		# dump a feature_space into the row dict
		def _dump_space(space, prefix):
			if space is None:
				return
			for feat in space.feature_list:
				col = f"{prefix}_{_sanitize_col(feat.name)}"
				row[col] = feat.value

		# pre-stimulation features
		_dump_space(neuron.pre_stimulation_spike_train.feature_space, "pre")
		# trial-averaged baseline
		_dump_space(neuron.trial_average_baseline, "baseline")
		# trial-averaged difference
		_dump_space(neuron.averaged_difference, "diff")

		rows.append(row)

	df = pd.DataFrame(rows)
	return df


def z_score_difference_cols(neuron_df):
	col_names = neuron_df.filter(regex=("^diff_.*")).columns.to_list() #filter column list for only columns labeled diff_
	print(col_names)
	z_score_df = neuron_df[col_names].apply(zscore) #mask only cols and apply zscore
	#relabel cols
	z_score_df.columns = [f"z_score_{name}" for name in col_names]
	neuron_df[z_score_df.columns] = z_score_df
	
def z_score_baseline_cols(neuron_df):
	col_names = neuron_df.filter(regex=("^baseline_.*")).columns.to_list() #filter column list for only columns labeled diff_
	print(col_names)
	z_score_df = neuron_df[col_names].apply(zscore) #mask only cols and apply zscore
	#relabel cols
	z_score_df.columns = [f"filtering_z_scores_{name}" for name in col_names]
	neuron_df[z_score_df.columns] = z_score_df

def remove_extraneous(neuron_df, z_score_threshold = 3.0, regex = "^z_score_.*"):
	'''
	drops all rows with z_scores outside of z_score_threshold, inplace
	'''
	col_names = neuron_df.filter(regex = (regex)).columns.to_list()
	mask = (neuron_df[col_names].abs() <= z_score_threshold).all(axis=1)
	#print(len(neuron_df))
	neuron_df.drop(index = neuron_df.index[~mask], inplace = True)
	#print(len(neuron_df))
	
def apply_PCA(neuron_df, num_features, regex = "^z_score_.*"):
	col_names = neuron_df.filter(regex = (regex)).columns.to_list()
	pca = PCA(n_components=num_features)
	reduced = pca.fit_transform(neuron_df[col_names])
	reduced_names = ["PC_1st", "PC_2nd", "PC_3rd"]
	neuron_df[reduced_names]= reduced
	return pca

def generate_elbow_distortion_vals(neuron_df, regex = "^z_score_.*"):
	col_names = neuron_df.filter(regex = (regex)).columns.to_list()
	X = neuron_df[col_names]
	distortions = []
	K = range(1, 10)

	for k in K:
		kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
		distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])
	return distortions

def apply_clusters(neuron_df, num_clusters, regex = "^z_score_.*"):
	cluster_cols = neuron_df.filter(regex=regex).columns
	X = neuron_df[cluster_cols]

	kmeans = KMeans(n_clusters=num_clusters, random_state=42)
	neuron_df["cluster"] = kmeans.fit_predict(X)

    


	
