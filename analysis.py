import pandas as pd
import numpy as np
import re
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def _sanitize_col(name: str) -> str:
	'''
	adapted from geeks4geeks implementation
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
	z_score_df = neuron_df[col_names].apply(zscore) #mask only cols and apply zscore
	#relabel cols
	z_score_df.columns = [f"z_score_{name}" for name in col_names]
	neuron_df[z_score_df.columns] = z_score_df
	
def z_score_baseline_cols(neuron_df):
	col_names = neuron_df.filter(regex=("^baseline_.*")).columns.to_list() #filter column list for only columns labeled diff_
	z_score_df = neuron_df[col_names].apply(zscore) #mask only cols and apply zscore
	#relabel cols
	z_score_df.columns = [f"filtering_z_score_{name}" for name in col_names]
	neuron_df[z_score_df.columns] = z_score_df

def remove_extraneous(neuron_df, z_score_threshold = 3.0, regex = "^filtering_z_score_.*"):
	'''
	drops all rows with filtering_z_scores outside of z_score_threshold, inplace
	'''
	col_names = neuron_df.filter(regex = (regex)).columns.to_list()
	mask = (neuron_df[col_names].abs() <= z_score_threshold).all(axis=1)
	neuron_df.drop(index = neuron_df.index[~mask], inplace = True)
	
def apply_PCA(neuron_df, num_features, regex = "^z_score_.*"):
	col_names = neuron_df.filter(regex = (regex)).columns.to_list()
	pca = PCA(n_components=num_features)
	reduced = pca.fit_transform(neuron_df[col_names])
	reduced_names = ["PC_1st", "PC_2nd", "PC_3rd", "PC_4th", "PC_5th", "PC_6th", "PC_7th", "PC_8th", "PC_9th", "PC_10th"]
	neuron_df[reduced_names]= reduced
	return pca

def generate_elbow_distortion_vals(neuron_df, max_clusters = 10, regex = "^z_score_.*"):
	col_names = neuron_df.filter(regex = (regex)).columns.to_list()
	X = neuron_df[col_names]
	distortions = []
	K = range(1, max_clusters + 1)

	for k in K:
		kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)
		distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])

	return distortions

def apply_kmeans_clusters(neuron_df, num_clusters, regex = "^z_score_.*"):
	cluster_cols = neuron_df.filter(regex=regex).columns
	X = neuron_df[cluster_cols]

	kmeans = KMeans(n_clusters=num_clusters, random_state=42)
	neuron_df["cluster"] = kmeans.fit_predict(X)

def apply_dbscan_clusters(neuron_df, regex = "^z_score_.*", eps = 0.7, min_samples = 10):
	cluster_cols = neuron_df.filter(regex=regex).columns
	X = neuron_df[cluster_cols]

	dbscan = DBSCAN(eps = eps, min_samples = min_samples)
	neuron_df["cluster"] = dbscan.fit_predict(X)

def find_num_clusters(neuron_df, regex= "^z_score_.*"):
	'''
	finds ideal num clusters based on kneedle algorithm in Satop¨a¨a et al as defined in kneedle_alorithm.ipynb
	'''
	# get distortions as a NumPy array for vectorized transformation
	distortion_vals = np.array(generate_elbow_distortion_vals(neuron_df, regex = regex))
	x = np.arange(1, len(distortion_vals) + 1)
	
	# transform the decreasing, convex distortion curve to increasing and concave
	y = distortion_vals.max() - distortion_vals
	
	#creates smoothed distortion_vals as ds_y
	uspline = interp1d(x, y)
	ds_y =  np.array(uspline(x))
	
	def _normalize(a):
		"""return the normalized input array"""
		return (a - min(a)) / (max(a) - min(a))
	
	#normalize x and y
	norm_y = _normalize(ds_y)
	norm_x = _normalize(x)

	#calulate differnce curve
	diff_y = norm_y - norm_x
	diff_x = norm_x

	#find absolute max
	max_index = np.argmax(diff_y)

	return x[max_index]

def find_db_scan_params(neuron_df, regex = "^z_score_.*"):
	'''
	returns eps, min_samples
	based on https://stataiml.com/posts/how_to_set_dbscan_paramter/
	and https://www.reneshbedre.com/blog/dbscan-python.html

	'''
	cluster_cols = neuron_df.filter(regex=regex).columns
	X =  neuron_df[cluster_cols]
	num_cols = X.shape[1]
	num_rows = len(X)

	#always allow for minimum of 4 clusters
	min_samples = min(num_cols * 2, int(num_rows / 4))

	#finding elbow in num neighbors graph
	#get nearest neighbor vals
	nearestNeighbors = NearestNeighbors(n_neighbors=min_samples+1).fit(X)
	dist, _ = nearestNeighbors.kneighbors(X)

	dist = dist[: ,-1]
	dist = np.sort(dist)

	#now reapply kneedle
	y = dist
	x = np.arange(0, len(dist))
	
	#smooth
	uspline = interp1d(x, y)
	ds_y =  np.array(uspline(x))

	#normalize
	def _normalize(a):
		"""return the normalized input array"""
		return (a - min(a)) / (max(a) - min(a))
	
	norm_y = _normalize(ds_y)
	norm_x = _normalize(x)

	#calulate differnce curve
	diff_y = norm_y - norm_x
	diff_x = norm_x

	#calculate eps
	max_index = np.argmax(diff_y)
	eps = dist[max_index]

	return eps, min_samples


def generate_cluster_differences(neuron_df, cluster_num):
	'''
	Return a two-column DataFrame for all z_score_diff feature averages and
	cluster-specific averages for the given cluster number.
	'''
	if "cluster" not in neuron_df.columns:
		raise ValueError("DataFrame must contain a 'cluster' column")

	col_names = neuron_df.filter(regex=r"^z_score_diff_.*").columns.to_list()
	if not col_names:
		return pd.DataFrame(columns=["all_average", "cluster_average"])

	all_mean = neuron_df[col_names].mean(axis=0)
	cluster_mask = neuron_df["cluster"] == cluster_num
	cluster_mean = neuron_df.loc[cluster_mask, col_names].mean(axis=0)

	result = pd.DataFrame({
		"all_average": all_mean,
		"cluster_average": cluster_mean,
	})
	print(result.head)
	return result







	
