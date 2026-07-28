#imports
import argparse
import inspect
import numpy as np
import analysis
import file_system as fs
import difference_methods as dm
import feature as ft
import pandas as pd

#############################################
#PARSE ARGS
#############################################
parser = argparse.ArgumentParser(description="STReaC 2.0 - Spike Train analysis")
parser.add_argument(
    "--main-data-path", 
    type=str,
    required=True,
    help = "path to main/training dataset")
parser.add_argument(
    "--bucket-data-paths",
    type=str,
    nargs="+",
    required=False,
    help="List of data paths to buckets"
)
parser.add_argument(
    "--difference-method",
    type=str,
    required=True,
    help="Name of the difference method function (e.g., absolute_difference, modulation_factor, squared_difference, relative_difference)"
)
parser.add_argument(
    "--trial-average",
    action="store_true",
    dest="trial_average",
    help="If set, use trial-average mode (default: False)"
)
parser.add_argument(
    "--feature-names",
    type=str,
    nargs="*",
    default=None,
    help="Optional list of feature names for custom feature space"
)
parser.add_argument(
    "--no-pca",
    action="store_true",
    help="Disable PCA when set (default: use PCA)"
)
parser.add_argument(
    "--pca-threshold",
    type=float,
    default=0.8,
    help="Float value 0.0-1.0 for required variance explained in PCA values"
)
parser.add_argument(
    "--outlier-thresholds",
    type=str,
    nargs="*",
    default=None,
    help = "Optionial outlier threshold values per feature in form of --outlier-thresholds feat_name float feat_name float ..."
)

args = parser.parse_args()

# Validate and retrieve the difference method function
if not hasattr(dm, args.difference_method):
    raise ValueError(f"Difference method '{args.difference_method}' not found in difference_methods.py")
difference_method = getattr(dm, args.difference_method)

# Set custom_features flag based on whether feature names are provided
custom_features = args.feature_names is not None and len(args.feature_names) > 0

# Save parsed args into clearly named variables
main_data_path = args.main_data_path
bucket_data_paths = args.bucket_data_paths or []
difference_method_name = args.difference_method
difference_method_fn = difference_method
trial_average = args.trial_average
feature_names = args.feature_names
custom_features = feature_names is not None and len(feature_names) > 0
use_pca = not args.no_pca
pca_threshold = args.pca_threshold
outlier_thresholds_raw = args.outlier_thresholds

# Combine main + bucket paths for downstream processing
data_paths = [main_data_path] + bucket_data_paths

# Parse outlier thresholds (expects pairs: feature_name value)
if outlier_thresholds_raw:
    if len(outlier_thresholds_raw) % 2 != 0:
        raise ValueError("--outlier-thresholds must be provided as pairs: feat_name value ...")
    outlier_thresholds = {}
    it = iter(outlier_thresholds_raw)
    for feat in it:
        val = float(next(it))
        outlier_thresholds[feat] = val
else:
    outlier_thresholds = None

# Parse custom feature names into feature class references
feature_list = []
if custom_features:
    def _normalize_feature_key(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    feature_class_map = {}
    for class_name, cls in inspect.getmembers(ft, inspect.isclass):
        if cls is ft.feature:
            continue
        if not issubclass(cls, ft.feature):
            continue
        if cls.__module__ != ft.__name__:
            continue

        normalized_class_name = _normalize_feature_key(class_name)
        feature_class_map[normalized_class_name] = cls

        try:
            display_name = cls(None).name # type: ignore
            normalized_display_name = _normalize_feature_key(display_name)
            feature_class_map[normalized_display_name] = cls
        except Exception:
            pass

    for raw_name in feature_names:
        normalized_name = _normalize_feature_key(raw_name)
        if normalized_name not in feature_class_map:
            raise ValueError(f"Unknown custom feature name: '{raw_name}'")
        feature_list.append(feature_class_map[normalized_name])

#######################################
#TRAINING/MAIN BUCKET ANALYSIS
#######################################
if custom_features:
    training_bucket_neurons = fs.load_neurons_from_path(main_data_path, feature_list)
else: 
    training_bucket_neurons = fs.load_neurons_from_path(main_data_path)

#generate feature spaces
for neuron in training_bucket_neurons:
    neuron.generate_trial_average_baseline()
    if  not trial_average:
        neuron.generate_per_trial_difference(difference_method)
    else:
        neuron.generate_trial_average_difference(difference_method)


#generate dataframe
training_df = analysis.neurons_to_dataframe(training_bucket_neurons)
training_df["bucket_path"] = main_data_path

#apply z scores and filtering
mean, std = analysis.z_score(training_df, regex = "^diff_.*")

analysis.z_score_baseline_cols(training_df)
#TODO Custom thresholds
analysis.remove_extraneous(training_df, regex = "^filtering_z_score_.*")

#apply PCA and k_means
if use_pca:
    pca_model = analysis.apply_PCA(training_df)
    used_pc_vals = 0
    pc_sums = np.cumsum(pca_model.explained_variance_ratio_)
    for i in range(len(pc_sums)):
        used_pc_vals = i
        if pc_sums[used_pc_vals] >= pca_threshold:
            break
    
    num_clusters = analysis.find_num_clusters(training_df, regex=f"^PC_[0-{used_pc_vals}].*")
    k_means = analysis.apply_kmeans_clusters(training_df, num_clusters=num_clusters, regex=f"^PC_[0-{used_pc_vals}].*")

else: #apply to z_scores
    num_clusters = analysis.find_num_clusters(training_df)
    k_means = analysis.apply_kmeans_clusters(training_df, num_clusters)






######################################
#OTHER BUCKET ANALYSIS
######################################
bucket_dfs = []
for bucket_data in bucket_data_paths:
    if custom_features:
        bucket_neurons = fs.load_neurons_from_path(bucket_data, feature_list)
    else: 
        bucket_neurons = fs.load_neurons_from_path(bucket_data)

    #generate feature spaces
    for neuron in bucket_neurons:
        neuron.generate_trial_average_baseline()
        if  not trial_average:
            neuron.generate_per_trial_difference(difference_method)
        else:
            neuron.generate_trial_average_difference(difference_method)

    #generate dataframe
    bucket_df = analysis.neurons_to_dataframe(bucket_neurons)
    bucket_df["bucket_path"] = bucket_data

    #apply z scores and filtering
    analysis.z_score(bucket_df, regex = "^diff_.*", mean=mean, std= std)

    analysis.z_score_baseline_cols(bucket_df)
    #TODO Custom thresholds
    analysis.remove_extraneous(bucket_df, regex = "^filtering_z_score_.*")


    #apply PCA and k_means
    if use_pca:
        analysis.apply_PCA(bucket_df, model = pca_model)
        analysis.apply_kmeans_clusters(bucket_df, num_clusters=num_clusters, regex=f"^PC_[0-{used_pc_vals}].*", model = k_means)

    else: #apply to z_scores
        analysis.apply_kmeans_clusters(bucket_df, num_clusters, model = k_means)

    #add to df collection
    bucket_dfs.append(bucket_df)



######################################
#FINAL DATAFRAME PROCESSING
######################################
all_dfs = [training_df]
all_dfs.extend(bucket_dfs)

final_df = pd.concat(all_dfs)

#save anything neccessary to save
final_df.to_csv("example_data/final.csv", index = False)
training_df.to_csv("example_data/training.csv", index = False)
for i in range(len(bucket_dfs)):
    bucket_dfs[i].to_csv(f"example_data/bucket_{i}.csv", index = False)