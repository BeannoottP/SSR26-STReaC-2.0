import pandas as pd
import numpy as np
import re
from scipy.stats import zscore

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
		row["src"] = neuron.src
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
		_dump_space(neuron.trial_average_difference, "diff")

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
	