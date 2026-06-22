import numpy as np

def absolute_difference(feature_a, feature_b):
    return np.abs(feature_a.value - feature_b.value)

def modulation_factor(feature_a, feature_b):
    return (feature_a.value - feature_b.value) / (feature_a.value + feature_b.value) \
            if feature_a.value + feature_b.value != 0.0 else 0.0

def squared_difference(feature_a, feature_b):
    return np.square(feature_a.value - feature_b.value)

def relative_difference(feature_a, feature_b):
    return np.abs(feature_a.value - feature_b.value) / feature_b.value \
            if feature_b.value != 0.0 else feature_a.value