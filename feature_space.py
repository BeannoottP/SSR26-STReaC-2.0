import feature as ft
import numpy as np


class feature_space:
    
    def __init__(self, spike_train = None, feature_space = None):
        self.feature_list = []
        if spike_train != None:
            self.spike_train = spike_train
        if feature_space != None:
            for feature in feature_space.feature_list:
                ft_type = type(feature)
                self.feature_list.append(ft_type(None))


    def add_feature(self, feature):
        self.feature_list.append(feature)

    def evaluate_features(self):
        for feature in self.feature_list:
            feature.evaluate()

    def load_default_features(self):
        self.add_feature(ft.fr(self.spike_train))
        self.add_feature(ft.cv(self.spike_train))
        self.add_feature(ft.bursts_per_second(self.spike_train))
        self.add_feature(ft.burst_fr(self.spike_train))
        self.add_feature(ft.percent_time_burst(self.spike_train))
        self.add_feature(ft.percent_spike_bursting(self.spike_train))
        self.add_feature(ft.mean_burst_duration(self.spike_train))
        self.add_feature(ft.mean_ibi(self.spike_train))
        self.add_feature(ft.non_bursting_fr(self.spike_train))
        self.add_feature(ft.bursting_fr_increase(self.spike_train))

    def get_feature(self, feature):
        for ft in self.feature_list:
            if type(ft) is feature:
                return ft
        return None
    
    def set_feature(self, feature, value):
        for ft in self.feature_list:
            if type(ft) is feature:
                ft.value = value
    
    def __str__(self) -> str:
        string = ""
        for feature in self.feature_list:
            string += str(feature) + "\n"
        return string
    

def average_features(feature_space_list):
    average_space = \
        feature_space(feature_space=feature_space_list[0]) #creates empty feature space of same shape
    
    for i in range(len(average_space.feature_list)): #iterate through ft indexs
        ft_type = type(average_space.feature_list[i]) #find feature type
        mean = np.mean([feature_space.get_feature(ft_type).value for feature_space in feature_space_list]) #calculate mean of all feature_spaces
        average_space.set_feature(ft_type, mean) #set value in trial_average

    return average_space


def calculate_difference(difference_method, feature_space_a, feature_space_b):
    difference_space = feature_space(feature_space=feature_space_a) #create empty feature space of same shape
    
    for i in range(len(difference_space.feature_list)): #iterate through ft indexs
        ft_type = type(difference_space.feature_list[i]) #find feature type
        difference = difference_method(feature_space_a.get_feature(ft_type), feature_space_b.get_feature(ft_type))
        difference_space.set_feature(ft_type, difference)

    return difference_space

