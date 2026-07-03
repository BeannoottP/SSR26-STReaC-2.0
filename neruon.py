"""
neuron.py

Represents neural spike train data for one neuron stored in lists of baseline and stimulation spike_trains
Stores feature spaces for: 
    trial_average_baseline #all baseline spike times averaged
    averaged_difference #difference between baseline and stimulation spike train features, either compared per trial or by average

Author: Bennett Ptak (2026)
"""


import numpy as np
import spike_train as st
import feature_space as ft_sp
import feature as ft

class neuron:
    def __init__(self, data_path, time = 1.0):
        self.src : str = data_path #path to folder containing neural data
        self.time = time #spike train length
        
        self.spike_times : np.ndarray | None = None #Full array of spike times
        self.stimulus_times : np.ndarray | None = None # full array of stimulus start times
        self.stimulus_count : int = 0 # integer count of number of stimulus
        self.meta_data : dict = {} # dictionary of metadata

        self.gather_data() #reads src files and saves to spike_times, stimulus_times, stimulus_count, and meta_data
        
        self.baseline_spike_trains : np.ndarray = np.empty(self.stimulus_count, dtype=st.spike_train)
        self.stimulation_spike_trains : np.ndarray = np.empty(self.stimulus_count, dtype=st.spike_train)

        self.generate_spike_trains()

        #generate feature spaces
        self.pre_stimulation_spike_train.evaluate_feature_space()
        for i in range(self.stimulus_count):
            self.baseline_spike_trains[i].evaluate_feature_space()
            self.stimulation_spike_trains[i].evaluate_feature_space()

        self.trial_average_baseline : ft_sp.feature_space | None = None
        self.averaged_difference : ft_sp.feature_space | None = None




    def gather_data(self):
        self.spike_times = np.loadtxt(
            f"{self.src}/spikes.txt", ndmin=1
        )  # Save the spike data
        self.stimulus_times = np.loadtxt(
            f"{self.src}/light_on.txt", ndmin=1,
        )   # Save the stimulus time data
        self.stimulus_count = self.stimulus_times.shape[
            0
        ]  # Set the stimulus count as the number of times the light was cut on
        
        self.read_in_meta_data()

    def read_in_meta_data(self):
        """
        Function that reads in meta data of neuron.
        """
        # Open meta_data.txt file
        for line in open(
            f"{self.src}/meta_data.txt"
        ).readlines():
            if line[0] != "#":  # Skip lines with #
                splits = line.split(":")  # Split data by column
                self.meta_data[splits[0]] = (
                    eval(splits[1])
                    if splits[0] == "cell_num" or splits[0] == "distance"
                    else splits[1][1:-1]
                )  # Store evaluated data

    def generate_spike_trains(self):
        '''
        generates three sets of spike trains
        Baseline Spike trains represent the time directly before stimulation
        Stimulation Spike trains represent the time directly after stimulation
        Pre_stimulation_spike_train is a single spike train with spike time 0 -> time first stimulation
        '''
        for i in range(self.stimulus_count): #iterates through stimulation times and generates spike trains for 
            self.baseline_spike_trains[i] = st.spike_train(self, i, True, time = self.time)
            self.stimulation_spike_trains[i] = st.spike_train(self, i, False, time = self.time)

        #generates spike train for time 0 to first stimulus time
        pre_stimulation_spikes = np.array([t for t in self.spike_times if 0.0 <= t < self.stimulus_times[0]]) # type: ignore
        self.pre_stimulation_spike_train = st.spike_train(time = self.stimulus_times[0], spike_times = pre_stimulation_spikes) # type: ignore

    def generate_trial_average_baseline(self):
        '''
        generates averaged feature space of all baseline spike trains
        '''
        #
        feature_space_list = [st.feature_space for st in self.baseline_spike_trains]
        self.trial_average_baseline = ft_sp.average_features(feature_space_list)

    def generate_trial_average_difference(self, difference_method):
        '''
        generates difference space comparing all stimulation trials against averaged baseline feature space
        '''
        difference_space_list = [st.generate_difference_space_against_average(difference_method) for st in self.stimulation_spike_trains]
        self.averaged_difference = ft_sp.average_features(difference_space_list)

    def generate_per_trial_difference(self, difference_method):
        '''
        generates difference space comparing all stimuation trials against their baseline features
        '''
        difference_space_list = [st.generate_difference_space_against_trial(difference_method) for st in self.stimulation_spike_trains]
        self.averaged_difference = ft_sp.average_features(difference_space_list)





