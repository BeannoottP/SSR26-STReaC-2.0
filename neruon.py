import numpy as np
import logging
import spike_train as st

class neuron:
    def __init__(self, data_path):
        self.src : str = data_path #path to folder containing 
        
        self.spike_times : np.ndarray = np.empty(0) #Full array of spike times
        self.stimulus_times : np.ndarray = np.empty(0) # full array of stimulus start times
        self.stimulus_count : int = 0 # integer count of number of stimulus
        self.meta_data : dict = {} # dictionary of metadata

        self.gather_data() #reads src files and saves to spike_times, stimulus_times, stimulus_count, and meta_data
        
        '''
        #test
        print(self.spike_times)
        print(self.stimulus_times)
        print(self.stimulus_count)
        print(self.meta_data)
        '''

        self.baseline_spike_trains : np.ndarray = np.empty(self.stimulus_count, dtype=st.spike_train)
        self.stimulation_spike_trains : np.ndarray = np.empty(self.stimulus_count, dtype=st.spike_train)

        self.generate_spike_trains()
        #print(self.baseline_spike_trains.shape)
        #print(self.stimulation_spike_trains.shape)

        #generate feature spaces
        self.pre_stimulation_spike_train.evaluate_feature_space()
        for i in range(self.stimulus_count):
            self.baseline_spike_trains[i].evaluate_feature_space()
            self.stimulation_spike_trains[i].evaluate_feature_space()


    def gather_data(self):
        self.spike_times = np.loadtxt(
            f"{self.src}/spikes.txt", ndmin=1
        )  # Save the spike data
        self.stimulus_times = np.loadtxt(
            f"{self.src}/light_on.txt", ndmin=1
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
        for i in range(self.stimulus_count):
            self.baseline_spike_trains[i] = st.spike_train(self, i, True)
            self.stimulation_spike_trains[i] = st.spike_train(self, i, False)

        pre_stimulation_spikes = np.array([t for t in self.spike_times if 0.0 <= t < self.stimulus_times[0]])
        self.pre_stimulation_spike_train = st.spike_train(time = self.stimulus_times[0], spike_times= pre_stimulation_spikes)
