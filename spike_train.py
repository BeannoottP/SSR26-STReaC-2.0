import neruon as nrn
import numpy as np
import poisson_surprise as ps_sp

class spike_train:
    '''
    this is a class definintion i need to 
    '''
    def __init__(self, neuron, stimulation_index, baseline_flag):
        #TODO parameters around time sampling
        self.time = 1.0 #total length of spike train in seconds
        self.neuron : nrn.neuron = neuron #parent neuron
        self.stim_index : int = stimulation_index #index in neuron.stimulation_times at which this neuron 
        self.stim_time : float = self.neuron.stimulus_times[self.stim_index] #time of stimulation
        self.baseline_flag : bool = baseline_flag #true means prestimulation, false means during/post stim

        self.spike_times : np.ndarray = np.empty(0) #spike times in train, normalized with start time of 0
        self.seperate_spike_times()

        #test
        print("Index :" + str(self.stim_index) + ", Stim Time: " + str(self.stim_time) + ", Baseline Flag : " + str(self.baseline_flag) + ", Num Spikes:" + str(self.spike_times.shape))
        print(self.spike_times)
        #calculate burst and burst properties
        self.bursts = ps_sp.run_poisson_surprise(self.spike_times, surprise_threshold=3.0)
        self.n_bursts, self.burst_firing_rate, self.burst_start_times, self.burst_durations, \
            self.burst_spikes, self.inter_burst_intervals = ps_sp.burst_properties(
                                                            self.bursts)
        
    


    def seperate_spike_times(self):
        full_times = self.neuron.spike_times
        start_time = self.stim_time #starting time bound of spike train, inclsive
        end_time = None #ending time bound of spike train, non-inclusive
        
        if self.baseline_flag: #seperates time bound for baseline vs. non baseline
            start_time = self.stim_time - self.time

        end_time = start_time + self.time #define end time

        self.spike_times = np.array([t for t in full_times if start_time <= t < end_time]) #list comprehension creates new list with only times in bound
        self.spike_times = np.subtract(self.spike_times, start_time) #normalize to start at 0
