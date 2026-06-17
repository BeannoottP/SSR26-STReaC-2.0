import abc
import spike_train as st
import numpy as np

class feature(abc.ABC):
    def __init__(self, train, name):
        self.spike_train : st.spike_train = train #spike train of which feature is of
        self.value = None #features value once evaluated
        self.name = name #short string describing feature

    @abc.abstractmethod
    def evaluate(self):
        pass

    def __str__(self) -> str:
        return self.name + " :" + str(self.value)


class isi(feature):
    '''
    Mean ISI of a spike train
    Returns spike_train.time for no spikes/ 1 spike
    '''
    def __init__(self, train) -> None:
        super().__init__(train, "Mean ISI")
    
    def evaluate(self):
        #edge case for 0/1 spikes
        trainLen = self.spike_train.spike_times.size
        if trainLen <= 1:
            self.value = self.spike_train.time
            return
        
        isis = np.diff(self.spike_train.spike_times) #difference between spike times
        self.value = np.mean(isis) #mean of ISIs


class fr(feature):
    '''
    firing rate of a spike train
    '''

    def __init__(self, train) -> None:
        super().__init__(train, "Firing Rate")

    def evaluate(self):
        self.value = self.spike_train.spike_times.size / self.spike_train.time

class cv(feature):
    '''
    Coefficient of variance of ISI
    Returns 0 for no/1 spikes
    '''

    def __init__(self, train):
        super().__init__(train, "CV")
    
    def evaluate(self):
                #edge case for 0/1 spikes
        trainLen = self.spike_train.spike_times.size
        if trainLen <= 1:
            self.value = 0.0
            return

        isis = np.diff(self.spike_train.spike_times) #difference between spike times
        mean = np.mean(isis) #mean of isis
        std_dev = np.std(isis, mean=mean)
        self.value = std_dev/mean

class bursts_per_second(feature):
    '''
    Bursts per second
    '''

    def __init__(self, train):
        super().__init__(train, "Burst/s")

    def evaluate(self):
        self.value = self.spike_train.n_bursts / self.spike_train.time

class burst_fr(feature):
    '''
    Mean burst firing rate
    Returns 0 for no bursts
    '''
    def __init__(self, train):
        super().__init__(train, "Burst FR")

    def evaluate(self):
        self.value = np.mean(self.spike_train.burst_firing_rate) \
                     if self.spike_train.burst_firing_rate.shape[0] > 0 else 0.0
        
class percent_time_burst(feature):
    '''
    Percent time bursting
    '''
    def __init__(self, train):
        super().__init__(train, "Percent Time Bursting")

    def evaluate(self):
        self.value = np.sum(self.spike_train.burst_durations) \
                    / self.spike_train.time

class percent_spike_bursting(feature):
    '''
    Percent of spikes in spike train that are part of a burst
    Returns 0 if no spikes
    '''

    def __init__(self, train):
        super().__init__(train, "Percent Spike Bursting")

    def evaluate(self):
        self.value = np.sum(self.spike_train.burst_spikes) \
                    / self.spike_train.spike_times.shape[0] \
                    if self.spike_train.spike_times.shape[0] > 0 else 0.0
        
class mean_burst_duration(feature):
    '''
    mean burst duration
    0 if no bursts
    '''                    
    def __init__(self, train):
        super().__init__(train, "Mean Burst Duration")

    def evaluate(self):
        self.value = np.mean(self.spike_train.burst_durations) \
                    if self.spike_train.burst_durations.shape[0] > 0 else 0.0

class mean_ibi(feature):
    '''
    Mean Inter-Burst Interval
    For 0/1 bursts, returns length of train
    '''
    def __init__(self, train):
        super().__init__(train, "Mean Inter-Burst Interval")
    
    def evaluate(self):
        self.value = np.mean(self.spike_train.inter_burst_intervals) \
                    if self.spike_train.n_bursts > 1 else self.spike_train.time
        
class non_bursting_fr(feature):
    '''
    Non-bursting Firing Rate
    '''
    def __init__(self, train):
        super().__init__(train, "Non-Bursting FR")

    def evaluate(self):
        self.value = (self.spike_train.spike_times.shape[0] - np.sum(self.spike_train.burst_spikes)) \
                 / (self.spike_train.time - np.sum(self.spike_train.burst_durations))
        
class bursting_fr_increase(feature):
    '''
    Increase from nonbursting to bursting firing rate by burst_fr/non_burst_fr
    0 if no bursts and/or no spikes
    '''
    def __init__(self, train):
        super().__init__(train, "Bursting Firing Rate Increase")

    def evaluate(self):
        burst_fr = np.mean(self.spike_train.burst_firing_rate) \
                    if self.spike_train.burst_firing_rate.shape[0] > 0 else 0.0
        non_bursting_fr = (self.spike_train.spike_times.shape[0] - np.sum(self.spike_train.burst_spikes)) \
                 / (self.spike_train.time - np.sum(self.spike_train.burst_durations))
        self.value = burst_fr / non_bursting_fr \
                    if non_bursting_fr > 0 else 0.0
