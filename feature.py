import abc
import spike_train as st
import numpy as np

class feature(abc.ABC):

    def __init__(self, train, name):
        self.spike_train : st.spike_train = train #spike train of which feature if of
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
    Returns -1 for no/1 spikes
    '''

    def __init__(self, train):
        super().__init__(train, "CV")
    
    def evaluate(self):
                #edge case for 0/1 spikes
        trainLen = self.spike_train.spike_times.size
        if trainLen <= 1:
            self.value = -1
            return

        isis = np.diff(self.spike_train.spike_times) #difference between spike times
        mean = np.mean(isis) #mean of isis
        std_dev = np.std(isis, mean=mean)
        self.value = std_dev/mean
        
        

                    
