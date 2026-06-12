import abc
import spike_train as st

class feature(abc.ABC):

    def __init__(self, train, name) -> None:
        self.spike_train : st.spike_train = train #spike train of which feature if of
        self.value = None #features value once evaluated
        self.name = name #short string describing feature

    @abc.abstractmethod
    def evaluate(self):
        pass


class isi(feature):
    '''
    Mean ISI of a spike train
    '''
    def __init__(self, train, name) -> None:
        super().__init__(train, "Mean ISI")
    
    def evaluate(self):
        trainLen = self.spike_train.spike_times.shape[0]
        sumISI = 0.0
        for i in range(1,trainLen):
            pass


class fr(feature):
    '''
    firing rate of a spike train
    '''
                    
