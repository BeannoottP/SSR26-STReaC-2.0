import feature as ft


class feature_space:
    
    def __init__(self, spike_train):
        self.feature_list = []
        self.spike_train = spike_train


    def add_feature(self, feature):
        self.feature_list.append(feature)

    def evaluate_features(self):
        for feature in self.feature_list:
            feature.evaluate()

    def load_default_features(self):
        self.add_feature(ft.fr(self.spike_train))
        self.add_feature(ft.isi(self.spike_train))
        self.add_feature(ft.cv(self.spike_train))

    def __str__(self) -> str:
        string = ""
        for feature in self.feature_list:
            string += str(feature) + "\n"
        return string