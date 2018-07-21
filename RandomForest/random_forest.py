# reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
# reference: https://github.com/amstuta/random-forest/tree/master/python
import numpy as np
import pandas as pd


def read_dataset(filename):
    data = np.array(pd.read_csv(filename))
    return data


class RandomForest(object):
    def __init__(self, dataset, max_depth):
        self.max_depth = max_depth
        self.N, self.M = dataset.shape[0], dataset.shape[1] - 1
        self.nr_feat = int(np.sqrt(self.N))

    def split_dataset(self, dataset, index):
        """
        Algorithm:
          1. compute the matrix that splits the dataset to left and right subtree according to the feature value
          2. split the dataset based on the matrix
          3. choose the feature value maximizing the gini index
        """
        min_gini = None
        feat_val = dataset[:, index]
        left_or_right = (feat_val[:, np.newaxis] <= feat_val[np.newaxis, :]).astype(np.int32)
        for split in left_or_right:
            left, right = dataset[split], dataset[~split]


    def get_split_point(self, dataset):
        feat_index = np.random.choice(self.M, self.nr_feat, replace=False)
        import pdb; pdb.set_trace()
        for index in feat_index:
            left, right = self.split_dataset(dataset, index)

    def train(self):
        self.get_split_point(dataset)


if __name__ == '__main__':
    dataset = read_dataset('./sonar.all-data')
    model = RandomForest(dataset, 5)
    model.train()

