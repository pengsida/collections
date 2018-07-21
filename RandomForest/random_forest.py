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

    def gini_index(self, dataset, split):
        """
        Algorithm:
          1. count the total number of features nr_feat
          2. split the dataset to left and right according to the split
          3. the gini_index of left: (1 - p_1^2 - p_2^2) / nr_left_feat * nr_feat
          4. compute the gini_index of right likewise
        """
        nr_feat = dataset.shape[0]
        left, right = dataset[split, -1], dataset[~split, -1]
        nr_left_feat, nr_right_feat = len(left), len(right)

        def div_cls(cls, p, nr):
            if len(cls) == 0:
                return 0, 0
            if len(cls) == 2:
                return p / nr
            return 0, p[0] / nr

        gini_index = 0
        if len(left) != 0:
            cls, p = np.unique(left, return_counts=True)
            p1, p2 = div_cls(cls, p, nr_left_feat)
            gini_index += (1 - p1**2 - p2**2) / nr_left_feat * nr_feat

        if len(right) != 0:
            cls, p = np.unique(right, return_counts=True)
            p1, p2 = div_cls(cls, p, nr_right_feat)
            gini_index += (1 - p1**2 - p2**2) / nr_right_feat * nr_feat

        return gini_index

    def split_dataset(self, dataset, index):
        """
        Algorithm:
          1. compute the matrix that splits the dataset to left and right subtree according to the feature value
          2. split the dataset based on the matrix
          3. choose the feature value minimizing the gini index
        """
        min_gini = None
        min_idx = 0
        feat_val = dataset[:, index]
        left_or_right = (feat_val[:, np.newaxis] >= feat_val[np.newaxis, :])
        for idx, split in enumerate(left_or_right):
            gini = self.gini_index(dataset, split)
            min_gini, min_idx = (gini, idx) if min_gini is None or gini < min_gini else (min_gini, min_idx)
        return min_gini, min_idx

    def get_split_point(self, dataset):
        """
        Algorithm:
          1. select the decision features randomly
          2. select the feature minimizing the gini index
        """
        feat_index = np.random.choice(self.M, self.nr_feat, replace=False)
        min_rs = None
        min_idx = 0
        for index in feat_index:
            rs = self.split_dataset(dataset, index)
            min_rs, min_idx = (rs, index) if min_rs is None or rs[0] < min_rs[0] else (min_rs, min_idx)
        pivot = dataset[min_rs[1], min_idx]
        split = (pivot >= dataset[:, min_idx])
        left, right = dataset[split], dataset[~split]
        return {'gini': min_rs[0], 'idx': min_idx, 'left': left, 'right': right}

    def train(self):
        self.get_split_point(dataset)


if __name__ == '__main__':
    dataset = read_dataset('./sonar.all-data')
    model = RandomForest(dataset, 5)
    model.train()

