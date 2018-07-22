# reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
# reference: https://github.com/amstuta/random-forest/tree/master/python
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool


def read_dataset(filename):
    data = np.array(pd.read_csv(filename))
    return data


class DecisionNode(object):
    def __init__(self, pivot=None, idx=None, left=None, right=None, cls=None):
        self.pivot = pivot
        self.idx = idx
        self.left = left
        self.right = right
        self.cls = cls


class DecisionTree(object):
    def __init__(self, max_depth=-1):
        self.max_depth = max_depth
        self.tree = None

    def gini_index(self, dataset, split):
        """
        Algorithm:
          1. count the total number of features nr_feat
          2. split the dataset to left and right according to the split
          3. compute the gini index: p * creterion(left) + (1 - p) * creterion(right)
        """
        nr_feat = dataset.shape[0]
        left, right = dataset[split, -1], dataset[~split, -1]
        p = len(left) / nr_feat

        def div_cls(cls, p, nr):
            if len(cls) == 0:
                return 0, 0
            if len(cls) == 2:
                return p / nr
            return 0, p[0] / nr

        def creterion(data):
            if len(data) == 0:
                return 0
            cls, p = np.unique(data, return_counts=True)
            p1, p2 = div_cls(cls, p, len(data))
            return 1 - p1**2 - p2**2

        gini_index = p * creterion(left) + (1 - p) * creterion(right)

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
        return DecisionNode(pivot, min_idx, left, right)

    def build_node(self, dataset):
        cls, counts = np.unique(dataset[:, -1], return_counts=True)
        inds = np.argsort(counts)
        return DecisionNode(cls=cls[inds[-1]])

    def build_tree(self, dataset, depth):
        if depth == 0:
            return self.build_node(dataset)

        node = self.get_split_point(dataset)
        if len(node.left) == 0 or len(node.right) == 0:
            node = self.build_node(dataset)
        else:
            node.left = self.build_tree(node.left, depth - 1)
            node.right = self.build_tree(node.right, depth - 1)
        return node

    def fit(self, dataset):
        self.N, self.M = dataset.shape[0], dataset.shape[1] - 1
        self.nr_feat = int(np.sqrt(self.M))
        self.tree = self.build_tree(dataset, depth=self.max_depth)

    def predict(self, feat):
        cur_node = self.tree
        while cur_node.cls is None:
            pivot = cur_node.pivot
            idx = cur_node.idx
            if feat[idx] <= pivot:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        return cur_node.cls


class RandomForest(object):
    def __init__(self, nr_trees, max_depth=-1):
        self.nr_trees = nr_trees
        self.max_depth = max_depth
        self.trees = None

    def build_tree(self, dataset):
        tree = DecisionTree(self.max_depth)
        tree.fit(dataset)
        return tree

    def fit(self, dataset):
        nr_samples = len(dataset)
        with Pool(self.nr_trees) as pool:
            data = map(lambda i: np.random.choice(len(dataset), size=nr_samples), range(self.nr_trees))
            data = map(lambda i: dataset[i], data)
            self.trees = pool.map(self.build_tree, data)

    def predict(self, feat):
        preds = [tree.predict(feat) for tree in self.trees]
        return max(set(preds), key=preds.count)


def demo():
    dataset = read_dataset('./sonar.all-data')
    np.random.shuffle(dataset)
    size = len(dataset)
    train_size = size // 5 * 4
    train, test = dataset[:train_size], dataset[train_size:]

    def test_rf(nr_trees):
        model = RandomForest(nr_trees)
        model.fit(train)
        rec = 0
        for feat in test:
            rec += int(feat[-1] == model.predict(feat))
        print('nr_trees: {}, accuracy: {}'.format(nr_trees, rec / len(test)))

    test_rf(1)
    test_rf(2)
    test_rf(5)
    test_rf(10)


if __name__ == '__main__':
    demo()

