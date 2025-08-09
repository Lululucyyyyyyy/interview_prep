import numpy as np

class Node:
    """
    Parameters
    ----------
    data: numpy.ndarray, default=None
        The dataset includes X and Y
    children: dict(feat_value: Node), default=None
        Dict of children
    split_on: int, default=None
        Index of the feature that node was split on that
    pred_class : str, default=None
        The predicted class for the node (only applicable to leaf nodes)
    is_leaf: bool, default=False
        Determine whether the node is leaf or not
    """

    def __init__(self, data=None, children=None, split_on = None, pred_class=None, is_leaf=False):

        self.data = data
        self.children = children
        self.split_on = split_on
        self.pred_class = pred_class
        self.is_leaf = is_leaf

class DecisionTreeClassifier:
    def __init__(self):
        self.root = Node()

    @staticmethod
    def calculate_entropy(Y):
        """
        Parameters: Y: numpy.ndarray
            The labels array.

        Returns: entropy: flaot
            The entropy value of the given labels.
        """
        _, labels_counts = np.unique(Y, return_counts=True)
        total_instances = len(Y)
        entropy = sum([label_count / total_instances * np.log2(1 / (label_count / total_instances)) for label_count in labels_counts])
        return entropy
    
    def split_on_feature(self, data, feat_index):
        """
        Split the dataset based on a specific feature index.

        Parameters:
        data: numpy.ndarray
            The dataset to be split.

        feat_index: int
            The index of the feature to perform the split.

        Returns:
        - split_nodes: dict
            A dictionary of split nodes. 
            (feature value as key, corresponding node as value)

        - weighted_entropy: float
            The weighted entropy of the split.
        """
        feature_values = data[:, feat_index]
        unique_values = np.unique(feature_values)

        split_nodes = {}
        weighted_entropy = 0
        total_instances = len(data)

        for unique_value in unique_values:
            partition = data[data[:, feat_index] == unique_value, :]
            node = Node(data=partition)
            split_nodes[unique_value] = node
            partition_y = self.get_y(partition)
            node_entropy = self.calculate_entropy(partition_y)
            weighted_entropy += (len(partition) / total_instances) * node_entropy

        return split_nodes, weighted_entropy
    
    def best_split(self, node):
        """
        Find the best split for the given node.
        (data in node.data)

        Parameters:
        ----------
        node: Node
            The node for which the best split is being determined.

        If the node meets the criteria to stop splitting:
            - Mark the node as a leaf.
            - Assign a predicted class for future predictions based on the target values (y).
            - return.

        Otherwise:
            - Initialize variables for tracking the best split.
            - Iterate over the features to find the best split.
            - Split the data based on each feature and calculate the weighted entropy of the split.
            - Compare the current weighted entropy with the previous best entropy.
            - Update the best split variables if the current split has lower entropy.
            - update the node with the best split information, including child nodes and the feature index used for the split.
            - Recursively call the best_split function for each child node.

        """
        # Check if the node meets the criteria to stop splitting
        if self.meet_criteria(node):
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        # Initialize variables for tracking the best split
        index_feature_split = -1
        min_entropy = 1

        # iterate over all features, ignore (y)
        for i in range(data.shape[1] - 1):
            split_nodes, weighted_entropy = self.split_on_feature(node.data, i)
            if weighted_entropy < min_entropy:
                child_nodes, min_entropy = split_nodes, weighted_entropy
                index_feature_split = i

        node.children = child_nodes
        node.split_on = index_feature_split

        # Recursively call the best_split function for each child node
        for child_node in child_nodes.values():
            self.best_split(child_node)

    def meet_criteria(self, node):
        """
        Check if the criteria for stopping the tree expansion is met for a given node. Here we only check if the entropy of the target values (y) is zero.
        Additionally, you can customize criteria based on your specific requirements. For instance, you can set the maximum depth for the decision tree or incorporate other conditions for stopping the tree expansion. Modify the implementation of this method according to your desired criteria.

        Parameters:
        -----------
        node : Node
            The node to check for meeting the stopping criteria.

        Returns:
        -----------
        bool
            True if the criteria is met, False otherwise.

        """

        y = self.get_y(node.data)
        return True if self.calculate_entropy(y) == 0 else False
    
    def get_y(data):
        """
        Get the target (y) from the data.

        Parameters:
        -----------
        data : numpy.ndarray
            The input data containing features and the target variable.

        Returns:
        -----------
        y: numpy.ndarray
            The target variable extracted from the data.

        """
        y = data[:, -1]
        return y
    
    def get_pred_class(Y):
        """
        Get the predicted class label based on the majority vote.

        Parameters:
        -----------
        Y : numpy.ndarray
            The array of class labels.

        Returns:
        -----------
        str
            The predicted class label.

        """

        labels, labels_counts = np.unique(Y, return_counts=True)
        index = np.argmax(labels_counts)
        return labels[index]
    
    def fit(self, X, Y):
        """
        Fit the decision tree model to the provided dataset.

        Parameters:
        -----------
        X: numpy.ndarray
            The input features of the dataset.

        Y: numpy.ndarray
            The target labels of the dataset.
        """
        data = np.column_stack([X, Y])
        self.root.data = data
        self.best_split(self.root)

    def predict(self, X):
        """
        Predict the class labels for the given input features.

        Parameters:
        -----------
        X: numpy.ndarray
            The input features for which to make predictions. Should be a 2D array-like object.

        Returns:
        -----------
        predictions: numpy.ndarray
            An array of predicted class labels.

        """

        # Traverse the decision tree for each input and make predictions
        predictions = np.array([self.traverse_tree(x, self.root) for x in X])
        return predictions
    
    def traverse_tree(self, x, node):
        """
        Recursively traverse the decision tree to predict the class label for a given input.

        Parameters:
        -----------
        x:
            The input for which to make a prediction.

        node:
            The current node being traversed in the decision tree.

        Returns:
        -----------
        predicted_class:
            The predicted class label for the input feature.

        """

        # Check if the current node is a leaf node
        if node.is_leaf:
            return node.pred_class

        # Get the feature value at the split point for the current node
        feat_value = x[node.split_on]

        # Recursively traverse the decision tree using the child node corresponding to the feature value
        predicted_class = self.traverse_tree(x, node.children[feat_value])

        return predicted_class