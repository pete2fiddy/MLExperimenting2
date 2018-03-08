import numpy as np
from classification.tree.general_decision_tree import TreeNode

class DecisionTree(TreeNode):

    def _init_node(self):
        return DecisionTree()

    def _pretransform(self, X):
        return X

    def _prefit(self, X, y):
        return None
