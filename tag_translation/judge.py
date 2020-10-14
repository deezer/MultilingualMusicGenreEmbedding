import numpy as np
from sklearn.metrics import roc_auc_score


class Judge():
    """Judge class to compute evaluation metrics"""
    def compute_macro_metrics(ground_truth, pred):
        """Compute macro roc auc scores
        :param ground_truth: the ground truth annotations
        :param pred: the predicted annotations:
        :return: macro auc"""
        # Check if there are classes not used in this fold and filter them
        mask = np.nonzero(np.sum(ground_truth, axis=0))[0]
        ground_truth = ground_truth[:, mask]
        pred = pred[:, mask]
        return roc_auc_score(ground_truth, pred, average='macro')
