import numpy as np

class Translator():
    """The base class for translator classes
    """

    def train_and_evaluate(self, train_data, target_data, eval_data, eval_target, score_function):
        """Train and evaluate the translator with the input data
        :param train_data: the training data as np.array
        :param target_data: the multi-label ground-truth
        :param eval_data: the evaluation data as np.array
        :param eval_target: the multi-label ground-truth used in evaluation
        :param score_function: the scoring function used in evaluation
        """
        raise NotImplementedError()

    def predict_scores(self, eval_data):
        """Predict the scores for each tag given the data
        :param eval_data: the evaluation data as np.array
        """
        norm = np.count_nonzero(eval_data, axis=1)
        return eval_data.dot(self.W.T) / norm.reshape(norm.shape[0], 1)
