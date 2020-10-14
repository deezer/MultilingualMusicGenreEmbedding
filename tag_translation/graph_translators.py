import numpy as np

from utils import read_translation_table
from tag_translation.translator import Translator


class DbpMappingTranslator(Translator):

    def __init__(self, tag_manager, table_path):
        self.tag_manager = tag_manager
        self.source_genres = self.tag_manager.source_tags
        self.target_genres = self.tag_manager.target_tags

        self.norm_table = read_translation_table(table_path, tag_manager)
        self.W = self.norm_table.values.T
        self.b = np.zeros((1,))

    def get_translation_table(self):
        raise NotImplementedError("")

    def get_translation_table_per_source(self):
        raise NotImplementedError("")

    def predict_scores(self, eval_data):
        norm = np.count_nonzero(eval_data, axis=1)
        print(norm)
        return eval_data.dot(self.W.T) / norm.reshape(norm.shape[0], 1)
