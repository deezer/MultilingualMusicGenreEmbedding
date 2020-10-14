import pandas as pd

import utils


class DataHelper:
    """ Data helper class to load and prepare data for evaluation"""

    def __init__(self, sources, target, dataset_path=None):
        self.sources = sources
        self.target = target
        self.dataset_path = dataset_path
        self.dataset_df = None
        if self.dataset_path is not None:
            self._load_dataset()

    def _load_dataset(self):
        """Load the dataset"""
        print("Loading dataset...")
        assert self.dataset_path is not None
        cols = self.sources + [self.target]
        dst = utils.load_tag_csv(self.dataset_path, cols)
        print("Loaded.")
        prev_len = len(dst)
        print("Filtering...")
        cols = self.sources + [self.target] + ["fold"]
        rows = []
        for t in zip(*[dst[s] for s in cols]):
            if len(self.sources) == 2:
                t1, t2, t3, f = t
            else:
                t1, t3, f = t
                t2 = 0
            if len(t1) + len(t2) == 0 or len(t3) == 0:
                continue
            rows.append((t1, t2, t3, f))
        self.dataset_df = pd.DataFrame(rows, columns=cols)
        print(f"Kept {len(self.dataset_df)} on {prev_len} initial rows in")

    def get_test_data(self, tag_manager, fold, as_array=True):
        """Return data for evaluation"""
        return self._get_dataset_split(tag_manager, fold, as_array)

    def _get_dataset_split(self, tag_manager, fold, as_array):
        """Get the dataset split and formatted"""
        bool_index = self.dataset_df.fold == fold
        df = self.dataset_df[bool_index]
        train_data, target_data = self._format_dataset_rows_and_split(df, self.sources, self.target)
        return self._transform_sources_and_target_data(tag_manager, train_data, target_data, as_array)

    def _format_dataset_rows_and_split(self, df, sources, target):
        """Format the dataset rows in train and target data for evaluation"""
        train_data = []
        target_data = []
        for t in zip(*[df[s] for s in list(sources) + [target]]):
            stags = t[:len(sources)]
            ttags = t[-1]
            train_data.append([])
            for i, s in enumerate(sources):
                train_data[-1].extend(self._append_source_to_tags(stags[i], s))
            target_data.append(self._append_source_to_tags(ttags, target))
        return train_data, target_data

    def _append_source_to_tags(self, tags, source):
        """Append the source in front of each tag"""
        return [source + ":" + t for t in tags]

    def _transform_sources_and_target_data(self, tag_manager, source_df, target_df, as_array):
        """Transforms the source and target data using the tag manager"""
        return tag_manager.transform_for_sources(source_df, as_array), tag_manager.transform_for_target(target_df, as_array)

