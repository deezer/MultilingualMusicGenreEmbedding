import os
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.utils import check_random_state
from sklearn.preprocessing import MultiLabelBinarizer

import utils
from utils import langs


def filter_data(df, target, sources):
    """Filter out the samples that do not contain tags from at least one source and target languages
    :param df: the corpus
    :param target: the target language
    :param sources: the source languages
    """
    filtered_df = pd.DataFrame(df.dropna(subset=[target]))
    # Drop also the lines which do not have at least a source
    filtered_df = filtered_df.dropna(subset=sources, thresh=1)
    filtered_df[target] = filtered_df[target].apply(literal_eval)
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def mark_groups_for_samples(df, n_samples, extra_criterion):
    """Return groups, an array of size n_samples, marking the group to which each sample belongs. The default group is -1 if extra_criterion is None. If a criterion is given (artist or album), then this information is taken into account.
    :param df: corpus
    :param n_samples: the total number of samples
    :param extra_criterion: an extra condition to be taken into account in the split such as music items from the same artist should be all in the same fold
    """
    groups = np.array([-1 for _ in range(n_samples)])
    if extra_criterion is None:
        return groups

    if extra_criterion == "artist":
        crit_col = "artistmbid"
    elif extra_criterion == "album":
        crit_col = "releasegroupmbid"
    else:
        return groups

    gp = df.groupby(crit_col)
    i_key = 0
    for g_key in gp.groups:
        samples_idx_per_group = gp.groups[g_key].tolist()
        groups[samples_idx_per_group] = i_key
        i_key += 1
    return groups


def select_fold(index_label, desired_samples_per_label_per_fold, desired_samples_per_fold, random_state):
    """For a label, find the fold where the next sample should be distributed
    :param index_label: the targetted label
    :param desired_samples_per_label_per_fold: the desired number of samples per label and per fold
    :param desired_samples_per_fold: the desired number of samples per fold
    :param random_state: a random object shared in each fold selection
    """
    # Find the folds with the largest number of desired samples for this label
    largest_desired_label_samples = max(desired_samples_per_label_per_fold[:, index_label])
    folds_targeted = np.where(desired_samples_per_label_per_fold[:, index_label] == largest_desired_label_samples)[0]

    if len(folds_targeted) == 1:
        selected_fold = folds_targeted[0]
    else:
        # Break ties by considering the largest number of desired samples
        largest_desired_samples = max(desired_samples_per_fold[folds_targeted])
        folds_re_targeted = np.intersect1d(np.where(
            desired_samples_per_fold == largest_desired_samples)[0], folds_targeted)

        # If there is still a tie break it by picking a random index
        if len(folds_re_targeted) == 1:
            selected_fold = folds_re_targeted[0]
        else:
            selected_fold = random_state.choice(folds_re_targeted)
    return selected_fold


def iterative_split(df, out_file, target, n_splits, extra_criterion=None, seed=None):
    """Implement the iterative split algorithm (see paper) and save results
    :param df: input data
    :param out_file: the output file containing the same data as the input corpus plus a column specifying the fold
    :param target: is the target language for which the files are generated
    :param n_splits: the number of folds
    :param extra_criterion: an extra condition to be taken into account in the split such as music items from the same artist should be all in the same fold
    """
    print("Starting the iterative split")
    random_state = check_random_state(seed)

    mlb_target = MultiLabelBinarizer(sparse_output=True)
    M = mlb_target.fit_transform(df[target])

    n_samples = len(df)
    n_labels = len(mlb_target.classes_)

    # If the extra criterion is given create "groups" showing to which group each sample belongs
    groups = mark_groups_for_samples(df, n_samples, extra_criterion)

    ratios = np.ones((1, n_splits))/n_splits
    # Calculate the desired number of samples for each fold
    desired_samples_per_fold = ratios.T * n_samples

    # Calculate the desired number of samples of each label for each fold
    number_samples_per_label = np.asarray(M.sum(axis=0)).reshape((n_labels, 1))
    desired_samples_per_label_per_fold = np.dot(ratios.T, number_samples_per_label.T)  # shape: n_splits, n_samples

    seen = set()
    out_folds = np.array([-1 for _ in range(n_samples)])

    count_seen = 0
    print("Going through the samples")
    while n_samples > 0:
        # Find the index of the label with the fewest remaining examples
        valid_idx = np.where(number_samples_per_label > 0)[0]
        index_label = valid_idx[number_samples_per_label[valid_idx].argmin()]
        label = mlb_target.classes_[index_label]

        # Find the samples belonging to the label with the fewest remaining examples
        # Select all samples belonging to the selected label and remove the indices of the samples which have been already seen
        all_label_indices = set(M[:, index_label].nonzero()[0])
        indices = all_label_indices - seen
        assert(len(indices) > 0)

        print(label, index_label, number_samples_per_label[index_label], len(indices))

        for i in indices:
            if i in seen:
                continue

            # Find the folds with the largest number of desired samples for this label
            selected_fold = select_fold(index_label, desired_samples_per_label_per_fold,
                                        desired_samples_per_fold, random_state)

            # Put in this fold all the samples which belong to the same group
            idx_same_group = np.array([i])
            if groups[i] != -1:
                idx_same_group = np.where(groups == groups[i])[0]

            # Update the folds, the seen, the number of samples and desired_samples_per_fold
            out_folds[idx_same_group] = selected_fold
            seen.update(idx_same_group)
            count_seen += idx_same_group.size
            n_samples -= idx_same_group.size
            desired_samples_per_fold[selected_fold] -= idx_same_group.size

            # The sample may have multiple labels so update for all
            for idx in idx_same_group:
                _, all_labels = M[idx].nonzero()
                desired_samples_per_label_per_fold[selected_fold, all_labels] -= 1
                number_samples_per_label[all_labels] -= 1

    df['fold'] = out_folds
    print(count_seen, len(df))
    df.to_csv(out_file, sep='\t', index=False)


corpus_file = utils.CORPUS_FILE_PATH
# Create the output folder if it does not exist
out_dir = utils.FOLDS_DIR
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print("Directory ", out_dir, " created ")

# Read the corpus
df = pd.read_csv(corpus_file, index_col='id')
# For each language as target split the data in 4 folds
for target in langs:
    sources = list(set(langs) - {target})
    df_target = filter_data(df, target, sources)
    out_file = os.path.join(out_dir, target + "_4-fold.tsv")
    iterative_split(df_target, out_file, target, 4)

