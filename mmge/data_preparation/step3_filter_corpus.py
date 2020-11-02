import ast
import numpy as np
import pandas as pd

from mmge.utils import utils
from mmge.utils.utils import langs

corpus_file = utils.RAW_CORPUS_FILE_PATH
out_file = utils.CORPUS_FILE_PATH

# Read raw corpus
df = pd.read_csv(corpus_file, index_col='id')

# Remove genres that do not appear at least 16 times in the corpus
while True:
    remove_flag = False
    # Temporarely save dataframe in a dictionary, on which the remove steps will be applied
    values = {}
    for lang in langs:
        values[lang] = []
        for annotations in df[lang].tolist():
            if str(annotations) == 'nan':
                values[lang].append([])
            else:
                values[lang].append(ast.literal_eval(str(annotations)))

    # Get the tags that appear less than 16 times
    selected_tags = utils.corpus_genres_per_lang(df, 16)
    # Update the annotations by removing the music genres that appear less than 16 times
    for lang in langs:
        for annotations in values[lang]:
            to_remove = set(annotations) - selected_tags[lang]
            if len(to_remove) > 0:
                remove_flag = True
            for g in to_remove:
                print(lang, g)
                annotations.remove(g)

    # Recreate dataframe after the removal of the selected tags
    filtered_df = pd.DataFrame(values, index=df.index)
    for lang in langs:
        filtered_df[lang] = filtered_df[lang].apply(lambda l: np.nan if len(l) == 0 else l)
    # After each new genre removal, make sure that each music item still has music genres in at least 2 languages
    filtered_df = filtered_df.dropna(thresh=2)
    # Update the dataframe
    df = filtered_df
    # Break if no music genre was removed in this iteration
    if not remove_flag:
        break

# Save the filtered dataset
df.to_csv(out_file, index=True, index_label="id")
