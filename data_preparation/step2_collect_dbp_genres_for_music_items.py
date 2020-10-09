import os
import pandas as pd
from string import Template
from SPARQLWrapper import SPARQLWrapper, JSON

import utils
from utils import langs

opj = os.path.join


def get_genres_for_entities(seeds, query_template, lang, ent_ids):
    """ Collect DBpedia genres for the previously collected artists and works
    :param seeds: the DBpedia URLs of the music items previously collected
    :param query_template: the SPARQL query template to be executed
    :param lang: the language of the DBpedia version to be queried
    :param ent_ids: the unique ids of the music items previously collected
    :return: seed music items with associated genres in the targetted language
    """
    if lang not in langs:
        raise Exception('Language not tested. It may require modifications of DBpedia entity names')
    print("Language, ", lang)
    endpoint = utils.get_endpoint_for_lang(lang)
    sparql_dbpedia = SPARQLWrapper(endpoint + "sparql")
    sparql_dbpedia.setReturnFormat(JSON)
    entities_with_genres = {}

    start = 0
    while start < len(seeds):
        if lang == 'ja':
            end = start + 50
        else:
            end = start + 100
        if end > len(seeds):
            end = len(seeds)
        print("Processing next 100 entities... ", start, end)

        list_genres_str = utils.get_seeds_filter(seeds[start:end])
        start = end
        query = query_template.substitute({'list': list_genres_str})
        #print(query)
        sparql_dbpedia.setQuery(query)

        results = sparql_dbpedia.query().convert()
        for result in results["results"]["bindings"]:
            entity = result["entity"]["value"]
            ent_id = ent_ids[entity]
            if ent_id not in entities_with_genres:
                entities_with_genres[ent_id] = []
            genre = result["genre"]["value"]
            entities_with_genres[ent_id].append(genre)

    return entities_with_genres


def create_and_save_df(entities):
    """ Create a dataframe from the data and save it
    :param entities: music items with their genres per language
    """
    data = list(entities.values())
    # Set index to be the list of languages
    df = pd.DataFrame(data, index=langs).T
    # Drop the music items which do not have genres in at least 2 languages
    df = df.dropna(thresh=2)
    df.to_csv(opj(data_dir, "musical_items.csv"), index=True, index_label="id")


data_dir = utils.DATA_DIR
# Load mapping between music items and their unique ids
ent_ids = {}
with open(opj(data_dir, 'musical_items_ids.csv'), 'r') as _:
    lines = [line.replace('\n', '') for line in _.readlines()]
    for line in lines:
        data = line.split('\t')
        ent_ids[data[1]] = data[0]

# Load music items per language
ent_per_lang = {}
for lang in langs:
    with open(opj(data_dir, lang + '_entities.txt'), 'r') as _:
        lines = [line.replace('\n', '') for line in _.readlines()]
        ent_per_lang[lang] = lines

# Query template for retrieving the genres of the entities
query_template_genre = Template("""SELECT ?entity, ?genre
                WHERE {
                ?entity <http://dbpedia.org/ontology/genre> ?genre
                FILTER (?entity IN ($list))
                }
            """)

# Collect music genres for the DBpedia music items
entities_with_genres = {}
for lang in langs:
    entities_with_genres[lang] = get_genres_for_entities(list(ent_per_lang[lang]), query_template_genre, lang, ent_ids)

# Save the corpus
create_and_save_df(entities_with_genres)
