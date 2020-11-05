import os
import uuid

from string import Template
from SPARQLWrapper import SPARQLWrapper, JSON

from mmge.utils import utils
from mmge.utils.utils import langs

opj = os.path.join

# DBpedia music artists and works are mapped on unique identifiers
ent_ids = {}
# For each language, the entities are saved together with their aliases in the other languages
ent_per_lang = {}
for lang in langs:
    ent_per_lang[lang] = set()


def get_relevant_music_entities(query_template, lang, ent_types=['MusicalWork', 'MusicalArtist', 'Band']):
    """ Collect DBpedia music artists and their works
    :param query_template: the SPARQL query template to be executed
    :param lang: the language of DBpedia to be crawled
    :param ent_types: the types of entities to be crawled
    """
    if lang not in langs:
        raise Exception('Language not tested.')
    other_langs_cond = utils.get_alias_filter(lang, langs)
    query_params = {}
    query_params['other_lang_cond'] = other_langs_cond
    endpoint = utils.get_endpoint_for_lang(lang)
    for ent in ent_types:
        print("Entity type: ", ent)
        query_params['entity_type'] = ent
        get_dbp_ents(endpoint, query_template, query_params, lang)


def get_dbp_ents(endpoint, query_template, query_params, lang):
    """Query DBpedia and process the returned entities
    :param endpoint: the DBpedia endpoint (for a specific language)
    :param query_template: the SPARQL query template
    :param query_params: the parameters of the queries such as the entity type
    :param lang: the language under consideration
    """
    sparql_dbpedia = SPARQLWrapper(endpoint + "sparql")
    sparql_dbpedia.setReturnFormat(JSON)
    offset = 0
    while (True):
        print('Offset :', offset)
        query = query_template.substitute({'entity': query_params['entity_type'], 'offset': offset, 'other_lang_cond': query_params['other_lang_cond']})
        #print(query)
        sparql_dbpedia.setQuery(query)
        results = sparql_dbpedia.query().convert()
        for result in results["results"]["bindings"]:
            other_lang = utils.get_lang(result["alias"]["value"])
            if other_lang not in langs:
                continue

            # Retrieve or generate a new uuid
            entity = result["entity"]["value"]
            if entity not in ent_ids:
                ent_ids[entity] = str(uuid.uuid4())

            # Map aliases on the same uuid as the original entity
            alias = result["alias"]["value"]
            ent_ids[alias] = ent_ids[entity]
            ent_per_lang[lang].add(entity)
            ent_per_lang[other_lang].add(alias)
        if len(results["results"]["bindings"]) < 10000:
            break
        offset += 10000



# Create the output dir if it does not exist
out_dir = utils.DATA_DIR
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print("Directory ", out_dir, " Created ")
else:
    print("Directory ", out_dir, " already exists")

# SPARQL query template
query_template = Template("""SELECT ?entity, ?alias{{
            SELECT ?entity, ?alias
            WHERE {
                ?entity rdf:type <http://dbpedia.org/ontology/$entity>.
                ?entity owl:sameAs ?alias.
                FILTER EXISTS {
                    ?entity <http://dbpedia.org/ontology/genre> ?genre
                }.
                FILTER ($other_lang_cond)
                }
            ORDER BY ?entity}}
            OFFSET $offset
            LIMIT 10000""")

for lang in langs:
    try:
        get_relevant_music_entities(query_template, lang)
    except Exception as ex:
        print('An exception was encountered when querying DBpedia in', lang)
        print(ex)

# Save DBpedia entities
# First their DBpedia URLs per language which will be used to collect their genres in the next step
for lang in ent_per_lang:
    with open(opj(out_dir, lang + '_entities.txt'), 'w') as _:
        for g in ent_per_lang[lang]:
            _.write(g + '\n')
# Second the mapping between each entity and the generated unique identifier
with open(opj(out_dir, 'musical_items_ids.csv'), 'w') as _:
    for ent in ent_ids:
        _.write(''.join([ent_ids[ent], '\t', ent, '\n']))
