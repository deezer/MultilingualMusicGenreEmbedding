from setuptools import setup, find_packages

setup(name='mmge',
      description='Multilingual Music Genre Embeddings for Effective Cross-Lingual Music Item Annotation',
      author='Deezer Research',
      install_requires=['numpy==1.17.2',
                        'pandas==0.25.1',
                        'sklearn==0.0',
                        'networkx==2.2',
                        'joblib==0.13.2',
                        'torch==1.4.0',
                        'SPARQLWrapper==1.8.4',
                        'spacy==2.2.2'],
      package_data={'mmge': ['README.md']},
      packages=find_packages())
