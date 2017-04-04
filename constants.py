import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
GLOVE_DIR = BASE_DIR + '/data/'
TEXT_DATA_DIR = BASE_DIR + '/data/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
INTENTIONS_DIR = BASE_DIR + '/data/comments'
RESULTS_DIR = BASE_DIR + '/results'

# Word2Vec related constants
WORD2VEC_BIN = 'ruwikiruscorpora_0_300_20.bin'
WORD2VEC_TXT = 'ruwikiruscorpora_0_300_20.txt'

EXCEL_NORMALIZED_DATASET_NAME = 'full_dataset_with_forms.xls'
