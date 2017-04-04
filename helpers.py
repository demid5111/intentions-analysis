import csv
import os
import pandas as pd
import subprocess
import json
import re
import numpy as np
from keras.engine import Input
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from constants import RESULTS_DIR, GLOVE_DIR, WORD2VEC_BIN, WORD2VEC_TXT, INTENTIONS_DIR, EMBEDDING_DIM, \
    MAX_SEQUENCE_LENGTH, MAX_NB_WORDS

class ANALYSIS_MODE:
    MIXED = 0
    ONLY_DIGITS = 1
    ONLY_LETTERS = 2
    GENERALIZE_LETTERS = 3

def read_russian_csv(base_dir):
    russian_dataset = []
    for folder in os.listdir(base_dir):
        new_path = os.path.join(base_dir, folder)
        if not os.path.isdir(new_path):
            continue
        print('Reading folder: {} ...'.format(folder))
        for fileName in os.listdir(new_path):
            if not fileName.endswith('.csv'):
                continue
            with open(os.path.join(new_path, fileName), newline='', encoding='windows-1251') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=';')
                for row in spamreader:
                    if row[7] == 'text':
                        continue
                    russian_dataset.append({'label_name': row[-2].split(',')[0][:2].lower(),
                                            'text': row[7],
                                            'text_id': row[2]})
    return russian_dataset


def get_label_name(el):
    return el['label_name'].strip()


def get_text(el):
    return el['text'].strip()


def get_text_id(el):
    return el['text_id'].strip()


def write_dataset_to_file(labels_names=(), labels=(), text_ids=(), texts=(), fileName='full_dataset'):
    """
    Writes the dataset into the Excel file with the structure: label_names|label_id|comment_id|text
    :param labels_names:
    :param labels:
    :param text_ids:
    :param texts:
    """
    df = pd.DataFrame({
        'Label_names': labels_names,
        'Label_ID': labels,
        'Comment_ID': text_ids,
        'Text': texts
    })

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join(RESULTS_DIR, fileName + '.xlsx'), engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Intentions')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def preprocess_raw_dataset(raw_dataset):
    labels_index = {}
    texts = []  # list of text samples
    labels = []
    text_ids = []
    labels_names = []
    for x in raw_dataset:
        label = get_label_name(x)
        try:
            label_id = labels_index[label]
        except KeyError:
            label_id = len(list(labels_index.keys()))
            labels_index[label] = label_id
        texts.append(get_text(x))
        text_ids.append(get_text_id(x))
        labels.append(label_id)
        labels_names.append(label)
    return texts, text_ids, labels, labels_names, labels_index


def _call_mystem(text):
    import random
    i = random.randrange(100000)
    with open('tmp/{}.txt'.format(i), 'w') as f:
        f.write(text)
    command = './lib/mystem -cgin --format json tmp/{}.txt'.format(i)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    try:
        process.wait(endtime=150000)
    except subprocess.TimeoutExpired:
        print("Attention timeout!!!")
    res = [line.decode('utf-8') for line in process.stdout]
    # print('Result ', res)
    return res


def _parse_mystem_output(output_list):
    result_data = []
    for line in output_list:
        try:
            data = json.loads(line)
            if type(data) is not dict or not 'analysis' in list(data.keys()):
                continue
            result_data.append(data)
        except json.decoder.JSONDecodeError:
            pass
    return result_data


def _compile_normalized_text(mystem_json_list):
    stem_forms_dic = {
        'A': 'ADJ',
        'ADV': 'ADV',
        'ADVPRO': 'ADV',
        'ANUM': 'ADJ',
        'APRO': 'DET',
        'COM': 'ADJ',
        'CONJ': 'SCONJ',
        'INTJ': 'INTJ',
        'NONLEX': 'X',
        'NUM': 'NUM',
        'PART': 'PART',
        'PR': 'ADP',
        'S': 'NOUN',
        'SPRO': 'PRON ',
        'UNKN ': 'X',
        'V': 'VERB'
    }
    res_list = []
    for x in mystem_json_list:
        if len(x['analysis']) != 0:
            norm_form = x['analysis'][0]['lex']
            form_name = list(filter(None, re.split('[,=]+', x['analysis'][0]['gr'])))[0]
            translated_form_name = stem_forms_dic[form_name]
            res_list.append('{}_{}'.format(norm_form, translated_form_name))
    return res_list


def normalize_dataset(texts):
    from multiprocessing import Pool as ThreadPool
    from multiprocessing import cpu_count
    print("Number of threads: {}".format(cpu_count()))
    chunk_size = 1000
    total_number_chunks = round(len(texts) / chunk_size)
    texts_normalized = []
    for i in range(0, total_number_chunks + 1):
        print("Analysing chunk: %d/%d   \r" % (i + 1, total_number_chunks))
        pool = ThreadPool(cpu_count())
        tmp = pool.map(compose_for_normalize, texts[i * chunk_size:(i + 1) * chunk_size])
        pool.close()
        pool.join()
        texts_normalized.extend(tmp)
    return texts_normalized


def compose_for_normalize(text):
    new_line = text.replace('\\', '')
    new_line = new_line.replace('!', '')
    new_line = new_line.replace('\n', ' ')
    new_line = new_line.replace('\r', ' ')
    new_line = new_line.replace('\t', ' ')
    output = _call_mystem(new_line)
    return _compile_normalized_text(_parse_mystem_output(output))


def make_txt_word2vec_from_bin():
    from gensim.models import KeyedVectors

    model = KeyedVectors.load_word2vec_format(os.path.join(GLOVE_DIR, WORD2VEC_BIN), binary=True)
    model.save_word2vec_format(os.path.join(GLOVE_DIR, WORD2VEC_TXT), binary=False)


def read_embeddings(unique_words=()):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, WORD2VEC_TXT))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        if not len(unique_words) or (word in unique_words):
            embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def clean_up_raw_dataset(raw_dataset):
    return [x for x in raw_dataset if len(get_label_name(x)) == 2 and len(get_text(x)) != 0]


def make_normalized_dataset():
    raw_dataset = read_russian_csv(INTENTIONS_DIR)

    # remove uncategorised symbols
    cleaned_up_raw_dataset = clean_up_raw_dataset(raw_dataset)

    texts, text_ids, labels, labels_names, labels_index = preprocess_raw_dataset(cleaned_up_raw_dataset)

    res_normalized = normalize_dataset(texts)

    texts_normalized = [" ".join(x) for x in res_normalized]
    indexes_to_remove = [i for (i, x) in enumerate(texts_normalized) if not len(x)]
    to_subtract = 0
    for i in indexes_to_remove:
        index = i - to_subtract
        del text_ids[index]
        del labels[index]
        del labels_names[index]
        del texts_normalized[index]
        to_subtract += 1

    extra_labels = set(list(labels_index.keys())) - set(labels_names)
    for label in extra_labels:
        del labels_index[label]

    return texts_normalized, text_ids, labels, labels_names, labels_index


def make_letters_class_map(labels_names):
    labels_letters_index = {}
    for label in labels_names:
        label_name = label[0]
        if label_name not in list(labels_letters_index.keys()):
            labels_letters_index[label_name] = len(labels_letters_index)

    labels_letter = []
    labels_digit_names = []
    for i in labels_names:
        label_name = i[0]
        label_digit = int(i[1]) - 1
        labels_letter.append(labels_letters_index[label_name])
        labels_digit_names.append(label_digit)
    return labels_letter, labels_digit_names, labels_letters_index


def read_normalized_dataset(file_name):
    import pandas
    f = pandas.read_excel(file_name, sheetname='Intentions')
    texts_normalized = list(f['Text'])
    labels_names = list(f['Label_names'])
    labels = list(f['Label_ID'])
    text_ids = list(f['Comment_ID'])
    labels_index = {}
    new_labels = []
    new_id = 0
    for (label, label_id) in zip(labels_names, labels):
        if label not in labels_index:
            labels_index[label] = new_id
            new_id += 1
        new_labels.append(labels_index[label])
    return texts_normalized, text_ids, new_labels, labels_names, labels_index


def generalize_labels(labels_names):
    """
    If you want to cut the number of classes to make better training results
    specify rules for translating detailed classes to general super-classes
    make sure you have letters typed in the target language
    :param labels_names: list of labels names from the dataset
    """
    generalization_rule = {"Оптимистическая интенция": ["а", "б", "в", "г", "д", "е", "ж", "з"],
                           "Критическая интенция": ["и", "к", "л", "м", "н", "о", "п", "р", "с", "т"],
                           "Пессимистическая интенция": ["у", "ф", "х", "ц", "ч", "ш"],
                           "Неопределенная интенция": ["щ", "э", "ю", "я", "l", "w", "r", "u", "i", "s", "d", "f", "z"]}

    new_labels_index = {}
    for key in generalization_rule.keys():
        new_labels_index[key] = len(new_labels_index)

    new_labels = []
    new_labels_ids = []
    for label in labels_names:
        new_name = ""
        for (key, value) in generalization_rule.items():
            # now we just check the first letter of the class to find it in the values list
            # in generalization_rule
            if label[0] in value:
                new_name = key
        new_labels.append(new_name)
        new_labels_ids.append(new_labels_index[new_name])
    return new_labels_ids, new_labels, new_labels_index


def construct_lstm(nb_words, class_num, embedding_matrix):
    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    text = LSTM(64, return_sequences=True)(embedded_sequences)
    text = LSTM(64)(text)
    text = Dropout(0.5)(text)
    text = Dense(class_num)(text)
    return sequence_input, Activation('sigmoid')(text)


def construct_cnn(nb_words, class_num, embedding_matrix):
    # load pre-trained word embeddings into an Embedding layer
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    text = Conv1D(128, 5, activation='relu')(embedded_sequences)
    text = MaxPooling1D(5)(text)
    text = Conv1D(128, 5, activation='relu')(text)
    text = MaxPooling1D(5)(text)
    text = Conv1D(128, 5, activation='relu')(text)
    text = MaxPooling1D(15)(text)
    text = Flatten()(text)
    text = Dense(128, activation='relu')(text)
    return sequence_input, Dense(class_num, activation='softmax')(text)

def create_embedding_matrix(word_index, texts_normalized):
    # to limit memory allocation, we need to get only those embeddings that have
    # counterparts in text for that, we need to:
    # 1. make set of all words
    # 2. take from embeddings store only necessary ones
    unique_words = set([i for t in texts_normalized for i in set(t.split())])
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    embeddings_index = read_embeddings(unique_words)

    print('Found %s word vectors.' % len(embeddings_index))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return nb_words, embedding_matrix

def tokenize_texts(texts_normalized):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(texts_normalized)
    sequences = tokenizer.texts_to_sequences(texts_normalized)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data, word_index