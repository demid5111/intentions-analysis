import csv
import os
import pandas as pd
import subprocess
import json
import re
import numpy as np
from constants import RESULTS_DIR, GLOVE_DIR, WORD2VEC_BIN, WORD2VEC_TXT, INTENTIONS_DIR


def read_russian_csv(base_dir):
    russian_dataset = []
    for folder in os.listdir(base_dir)[3:4]:
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
    command = 'echo "{}" | tee tmp/logfile.txt && ./lib/mystem -cgin --format json logfile.txt'.format(text)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    return [line.decode('utf-8') for line in process.stdout]


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
        'COM ': 'ADJ',
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


def normalize_dataset(texts, limit):
    texts_normalized = ['' for x in texts]
    for (i, text) in list(enumerate(texts))[:limit]:
        print('{}/{}...'.format(i, len(texts)))
        output = _call_mystem(text)
        texts_normalized[i] = _compile_normalized_text(_parse_mystem_output(output))
    return texts_normalized


def make_txt_word2vec_from_bin():
    from gensim.models import KeyedVectors

    model = KeyedVectors.load_word2vec_format(os.path.join(GLOVE_DIR,WORD2VEC_BIN), binary=True)
    model.save_word2vec_format(os.path.join(GLOVE_DIR,WORD2VEC_TXT), binary=False)


def read_embeddings():
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, WORD2VEC_TXT))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def clean_up_raw_dataset(raw_dataset):
    return [x for x in raw_dataset if len(get_label_name(x)) == 2]


def make_normalized_dataset():
    raw_dataset = read_russian_csv(INTENTIONS_DIR)

    # remove uncategorised symbols
    cleaned_up_raw_dataset = clean_up_raw_dataset(raw_dataset)

    texts, text_ids, labels, labels_names, labels_index = preprocess_raw_dataset(cleaned_up_raw_dataset)

    texts_normalized = normalize_dataset(texts, limit=10)

    return texts_normalized, text_ids, labels, labels_names, labels_index
