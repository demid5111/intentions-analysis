import os
import numpy as np

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model
from keras.layers.recurrent import LSTM
from sklearn.metrics import classification_report

import helpers

from constants import GLOVE_DIR, WORD2VEC_TXT, MAX_NB_WORDS, \
    MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM, EXCEL_NORMALIZED_DATASET_NAME, RESULTS_DIR

print('Processing text dataset')

if not os.path.exists(os.path.join(RESULTS_DIR, EXCEL_NORMALIZED_DATASET_NAME)):
    print('Create normalized dataset with forms...')
    texts_normalized, text_ids, labels, labels_names, labels_index = helpers.make_normalized_dataset()
    labels_letter, labels_digit_names = helpers.make_letters_class_map(labels_names)
    helpers.write_dataset_to_file(labels_names=labels_names,
                                  labels=labels,
                                  text_ids=text_ids,
                                  texts=texts_normalized,
                                  fileName="full_dataset_with_forms")
else:
    file_name = os.path.join(RESULTS_DIR, EXCEL_NORMALIZED_DATASET_NAME)
    print('Read normalized dataset from file {}...'.format(file_name))
    texts_normalized, text_ids, labels, labels_names, labels_index = helpers.read_normalized_dataset(
        file_name=file_name)
    labels_letter, labels_digit_names, labels_letters_index = helpers.make_letters_class_map(labels_names)

CURRENT_MODE = helpers.ANALYSIS_MODE.GENERALIZE_LETTERS

if CURRENT_MODE == helpers.ANALYSIS_MODE.GENERALIZE_LETTERS:
    group_labels_ids, group_labels, group_labels_index = helpers.generalize_labels(labels_names)
    labels_for_training = group_labels_ids
    labels_to_id_dic = group_labels_index
elif CURRENT_MODE == helpers.ANALYSIS_MODE.MIXED:
    labels_for_training = labels
    labels_to_id_dic = labels_index
elif CURRENT_MODE == helpers.ANALYSIS_MODE.ONLY_LETTERS:
    labels_for_training = labels_letter
    labels_to_id_dic = labels_letters_index
elif CURRENT_MODE == helpers.ANALYSIS_MODE.ONLY_DIGITS:
    labels_for_training = labels_digit_names
    labels_to_id_dic = {str(i):i for i in set(labels_digit_names)}

print('Found %s texts.' % len(texts_normalized))

if not os.path.exists(os.path.join(GLOVE_DIR, WORD2VEC_TXT)):
    print('Create txt file from Russian binary word2vec file...')
    helpers.make_txt_word2vec_from_bin()

print('Indexing word vectors.')

# finally, vectorize the text samples into a 2D integer tensor

data, word_index = helpers.tokenize_texts(texts_normalized=texts_normalized)
labels = to_categorical(np.asarray(labels_for_training))

# number of distinct classes
class_num = labels.shape[1]
print("Number of distinct classes: {}".format(class_num))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')


nb_words, embedding_matrix = helpers.create_embedding_matrix(word_index=word_index, texts_normalized=texts_normalized)

print('Training model.')

# sequence_input, preds = helpers.construct_cnn(nb_words=nb_words, class_num=class_num, embedding_matrix=embedding_matrix)

sequence_input, preds = helpers.construct_lstm(nb_words=nb_words, class_num=class_num, embedding_matrix=embedding_matrix)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

NUMBER_OF_EPOCHS = 5
# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=NUMBER_OF_EPOCHS, batch_size=200)

model.save('results/russian_{}_ep.h5'.format(NUMBER_OF_EPOCHS))
print('Saved model.')

model = load_model('results/russian_{}_ep.h5'.format(NUMBER_OF_EPOCHS))
print('Loaded model.')

label_list = [0 for i in range(len(list(labels_to_id_dic.keys())))]
for (key, value) in labels_to_id_dic.items():
    try:
        label_list[value] = key
    except IndexError:
        print(value, key)

y_est = model.predict(x_val)
print(classification_report(np.argmax(y_val, axis=1), np.argmax(y_est, axis=1), target_names=label_list))
