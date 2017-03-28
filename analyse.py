import os
import numpy as np

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import helpers

from constants import GLOVE_DIR, WORD2VEC_TXT, MAX_NB_WORDS, \
    MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, EMBEDDING_DIM, EXCEL_NORMALIZED_DATASET_NAME, RESULTS_DIR

print('Processing text dataset')

if not os.path.exists(os.path.join(RESULTS_DIR, EXCEL_NORMALIZED_DATASET_NAME)):
    print('Create normalized dataset with forms...')
    texts_normalized, text_ids, labels, labels_names, labels_index = helpers.make_normalized_dataset()
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

unique_labels = set(labels_names)
print("Number of distinct classes: {}".format(len(unique_labels)))

print('Found %s texts.' % len(texts_normalized))

if not os.path.exists(os.path.join(GLOVE_DIR, WORD2VEC_TXT)):
    print('Create txt file from Russian binary word2vec file...')
    helpers.make_txt_word2vec_from_bin()

print('Indexing word vectors.')

# to limit memory allocation, we need to get only those embeddings that have
# counterparts in text for that, we need to:
# 1. make set of all words
# 2. take from embeddings store only necessary ones
unique_words = set([i for t in texts_normalized for i in set(t.split())])

embeddings_index = helpers.read_embeddings(unique_words)

print('Found %s word vectors.' % len(embeddings_index))

# now we need to make texts - texts and not list of words

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
tokenizer.fit_on_texts(texts_normalized)
sequences = tokenizer.texts_to_sequences(texts_normalized)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
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

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
text = Conv1D(128, 5, activation='relu')(embedded_sequences)
text = MaxPooling1D(5)(text)
text = Conv1D(128, 5, activation='relu')(text)
text = MaxPooling1D(5)(text)
text = Conv1D(128, 5, activation='relu')(text)
text = MaxPooling1D(35)(text)
text = Flatten()(text)
text = Dense(128, activation='relu')(text)
preds = Dense(len(labels_index), activation='softmax')(text)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

NUMBER_OF_EPOCHS = 2
# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=NUMBER_OF_EPOCHS, batch_size=128)

model.save('results/russian_{}_ep.h5'.format(NUMBER_OF_EPOCHS))

print('Saved model.')
