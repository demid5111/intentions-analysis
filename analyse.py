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

from constants import *

print('Processing text dataset')

texts_normalized, text_ids, labels, labels_names, labels_index = helpers.make_normalized_dataset()
helpers.write_dataset_to_file(labels_names=labels_names,
                              labels=labels,
                              text_ids=text_ids,
                              texts=texts_normalized,
                              fileName="full_dataset_with_forms")


unique_labels = set(labels_names)
print("Number of distinct classes: {}".format(len(unique_labels)))

print('Found %s texts.' % len(texts_normalized))

if not os.path.exists(os.path.join(GLOVE_DIR,WORD2VEC_TXT)):
    print('Create txt file from Russian binary word2vec file...')
    helpers.make_txt_word2vec_from_bin()

print('Indexing word vectors.')

embeddings_index = helpers.read_embeddings()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
# print('Processing text dataset')
#
# texts = []  # list of text samples
# labels_index = {}  # dictionary mapping label name to numeric id
# labels = []  # list of label ids
# for name in sorted(os.listdir(TEXT_DATA_DIR)):
#     path = os.path.join(TEXT_DATA_DIR, name)
#     if os.path.isdir(path):
#         label_id = len(labels_index)
#         labels_index[name] = label_id
#         for fname in sorted(os.listdir(path)):
#             if fname.isdigit():
#                 fpath = os.path.join(path, fname)
#                 if sys.version_info < (3,):
#                     f = open(fpath)
#                 else:
#                     f = open(fpath, encoding='latin-1')
#                 texts.append(f.read())
#                 f.close()
#                 labels.append(label_id)
#
# print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

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

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)

model.save('results/my_model.h5')

print('Saved model.')
