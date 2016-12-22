import nltk
import numpy as np
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from reader import Read

x = ['hello.', 'what is your name?', 'how are you?', 'i feel sick.', 'who is the president elect?']
y = ['hello.', 'my name is jimmy.', 'i am great.', 'you need to lie down.', 'Donald Trump is the next president.']

x = [nltk.word_tokenize(word.lower()) for word in x]
y = [nltk.word_tokenize(word.lower()) for word in y]

vocab, ivocab, embed = Read("C:/Users/17twa/Documents/glove.6B.100d.txt")

MAXLEN = 7
n_in_out = 100

#n_in and n_out are the dimensions of each element in the time series
n_in = 100
n_out = 100

#Hidden layer dim
n_hidden = 1000
n_examples = len(x)

embedding_matrix = np.zeros((len(vocab), n_in_out), dtype=float)
for idx, word in enumerate(vocab):
    embedding_matrix[idx] = np.array(list(embed[word]), dtype=float)

model = Sequential()
# `return_sequences` controls whether to copy the input automatically
model.add(Embedding(len(vocab), n_in_out, weights=[embedding_matrix], input_length=MAXLEN, mask_zero=True))
model.add(GRU(n_hidden, return_sequences=True))
#model.add(GRU(n_hidden, input_dim=n_in_out, input_length=MAXLEN, return_sequences=True))
model.add(TimeDistributedDense(n_in_out, input_dim = n_hidden))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

X = x
Y = []
embed['<PAD>'] = [0.0]*n_in_out
for idx, sent in enumerate(X):
    if len(sent) < MAXLEN:
        X[idx] += ['<PAD>']*(MAXLEN-len(sent))
for idx, sent in enumerate(y):
    if len(sent) < MAXLEN:
        y[idx] += ['<PAD>']*(MAXLEN-len(sent))
for idx, sent in enumerate(X):
    new_sent = []
    for word in sent:
        new_sent += [ivocab[word]]
    X[idx] = list(new_sent)
for idx, sent in enumerate(y):
    new_sent = []
    for word in sent:
        new_sent += [embed[word]]
    Y += [list(new_sent)]
X = np.array(X)
Y = np.array(Y)

Xp = model.predict(X)
print Xp.shape
print Y.shape

def closest_word(vec):
    vec=np.array(vec.tolist())
    N = 2
    vec_result = vec
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(embedding_matrix, vec_norm.T)

    a = np.argsort(-dist)[:N]
    idx = a[0]
    if idx == 0:
        return '<PAD>'
    else:
        return vocab[idx-1]

def decode_indices(idx_list):
    sent = []
    for idx in idx_list:
        if idx == 0:
            sent += ['<PAD>']
        else:
            sent += [vocab[idx-1]]
    return sent

def decode_vec(vec):
    sent = []
    for row in vec:
        sent += [closest_word(row)]
    return sent

model.fit(X, Y, nb_epoch=50)

embedding_matrix = np.array(([np.zeros(n_in_out, dtype=float)] + list(embedding_matrix)), dtype=float)

print('Testing...')
for test_input in X:
    pred = model.predict(np.array([test_input], dtype=float))
    print('Input:', decode_indices(test_input))
    print('Output:', decode_vec(pred[0]))
    print('.'*50)

while True:
    text = raw_input('Enter input (EXIT to break): ')
    if text == 'EXIT':
        break
    else:
        X = nltk.word_tokenize(text)
        if len(X) > MAXLEN:
            print('Error: Sentence too long.')
        else:
            if len(X) < MAXLEN:
                X += ['<PAD>']*(MAXLEN-len(X))
            new_sent = []
            for word in X:
                new_sent += [ivocab[word]]
            X = list(new_sent)
            pred = model.predict(np.array([X], dtype=float))
            print('Output:', decode_vec(pred[0]))
            print('.'*50)