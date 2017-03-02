import nltk
import numpy as np
from keras.layers import Dense, RepeatVector
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import TimeDistributedDense, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from reader import Read

MAXLEN = 9
'''with open('dialog-babi-task1-API-calls-dev.txt') as f:
    doc = f.read()
doc = doc.split('\n')
doc = [sent.split('\t') for sent in doc]
x = []
y = []
print('before maxlen reduction', len(doc))
for pair in doc:
    if len(pair) == 2:
        x1 = nltk.word_tokenize(pair[0].lower())
        y1 = nltk.word_tokenize(pair[1].lower())
        if len(x1) <= MAXLEN and len(y1) <= MAXLEN:
            x += [x1[1:]]
            y += [y1]
print('after maxlen reduction')
print('len(x)', len(x))
print('len(y)', len(y))'''

x = ['hello.', 'what is your name?', 'how are you?', 'what is your favorite color?', 'what do you like to do?', 'where do you live?', 'who are you?']
y = ['hello.', 'my name is jimmy.', "i'm good.", 'my favorite color is blue.', 'i like to party.', 'i live in Boston.', 'i am jimmy.']

x += ['what is the capital of Germany?', 'what is the capital of France?', 'what is the capital of Spain?', 'what is the capital of China?', 'what is the capital of Sweden?', 'what is the capital of Italy?', 'what is the capital of Britain?', 'what is the capital of Russia?', 'what is the capital of Australia?', 'what is the capital of Portugal?', 'what is the capital of Norway?']
y += ['Berlin.', 'Paris.', 'Madrid.', 'Beijing.', 'Stockholm.', 'Rome.', 'London.', 'Moscow.', 'Sydney.', 'Lisbon.', 'Oslo.']

x += ['who is the president?', 'what is Microsoft?', 'what is Apple?', 'what is Google?', 'what is an apple?', 'what is a peach?', 'what are green beans?', 'who was the first president?', 'who was the sixteenth president?']
y += ['Donald Trump is the president.', 'Microsoft is a company.', 'Apple is a company.', 'Google is a company.', 'An apple is a fruit.', 'A peach is a fruit.', 'Green beans are vegetables.', 'George Washington was the first president.', 'Abraham Lincoln was the sixteenth president?']

x += ['bye.', 'what is two plus two?', 'when was the revolutionary war?', 'when was the war of 1812?', 'what is the purpose of life?', 'can a cat fly?', 'does a tree have a tail?', 'what color is water?', 'what color is the sky?', 'what color is a cherry?', 'what color is a banana?', 'what is the purpose of living?', 'what is the purpose of dying?', 'who is luke skywalker?']
y += ['bye.', 'four.', '1776.', '1812.', 'to serve the greater good.', 'no.', 'no.', 'blue.', 'blue.', 'red.', 'yellow.', 'to live forever.', 'to have a life.', 'he is a hero.']

x = [nltk.word_tokenize(word.lower()) for word in x]
y = [nltk.word_tokenize(word.lower()) for word in y]

vocab, ivocab, embed = Read("file_name_here")
embed['<PAD>'] = [0.0]*100

done = False
while not done:
    checked = False
    for sent in x:
        if len(sent) > MAXLEN:
            x.remove(sent)
            checked = True
    for sent in y:
        if len(sent) > MAXLEN:
            y.remove(sent)
            checked = True
    if checked == False:
        done = True


n_in_out = 100
#n_in and n_out are the dimensions of each element in the time series
n_in = 100
n_out = 100

#Hidden layer dim
n_hidden = 256
LAYERS = 2
n_examples = len(x)

embedding_matrix = np.zeros((len(vocab), n_in_out), dtype=float)
for idx, word in enumerate(vocab):
    embedding_matrix[idx] = np.array(list(embed[word]), dtype=float)

model = Sequential()
# `return_sequences` controls whether to copy the input automatically
model.add(Embedding(len(vocab), n_in_out, weights=[embedding_matrix], input_length=MAXLEN, mask_zero=False))
model.add(LSTM(n_hidden, return_sequences=True))
for i in range(LAYERS-1):
    model.add(LSTM(n_hidden))
model.add(Dense(100))
model.add(RepeatVector(MAXLEN))
for i in range(LAYERS):
    model.add(LSTM(n_hidden, return_sequences=True))
model.add(TimeDistributedDense(n_in_out, input_dim = n_hidden))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

X = x
Y = []
for idx, sent in enumerate(X):
    if len(sent) < MAXLEN:
        X[idx] += ['<PAD>']*(MAXLEN-len(sent))
for idx, sent in enumerate(y):
    if len(sent) < MAXLEN:
        y[idx] += ['<PAD>']*(MAXLEN-len(sent))
for idx, sent in enumerate(X):
    new_sent = []
    for word in sent:
        if word in ivocab:
            new_sent += [ivocab[word]]
        else:
            new_sent += [ivocab['unk']]
    X[idx] = list(new_sent)
for idx, sent in enumerate(y):
    new_sent = []
    for word in sent:
        if word in embed:
            new_sent += [embed[word]]
        else:
            new_sent += [embed['unk']]
    Y += [list(new_sent)]

X = np.array(X)
Y = np.array(Y)
print('x len', len(X))
print('y len', len(Y))
print('x shape', X.shape)
print('y shape', Y.shape)


Xp = model.predict(X)
print('MAXLEN', MAXLEN)
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
    return vocab[idx-1]

def decode_indices(idx_list):
    sent = []
    for idx in idx_list:
        sent += [vocab[idx]]
    return sent

def decode_vec(vec):
    sent = []
    for row in vec:
        sent += [closest_word(row)]
    return sent

model.fit(X, Y, nb_epoch=370)

embedding_matrix = np.array(([np.zeros(n_in_out, dtype=float)] + list(embedding_matrix)), dtype=float)

print('Testing...')
for test_input in X[:40]:
    pred = model.predict(np.array([test_input], dtype=float))
    print('Input:', decode_indices(test_input))
    print('Output:', decode_vec(pred[0]))
    print('.'*50)

while True:
    text = raw_input('Enter input (EXIT to break): ')
    if text == 'EXIT':
        print('Saving weights...')
        model.save_weights('bot_weights.h5')
        print('Done saving weights.')
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
