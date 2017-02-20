from utils.parse import extract
from sklearn.preprocessing import normalize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.engine.topology import Merge

from random import random

print('Loading data.')
data = extract('data/train_2008.csv', 'PES1')

def generate(X_, Y_, X):
    model = Sequential()
    model.add(Dense(output_dim=500, input_dim=382))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    model.fit(X_, Y_, batch_size=256, nb_epoch=10, verbose=0)
    return model.predict(normalize(X))

generate()

N = 1
chance = 0.3
prob = 0
X, Y = zip(*data)
for i in range(N):
    print('Fitting {}'.format(i))
    d = list(filter(lambda x: random() < chance, data))
    X_, Y_ = zip(*d)
    X_ = normalize(X_)
    Y_ = to_categorical(Y_)

    prob += generate(X_, Y_, X)
print('Models fit.')

print('Computing Error.')
count = 0
Y = to_categorical(Y)
for y1, y2  in zip(prob, Y):
    i1 = max(enumerate(y1), key=lambda x: x[1])[0]
    i2 = max(enumerate(y2), key=lambda x: x[1])[0]
    if i1 != i2:
        count += 1

print('Test error: {}'.format(1 - count/len(Y)))