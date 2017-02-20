import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from collections import defaultdict, Counter

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical

from random import randrange, choice

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

class Ensemble:
    def __init__(self, input_dim, output_dim):
        self.nns = []
        for i in range(10):
            nn = NeuralNetwork(input_dim, output_dim)
            nn.randomize(6, (300, 600))
            nn.compile()
            self.nns.append(nn)

        self.rf = RandomForest()
        self.et = ExtraTrees()

        self.residual = SVC()

    def fit(self, X, Y):
        print('Fitting models.')
        X = normalize(X)
        Yc = to_categorical(Y)

        print('Fitting Neural Networks.')
        for i, nn in enumerate(self.nns):
            print(' ' * 4 + 'Fitting Neural Network {}.'.format(i))
            nn.fit(X, Yc)

        print('Fitting Random Forest.')
        self.rf.fit(X[:15000], Y[:15000])

        print('Fitting Extra Trees.')
        self.et.fit(X[:15000], Y[:15000])

        # Nonsense 101
        with open('residual.txt') as f:
            X = []
            Y = []
            for line in f:
                y, x = eval(line)
                Y.append(int(y))
                X.append(x)
        X = normalize(X)
        self.residual.fit(X, Y)

    def predict(self, X, Y):
        X = normalize(X)
        out = []
        p1 = 0
        for nn in self.nns:
            p1 += nn.predict(X)

        p2 = self.rf.predict(X)
        p3 = self.et.predict(X)
        for x, y, z, k in zip(p1, p2, p3, Y):
            x1, x2, x3 = 0, 0, 0
            x1 = max(enumerate(x), key=lambda x: x[1])[0]
            x2 = int(y)
            x3 = int(z)
            v = Counter([x1] + 4 * [x2] + 6 * [x3]).most_common(1)[0][0]

            if k != v and k in (x1, x2, x3):
                print("y: {}, y': {}, ensemble: {}, nn: {}".format(k, (x1, x2, x3), v, x))
            out.append(v)
        return out

class NeuralNetwork:
    '''
    Neural network for classification
    '''
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = Sequential()
        self.layers = [(input_dim, 'input')]

    def randomize(self, num_layers, neuron_range):

        action = ['relu', 'tanh']

        for l in range(num_layers - 1):
            layer_size = randrange(*neuron_range)
            layer_action = choice(action)

            if l == 0:
                dense = Dense(output_dim=layer_size, input_dim=self.input_dim)
            else:
                dense = Dense(output_dim=layer_size)
            self.model.add(dense)
            self.model.add(Activation(choice(action)))
            self.model.add(Dropout(0.2))
            self.layers.append((layer_size, action))

        self.model.add(Dense(self.output_dim))
        self.model.add(Activation('softmax'))

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    def fit(self, X, Y):
        self.model.fit(X, Y, batch_size=128, nb_epoch=10, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

class RandomForest:
    '''
    Random Forest for classification
    '''
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                           min_samples_split=15, random_state=0)

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)

class ExtraTrees:
    def __init__(self):
        self.clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
                           min_samples_split=15, random_state=0)

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)
