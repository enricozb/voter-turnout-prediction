from utils.parse import extract, parse
from sklearn.preprocessing import normalize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

print('Loading data.')
X, Y = extract('data/train_2008.csv', 'PES1', separate=True)
X = normalize(X)
Y = to_categorical(Y)

