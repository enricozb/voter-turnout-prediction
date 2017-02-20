import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from utils.parse                import extract, prune, prune_by_column
from sklearn.ensemble           import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection    import cross_val_score
from sklearn.preprocessing      import normalize
from sklearn.svm                import LinearSVC

import keras
from keras.utils.np_utils       import to_categorical
from keras.models               import Sequential
from keras.layers.core          import Dense, Activation, Dropout

print('Loading data.')
X_train, Y_train = extract('data/train_2008.csv', 'PES1', separate=True)
# X_test = extract('data/test_2008.csv')
Y_train = to_categorical(Y_train)

p = 0.9

X_t, pruned = prune(X_train, threshold=p, to_remove={0})
X_t = normalize(X_t)

print('Fitting Neural network with proportion {}'.format(p))
model = Sequential()
model.add(Dense(output_dim=500, input_dim=X_t.shape[1]))
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

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.fit(X_t[:50000], Y_train[:50000], batch_size=256, nb_epoch=10, verbose=0)
scores = model.evaluate(X_t[50000:], Y_train[50000:], verbose=0)
print(scores)
quit()