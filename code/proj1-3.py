from utils.parse import extract, prune
from models import Ensemble
from random import shuffle

print('Loading Data.')
data = extract('data/train_2008.csv', 'PES1')
shuffle(data)
X, Y = zip(*data)
Y = list(Y)
for i, y in enumerate(Y):
    Y[i] = int(y - 1)
N = 50000

ensemble = Ensemble(len(X[0]), 2)
ensemble.fit(X[:N], Y[:N])
Y_prediction = ensemble.predict(X[N:], Y[N:])

count = 0
for y1, y2 in zip(Y_prediction, Y[N:]):
    if y1 != y2:
        count += 1
print('Test Accuracy: {}'.format(1 - count/len(Y_prediction)))