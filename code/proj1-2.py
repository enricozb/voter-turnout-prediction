from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn import svm
from utils.parse import extract

from random import shuffle

print('Loading Data.')
data = extract('data/train_2008.csv', 'PES1')
shuffle(data)
X, Y = zip(*data)
X = normalize(X)

print('Training SVM.')

N = 50000
clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
                           min_samples_split=11, random_state=0)
clf.fit(X, Y)
print(' ' * 4 + 'For mss = 11')
print(' ' * 8 + 'e_in = {}'.format(clf.score(X[:N], Y[:N])))
print(' ' * 8 + 'eout = {}'.format(clf.score(X[N:], Y[N:])), flush=True)
with open('out.txt', 'w') as f:
    with open('data/test_2008.csv') as test:
        test.readline()
        for line in test:
            x = list(map(float,line.split(',')))
            v = clf.predict([x[1:]])
            f.write('{},{}\n'.format(int(x[0]), int(v)))

quit()

for i in range(10, 20):
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
                               min_samples_split=i, random_state=0,
                               criterion='entropy')
    N = 50000
    clf.fit(X[:N], Y[:N])
    print(' ' * 4 + 'For mss = {}'.format(i))
    print(' ' * 8 + 'e_in = {}'.format(clf.score(X[:N], Y[:N])))
    print(' ' * 8 + 'eout = {}'.format(clf.score(X[N:], Y[N:])), flush=True)
quit()

clf = svm.SVC()
clf.fit(normalize(X[:2000]), Y[:2000])

print(clf.score(X, Y))
