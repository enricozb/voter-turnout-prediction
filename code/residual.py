from sklearn.svm import SVC
from sklearn.preprocessing import normalize

with open('residual.txt') as f:
    X = []
    Y = []
    for line in f:
        y, x = eval(line)
        Y.append(int(y))
        X.append(x)

clf = SVC()
X = normalize(X)
clf.fit(X[:100], Y[:100])
print(clf.score(X[100:], Y[100:]))
