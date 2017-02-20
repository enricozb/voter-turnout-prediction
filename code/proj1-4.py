from utils.parse import extract

X, Y = extract('data/train_2008.csv', 'PES1', separate=True)

blanks = [0] * len(X[0])
for x in X:
    for i, x_ in enumerate(x):
        if x_ == -1:
            blanks[i] += 1

print('Pruning.')
blanks = list(map(lambda x: x/len(X), blanks))
prune = [0] # remove the Id field when training.
for i, p in enumerate(blanks):
    if p > 0.90:
        prune.append(i)

X_prune = []
for x in X:
    x_ = []
    for i, v in enumerate(x):
        if i not in prune:
            x_.append(v)
    X_prune.append(x_)

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

X_prune = normalize(X_prune)
print("TRAIN SHAPE:", X_prune.shape)


print('Training.')
clf = ExtraTreesClassifier(n_estimators=100, max_depth=None,
                           min_samples_split=11, random_state=0)
print('Cross eval:')
print(cross_val_score(clf, X_prune, Y))
quit()

clf.fit(X_prune, Y)
with open('data/test_2008.csv') as test:
    out = open('out.txt', 'w')
    test.readline()
    X = []
    for line in test:
        x = list(map(float, line.split(',')))
        x_ = []
        for i, v in enumerate(x):
            if i not in prune:
                x_.append(v)
        X.append(x_)
    X = normalize(X)
    print("TEST SHAPE:", X.shape)
    for i, y in enumerate(clf.predict(X)):
        out.write('{},{}\n'.format(i, int(y)))

