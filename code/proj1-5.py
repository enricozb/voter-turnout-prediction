import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from utils.parse                import extract, prune, prune_by_column
from sklearn.ensemble           import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors          import KNeighborsClassifier
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

l = []
k = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for p in k:
    X_t, pruned = prune(X_train, threshold=p, to_remove={0})
    print('Fitting Random Forest with prune {}'.format(p))

    clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                               min_samples_split=4, random_state=0)
    v = cross_val_score(clf, X_t, Y_train).mean()
    print(v)
    print()
    l.append(v)

import matplotlib.pyplot as plt

plt.plot(k, l)
plt.title('Random Forest Cross-Validation Accuracy vs. Pruning proportion')
plt.xlabel('Pruning proportion')
plt.ylabel('Cross-Validation Accuracy')
plt.show()
# quit()
# clf.fit(X_train, Y_train)

# print('Predicting and Writing 2008.')
# with open('out_2008.csv', 'w') as out:
#     X_test = extract('data/test_2008.csv')
#     X_test = prune_by_column(X_test, pruned)
#     Y_test = clf.predict(X_test)
#     out.write('id,PES1\n')
#     for i,y in enumerate(Y_test):
#         out.write('{},{}\n'.format(i, y))

# quit()

# print('Predicting and Writing 2012.')
# with open('out_2012.csv', 'w') as out:
#     X_test = extract('data/test_2012.csv')
#     X_test = prune_by_column(X_test, pruned)
#     Y_test = clf.predict(X_test)
#     out.write('id,PES1\n')
#     for i,y in enumerate(Y_test):
#         out.write('{},{}\n'.format(i, y))
