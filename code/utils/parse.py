'''
Parses csv format, where first line indicates column labels.
'''

def prune_by_column(X, to_remove):
    X_pruned = []
    for x in X:
        X_pruned.append([v for i, v in enumerate(x) if i not in to_remove])
    return X_pruned

def prune(X, threshold, to_remove=None):
    '''
    Removes columns above a certain threshold of -1 occurences
    Returns the pruned data set and the columns removed as a set.
    '''
    if to_remove is None:
        to_remove = []

    blanks = [0] * len(X[0])
    for x in X:
        for i, v in enumerate(x):
            if v == -1:
                blanks[i] += 1

    blanks = list(map(lambda x: x/len(X), blanks)) # convert to percentages
    to_remove |= {i for i, b in enumerate(blanks) if b > threshold}
    return prune_by_column(X, to_remove), to_remove

def parse(filename):
    '''
    Returns (L, D) where
        - L is a tuple of labels
        - D is a list of dictionaries of the datapoints

    For some element d in D:
        d[L[i]] returns the respective value.
        d[0] returns the original list of values
    '''

    with open(filename) as f:
        L = tuple(f.readline().strip().split(','))
        D = []
        for line in f:
            vec = list(map(int, line.split(',')))
            d = {L[i] : vec[i] for i in range(len(L))}
            d[0] = vec
            D.append(d)

        return L, D

def extract(filename, label=None, separate=False):
    '''
    Used to filter a certain label.

    If separate is False: returns D where
        - D is a list of tuples (x, y) where
            - x is a tuple of features with labels not in labels
            - y is a feature whose label == label

    If separate is True: returns X, Y where:
        - X is a list of x as described above
        - Y is a list of y as described above
        - Relative order of elements of X and Y are kept
    '''
    L, D = parse(filename)
    data = []
    if label:
        i = L.index(label)
    for d in D:
        if label:
            y = d[0].pop(i)
            data.append((d[0], y))
        else:
            data.append(d[0])

    if separate:
        return tuple(zip(*data))
    return data
