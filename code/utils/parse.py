'''
Parses csv format, where first line indicates column labels.
'''
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
            vec = list(map(float, line.split(',')))
            d = {L[i] : vec[i] for i in range(len(L))}
            d[0] = vec
            D.append(d)

        return L, D

def extract(filename, label, separate=False):
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
    for d in D:
        i = L.index(label)
        y = d[0].pop(i)
        data.append((d[0], y))

    if separate:
        return tuple(zip(*data))
    return data
