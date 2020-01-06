import pickle
def load_pickle(fname):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)

def dump_pickle(data, fname):
    with open(fname, 'wb') as fout:
        return pickle.dump(data, fout)

def mmap(func, *args):
    return list(map(func, *args))


def mfilter(func, *args):
    return list(filter(func, *args))

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f
