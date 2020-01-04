import pickle
def load_pickle(fname):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)

def dump_pickle(data, fname):
    with open(fname, 'wb') as fout:
        return pickle.dump(data, fout)
