import pickle

def save_cache_to_file(cache, filename):
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)


def load_cache_from_file(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}
