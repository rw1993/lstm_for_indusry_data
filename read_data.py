from __future__ import print_function
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import numpy as np
import pickle

def get_normalize_data(data_path="yfj.csv"):
    with open(data_path, "r", errors="ignore") as f:
        lines = [str(line) for line in f.readlines()][2:]

    def get_feature_from_line(line):
        attributes = line.strip().split(",")
        try:
            feature = list(map(float, attributes[1:]))
            if len(feature) != 40:
                return None
        except:
            return None
        return feature

    numpy_data = list(filter(lambda x: x is not None,
                      map(get_feature_from_line, lines)))

    numpy_data = np.array(numpy_data)
    # Preprocessor = MinMaxScaler
    Preprocessor = StandardScaler
    normalizer = Preprocessor()
    normalizer.fit(numpy_data)
    normalize_data = normalizer.transform(numpy_data)
    return normalize_data
print("getting data")
try:
    normalize_data = pickle.load(open("normalize_data", "rb"))
except:
    normalize_data = get_normalize_data()
    import ipdb; ipdb.set_trace()
    pickle.dump(normalize_data, open("normalize_data", "wb"))
print("data got")