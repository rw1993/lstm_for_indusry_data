# -*- coding: utf-8 -*-
from sklearn.preprocessing import Normalizer
import numpy as np

with open("yfj.csv", "r") as f:
    lines = f.readlines()[2:]

counter = 1
def get_feature_from_line(line):
    global counter
    attributes = line.strip().split(",")
    try:
        feature = list(map(float, attributes[1:]))
        if len(feature) != 40:
            return None
    except:
        return None
    print counter
    counter += 1
    return feature


numpy_data = list(filter(lambda x: x is not None,
                  map(get_feature_from_line, lines)))

numpy_data = np.array(numpy_data)
normalizer = Normalizer()
normalizer.fit(numpy_data)
normalize_data = normalizer.transform(numpy_data)
