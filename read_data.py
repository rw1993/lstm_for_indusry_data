# -*- coding: utf-8 -*-
from sklearn.preprocessing import Normalizer
import numpy as np
with open("yfj.csv", "r") as f:
    lines = f.readlines()[2:]

def get_feature_from_line(line):
    attributes = line.strip().split(",")
    feature = map(float, attributes[1:])
    return np.array(feature)


numpy_data = map(get_feature_from_line, lines)
numpy_data = np.array(numpy_data)
import ipdb; ipdb.set_trace()
normalizer = Normalizer()
normalizer.fit(numpy_data)
normalize_data = normalizer.transform(numpy_data)

import ipdb; ipdb.set_trace()
