import tensorflow as tf 
import numpy as np
import tflearn


def crf_layer(net, batch_size, dimensions, distance, feature_tensor):
    Q = net
    U = -tf.log(net)
    near_cities = 5
    nearest = {}
    for city0 in range(dimensions):
        distance_index = []
        for city1 in range(dimensions):
            if city0 != city1:
                d = distance(city0, city1)
                distance_index.append((d, city1))
        distance_index = sorted(distance_index)
        nearest[city0] = [i for d, i in distance_index[:near_cities]]


    def k1(i, j, batch):
        f = feature_tensor[batch]
        f2 = tf.norm(f[i]-f[j]) ** 2 / 100 ** 2 / 2.0
        d2 = distance(i, j) ** 2 / 100 ** 2 / 2.0
        return tf.exp(-d2-f2)

    def k2(i, j, batch):
        d2 = distance(i, j) ** 2 / 100 ** 2 / 2.0
        return np.exp(-d2)
        
    def message_passing(k):
        tensor_cache = []
        for batch in range(batch_size):
            b = []
            tensor_cache.append(b)
            for city in range(dimensions):
                c = []
                b.append(c)
                for class_ in range(2):
                    c.append(1)
        for batch in range(batch_size):
            for city in range(dimensions):
                for class_ in range(2):
                    tensor_cache[batch][city][class_] = sum([Q[batch, c, class_]*k(c, city, batch) for c in range(dimensions) if c != city and c in nearest[city]])
        for batch in range(batch_size):
            for city in range(dimensions):
                tensor_cache[batch][city] = tf.stack(tensor_cache[batch][city])
        for batch in range(batch_size):
            tensor_cache[batch] = tf.stack(tensor_cache[batch])
        return tf.stack(tensor_cache)

    MAX_ITR = 5

    for itr in range(MAX_ITR):
        # message passing
        #Q1 = message_passing(k1)
        Q2 = message_passing(k2)
    
        # weight filter
        '''
        Q1_ = tflearn.layers.conv_1d(incoming=Q1, nb_filter=2,
                                    filter_size=1,
                                    reuse=tf.AUTO_REUSE, name="weight_filter1",
                                    scope="weight_filter1")
        '''
        Q2_ = tflearn.layers.conv_1d(incoming=Q2, nb_filter=2,
                                    filter_size=1,
                                    reuse=tf.AUTO_REUSE, name="weight_filter2",
                                    scope="weight_filter2")

        #Q = tf.nn.softmax(-U-Q1_-Q2_)
        Q = tf.nn.softmax(-U-Q2_)
    return Q
