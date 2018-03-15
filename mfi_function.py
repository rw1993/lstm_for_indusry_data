import tensorflow as tf 
import numpy as np
import tflearn


def crf_layer(net, batch_size, dimensions, distances):
    Q = net
    U = -tf.log(net)
    near_cities = 5

    def get_nearest(distance):
        nearest = {}
        for city0 in range(dimensions):
            distance_index = []
            for city1 in range(dimensions):
                if city0 != city1:
                    d = distance[(city0, city1)]
                    distance_index.append((d, city1))
            distance_index = sorted(distance_index)
            nearest[city0] = [i for d, i in distance_index[:near_cities]]
        return nearest

    def kernel(i, j, distance):
        d = distance[(i, j)] ** 2 / 100.0 ** 2 / 2.0
        return np.exp(-d)

    def message_passing(distance, Q):
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
                    nearest  = get_nearest(distance)
                    tensor_cache[batch][city][class_] = sum([Q[batch, c, class_]*distance[(c, city)] for c in range(dimensions) if c != city and c in nearest[city]])
        for batch in range(batch_size):
            for city in range(dimensions):
                tensor_cache[batch][city] = tf.stack(tensor_cache[batch][city])
        for batch in range(batch_size):
            tensor_cache[batch] = tf.stack(tensor_cache[batch])
        return tf.stack(tensor_cache)

    def for_a_k(distance, Q):
        Q2 = message_passing(distance, Q)
        Q2_ = tflearn.layers.conv_1d(incoming=Q2, nb_filter=2,
                                     filter_size=1,
                                     reuse=tf.AUTO_REUSE, name="weight_filter2",
                                     scope="weight_filter2")
        return Q2_

    MAX_ITR = 5
    for itr in range(MAX_ITR):
        Qs = [for_a_k(distance,Q) for distance in distances]
        Q = tf.nn.softmax(tf.exp(-U-sum(Qs)))
    return Q
