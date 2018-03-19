import tensorflow as tf 
import numpy as np
import tflearn

def crf_layer(net, batch_size, dimensions, distances):
    Q = net
    U = -tf.log(net)

    def get_nearest(distance, near_cities=5):
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
        nearest = get_nearest(distance)

        def message_passing_ij(i, j):
            if j not in nearest[i]:
                return 0.0
            else:
                return kernel(i, j, distance) * Q[:, j,:]
        
        def message_passing_i(i):
            tensor = sum(message_passing_ij(i, j) for j in range(dimensions))
            return tensor

        tensors = [message_passing_i(i) for i in range(dimensions)]
        return tf.stack(tensors, axis=1)
        

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

def main():
    batch_size = 4
    dimensions = 40
    classes = 2
    from correlation import pearson_dict
    Q = tf.placeholder(tf.float32, (4, 40, 2))
    Q = crf_layer(Q, batch_size, dimensions, [pearson_dict])
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()