import tensorflow as tf
import tflearn
import numpy as np
from mfi_function import crf_layer
import pickle
from read_data import normalize_data
from correlation import pearson_dict

batch_size = 2
timestep = 5
dimensions = 40


def generate_batch(data_set="train"):
    current = 0
    total = len(normalize_data)
    train = normalize_data[: 838858]
    test = normalize_data[838858: ]
    if data_set == "train":
        data = train
    else:
        data = test
    by = []
    bf = []
    bx = []
    while True:
        index = np.random.randint(len(data))
        x = map(lambda i: data[i],
                [index - i for i in range(1, 6)])
        
        f = np.zeros(shape=(dimensions, 6))
        if len(x) < timestep:
            continue
        bx.append(x)
        feature = data[index]
        y = [0 if feature[i] < x[-1][i] else 1 for i in range(dimensions)]
        by.append(y)
        bf.append(f)
        if len(bx) == batch_size:
            yield np.array(bx), np.array(by).astype(np.int32), np.array(bf)
            bx = []
            by = []
            bf = []


def main():
    print "building"
    input_tensor = tf.placeholder(dtype=np.float32, shape=(batch_size, timestep, dimensions))
    feature_tensor = tf.placeholder(dtype=np.float32, shape=(batch_size, dimensions, 6))
    net = tflearn.layers.recurrent.lstm(input_tensor, n_units=64)
    net = tflearn.layers.fully_connected(net, n_units=128, activation="relu")
    net = tflearn.layers.fully_connected(net, n_units=256, activation="relu")
    net = tflearn.layers.fully_connected(net, n_units=128, activation="relu")
    net = tflearn.layers.fully_connected(net, n_units=80, activation="linear")
    net = tf.reshape(net, (batch_size, 40, 2))
    net = tf.nn.softmax(net)
    Q = crf_layer(net, batch_size, dimensions, [pearson_dict])
    flag = "train"
    if flag == "train":
        label_tensor = tf.placeholder(dtype=np.int32, shape=(batch_size, dimensions))
        loss_tensor = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Q,
                                                                            labels=tf.one_hot(label_tensor,
                                                                                              2)))

        train_step = tf.train.AdamOptimizer(0.000001).minimize(loss_tensor)
        init = tf.global_variables_initializer()
        tf.summary.scalar("loss", loss_tensor)
        merge_summary_op = tf.summary.merge_all()
        step = 0
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter("lstm_log", graph=sess.graph)
            sess.run(init)
            recent_avgloss = 0.0
            saver = tf.train.Saver()
            print "builded"
            for bx, by, bf in generate_batch():
                _, summary_string, loss = sess.run([train_step, merge_summary_op, loss_tensor],
                                                   feed_dict={input_tensor: bx, feature_tensor: bf,
                                                        label_tensor: by})
                summary_writer.add_summary(summary_string, step)
                step += 1
                recent_avgloss += loss
                if step % 1000 == 0:
                    print recent_avgloss / 1000.0, step
                    recent_avgloss = 0.0
                if step % 10000 == 0:
                    saver.save(sess, "lstm_model/", global_step=step)
                    # valid
                    valid_num = 50
                    total = 0
                    acc = 0.0
                    valid = 0
                    for bx, by, bf in generate_batch('test'):
                        q_result, = sess.run([Q,], feed_dict={input_tensor: bx, feature_tensor: bf, 
                                                   label_tensor: by})
                        valid += 1
                        for b in range(batch_size):
                            for i in range(dimensions):
                                total += 1
                                if q_result[b][i][0] > q_result[b][i][1]:
                                    if by[b][i] == 0:
                                        acc += 1
                                else:
                                    if by[b][i] == 1:
                                        acc += 1
                        if valid >= valid_num:
                            break
                    print "valid acc is {} after step {}".format(float(acc)/total, step)


if __name__ == '__main__':
    main()
