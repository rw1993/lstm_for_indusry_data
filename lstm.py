import tensorflow as tf
import tflearn
import numpy as np
from mfi_function import crf_layer
from sklearn import ensemble
import pickle

batch_size = 1
timestep = 5
dimensions = 40

CLASSIFIERS = [ensemble.AdaBoostClassifier,
               ensemble.GradientBoostingClassifier,
               ensemble.RandomForestClassifier]
def loading_base_classifiers():
    classifierss = []
    file_path = "base_classifiers/classifier_{}_at_index_{}"
    for C in CLASSIFIERS:
        classifiers = []
        for index in range(dimensions):
            classifiers.append(pickle.load(open(file_path.format(C, index), "rb")))
            print C, index
        classifierss.append(classifiers)
    return classifierss

# classifierss = loading_base_classifiers()
def generate_batch(data_set="train", begin=None):

    scaler = pickle.load(open("scaler", "rb"))
    current = 0
    if data_set == "train":
        begin = 2
        end = 838858
    else: 
        begin = 838859 if begin is None else begin
        end = 1048571
    
    while True:
        by = []
        bf = []
        bx = []
        x = []
        f = np.zeros(shape=(dimensions, 6))
        with open("yfj.csv", "r") as data:
            for line in data.xreadlines():
                if current < begin:
                    current += 1
                    continue
                if current >= end:
                    current = 0
                    break
                current += 1
                attributes = line.strip().split(",")
                feature = attributes[1:]
                feature = scaler.transform([feature])[0]
                if len(x) < timestep:
                    x.append(feature)
                else:
                    bx.append(x)
                    y = [0 if feature[i] < x[-1][i] else 1 for i in range(dimensions)]
                    by.append(y)
                    '''
                    X = np.stack(x).reshape(-1)
                    for c, C in enumerate(CLASSIFIERS):
                        for i in range(dimensions):
                            r = classifierss[c][i].predict_proba([X])
                            f[i][c*2] = r[0][0]
                            f[i][c*2+1] = 1.0 - r[0][0]
                    '''
                    bf.append(f)
                    if len(bx) == batch_size:
                        yield np.array(bx), np.array(by).astype(np.int32), np.array(bf)
                        bx = []
                        by = []
                        bf = []
                        f = np.zeros((dimensions, 6))
                        x = x[1:] + [feature]


def main():
    print "building"
    input_tensor = tf.placeholder(dtype=np.float32, shape=(batch_size, timestep, dimensions))
    feature_tensor = tf.placeholder(dtype=np.float32, shape=(batch_size, dimensions, 6))
    net = tflearn.layers.recurrent.lstm(input_tensor, n_units=80)
    net = tf.reshape(net, (batch_size, 40, 2))
    net = tf.nn.softmax(net)
    pearson = pickle.load(open("pearson_distance", "rb"))
    def distance(i, j):
        return pearson[(i, j)]   
    Q = crf_layer(net, batch_size, dimensions, distance, feature_tensor)
    flag = "train"
    if flag == "train":
        label_tensor = tf.placeholder(dtype=np.int32, shape=(batch_size, dimensions))
        loss_tensor = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Q,
                                                                            labels=tf.one_hot(label_tensor,
                                                                                              2)))

        train_step = tf.train.AdamOptimizer(0.00001).minimize(loss_tensor)
        init = tf.global_variables_initializer()
        tf.summary.scalar("loss", loss_tensor)
        merge_summary_op = tf.summary.merge_all()
        step = 0
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter("lstm_log", graph=sess.graph)
            sess.run(init)
            recent_avgloss = 0.0
            saver = tf.train.Saver()
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
                    test_begin = np.random.randint(838859,  1048571-6000)
                    valid = 0
                    for bx, by, bf in generate_batch('test', begin=test_begin):
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
