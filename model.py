import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config, on_train=False):
        self.batch_size = args.batch_size
        self.embedding_dim = args.embedding_dim
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config
        self.on_train = on_train
        self.step_num = 0

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.conv_op()
        self.predict_logit()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def Batch_normalization(self, vec):
        fc_mean, fc_var = tf.nn.moments(vec, axes=list(range(len(vec.shape) - 1)))
        scale = tf.Variable(tf.ones([vec.shape[-1]]), name='scale')  # 需要训练，初始化为1，具体参考上面链接中图中公式的γ
        shift = tf.Variable(tf.zeros([vec.shape[-1]]), name='shift')  # 就是batch_normalization中的参数offset，这个参数是需要训练的，初始化为0，参考图中的β
        epsilon = 0.001  # 图中的ε，选取一个适当小的数就可以
        ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=self.step_num)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            # 下面control_dependencies和identity可以参考我的博客http://blog.csdn.net/u013061183/article/details/79335065
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = tf.cond(tf.cast(self.on_train, tf.bool),  # on_train 的值是 True/False,但是不能用python的bool类型，所以要用cast转换成tf.bool
                            mean_var_with_update,  # 如果是 True, 更新 mean/var
                            lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                ema.average(fc_mean),
                                ema.average(fc_var)))
        return tf.nn.batch_normalization(vec, mean, var,
                                         shift, scale, epsilon)

    def lookup_layer_op(self):
        with tf.variable_scope("Embedding_lookup_layer"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            self.lookup = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
            word_embeddings = self.lookup
        with tf.variable_scope('Embeding_normalization_layer'):
            self.word_embeddings = self.Batch_normalization(word_embeddings)
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)
        self.word_embeddings = word_embeddings

    def conv_op(self):
        pooled_outputs = []
        # filter_sizes = [3, 4, 5]

        with tf.name_scope("Conv1d_layer"):
            # Convolution Layer
            conv1d = tf.layers.conv1d(self.word_embeddings,50,2,padding='same',activation=tf.nn.relu6)
            conv1d = tf.layers.dropout(conv1d,self.dropout_pl)
            s = tf.shape(conv1d)
            self.dense = tf.layers.dense(tf.reshape(conv1d, [-1, 50]), 50,activation=tf.nn.relu6)
            self.conv_out = tf.reshape(self.dense, [-1, s[1], 50])

    def biLSTM_layer_op(self):
        with tf.variable_scope("Bi-LSTM_layer"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            # time_major=False,所以input是batch*step*
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = output * tf.layers.dense(output, self.hidden_dim * 2, activation=tf.nn.tanh, name='dense',
                                              kernel_initializer=tf.truncated_normal_initializer)
            with tf.variable_scope('LSTM_normalization'):
                output = self.Batch_normalization(output)
            with tf.variable_scope('drop_out'):
                self.output = tf.nn.dropout(output, self.dropout_pl)



    def predict_logit(self):
        with tf.variable_scope("concat"):
            concat = tf.concat([self.conv_out, self.output], 2)
            self.concat = concat
            s1 = concat.shape
            s2 = tf.shape(concat)
            output = tf.reshape(concat, [-1, s1[-1]])
        with tf.variable_scope("dense"):
            pred = tf.layers.dense(output, self.num_tags)
        with tf.variable_scope('dense_norm'):
            pred = self.Batch_normalization(pred)
        self.logits = tf.reshape(pred, [s2[0], s2[1], self.num_tags])

        # W = tf.get_variable(name="W",
        #                     shape=[2 * self.hidden_dim, self.num_tags],
        #                     initializer=tf.contrib.layers.xavier_initializer(),
        #                     dtype=tf.float32)
        #
        # # b = tf.get_variable(name="b",
        # #                     shape=[self.num_tags],
        # #                     initializer=tf.zeros_initializer(),
        # #                     dtype=tf.float32)
        #
        # s = tf.shape(self.output)
        # output = tf.reshape(self.output, [-1, 2 * self.hidden_dim])
        # # pred = tf.matmul(output, W) + b
        # pred = tf.matmul(output, W)
        # with tf.variable_scope('dense_norm'):
        #     pred = self.Batch_normalization(pred)
        #
        # self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            with tf.variable_scope("CRF_layers"):
                log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                            tag_indices=self.labels,
                                                                            sequence_lengths=self.sequence_lengths)
            with tf.variable_scope("loss"):

                self.log_loss = -tf.reduce_mean(log_likelihood)
                # 正则
                self.l2_loss = tf.contrib.layers.apply_regularization(
                    regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                    weights_list=tf.trainable_variables())
                self.loss = self.log_loss + self.l2_loss

        else:
            with tf.variable_scope("loss"):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            with tf.variable_scope("soft_max"):

                self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
                self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())
        # saver1 = tf.train.Saver()

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            # saver1.restore(sess, r'.\data_path_save\1527663228\checkpoints\model-17136')

            self.add_summary(sess)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)
    def all(self,test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            return label_list, seq_len_list
    def demo_one(self, sess, sent):
        """
        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        # for tag, label in self.tag2label.items():
        #     label2tag[label] = tag if label != 0 else label
        tag = [label for label in label_list[0]]
        return tag
    def demo_many(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        count = 0
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            count += self.batch_size
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            print(count/len(sent))
        # label2tag = {}
        tags = []
        # for tag, label in self.tag2label.items():
        #     label2tag[label] = tag if label != 0 else label
        for labels in label_list:
            tag = [label for label in labels]
            tags.append(tag)
        return tags

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            # 相当print, input相当于sys.stdin.realine()
            sys.stdout.write(' processing: {} batch / {} batches. \n'.format(step + 1, num_batches) + '\r')
            self.step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {} '.format(start_time, epoch + 1, step + 1,
                                                                                 loss_train, self.step_num))

            self.file_writer.add_summary(summary, self.step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=self.step_num)


        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        TP, FP, TN, FN = 0, 0, 0, 0
        for label_, (sent, tag) in zip(label_list, data):
            for p, t in zip(label_, tag):
                if p == 0 and t == 'O': TN += 1
                if p == 0 and t != 'O': FN += 1
                if p != 0 and t != 'O': TP += 1
                if p != 0 and t == 'O': FP += 1


            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F = 2 * precision * recall / (precision + recall)
        print("precision is ", precision, " recall is ", recall, " F-value is ", F)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)
        with open(metric_path, 'a+') as f:
            f.write("precision is " + str(precision) + " recall is " + str(recall) + " F-value is " + str(F))