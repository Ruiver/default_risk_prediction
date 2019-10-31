#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
import os
import multiprocessing
import json
import requests
from sklearn.externals import joblib
from tensorflow.contrib.rnn import LSTMCell
import re
import pymongo
from sklearn import tree
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle

def load_word2vec_model():
    model = Word2Vec.load('../data_path/Word2vec_model.pkl')
    return model

def get_data():
    plat_word = {}
    with open("result_new.txt", 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if '\1' in line:
                key = line.split('\1')[1]
                plat_word[key] = []
            elif '\2' in line:
                continue
            else:
                eva_ent = line.split('\t')[2]
                try:
                    # plat_word[key] += eval(eva_ent.split("ALL: ")[1])
                    plat_word[key] += eval(eva_ent.split("EVA: ")[1].split(' ALL')[0])
                except Exception as e:
                    # print(line)
                    pass
    empty_key = []
    for k, v in plat_word.items():
        if not v:
            empty_key.append(k)
    for k in empty_key:
        del plat_word[k]
    return plat_word


def get_platAndproblemPlat():
    # all_pingtai = requests.get("http://www.wdzj.com/wdzj/html/json/dangan_search.json").json()
    with open("temp", 'r', encoding='utf8') as f:
        # json.dump(all_pingtai, f)
        all_pingtai = json.load(f)
    plats = set()
    for pingtai in all_pingtai:
        plats.add(pingtai['platName'])
    #collection = pymongo.MongoClient(host='10.205.27.72')['lunwen']['problem']
    collection = pymongo.MongoClient(host='10.108.25.10')['lunwen']['problem']
    problem_plats = set()
    shut_plats = set()
    for i in collection.find({}):
        if i["problemTime"] < "2018-12-30":
            if str(i['class']) == '2':
                problem_plats.add(i["platName"])
            elif str(i['class']) == '1':
                shut_plats.add(i["platName"])

    # problem_plats = collection.distinct('platName')
    for prob in problem_plats:
        plats.add(prob)
    return plats, problem_plats,shut_plats

def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id
def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id
def get_x_y(plat_word):
    plats, problemPlat,shut_plats = get_platAndproblemPlat()
    word2id = read_dictionary('../../data_path/word2id.pkl')
    x = []
    y = []

    for key, value in plat_word.items():
        value = "".join(value)[-200:]
        sen_id = sentence2id(value, word2id)
        if key not in plats:
            continue
        x.append(sen_id)
        if key in problemPlat:
            y.append([0, 0, 1])
        elif key in shut_plats:
            y.append([0, 1, 0])

        else:
            y.append([1, 0, 0])


    return x ,y

def pad_seq(x, max_len=0):
    if not max_len:
        max_len = max(map(lambda sentence: len(sentence), x))
    seq_list, seq_len_list = [], []
    for seq in x:
        seq = list(seq)
        # seq后面没必要接[:max_len]呀，不知道为什么这么做
        seq_ = seq[:max_len] + [0] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def get_accuacy(y_t,y_pre):
    count = 0
    for a, b in zip(y_t, y_pre):
        if a==b:
            count += 1
    return count/len(y_t)
def random_embedding(word2id, embedding_dim):
    print('load the w2v_model')
    model = Word2Vec.load('../../data_path/Word2vec_model.pkl')
    embedding_mat = np.zeros((len(word2id), embedding_dim))
    for word, id in word2id.items():
        if word != '<UNK>' and word != '<PAD>' and word != '<UNK>':
            embedding_mat[id] = model[word]
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat
def get_dense(train_x, train_y, test_x, test_y, word2id):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    tf_x = tf.placeholder(tf.int32, (None, len(train_x[0])))  # input x
    tf_y = tf.placeholder(tf.float32, (None, 3))
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    embeddings = random_embedding(word2id,300)
    with tf.variable_scope("Embedding_lookup_layer"):
        _word_embeddings = tf.Variable(embeddings,
                                       dtype=tf.float32,
                                       name="_word_embeddings")
        lookup = tf.nn.embedding_lookup(params=_word_embeddings, ids=tf_x, name="word_embeddings")
        print(lookup.shape)
        word_embeddings = tf.reshape(lookup, [-1,300*len(train_x[0])])

    word_embeddings_ = tf.nn.dropout(word_embeddings, dropout_pl)

    layer1 = tf.layers.dense(word_embeddings_, 200, activation=tf.nn.relu6)
    layer2 = tf.layers.dense(layer1, 500, activation=tf.nn.relu6)
    layer3 = tf.layers.dense(layer2, 500, activation=tf.nn.relu6)
    layer5 = tf.layers.dense(layer3, 100, activation=tf.nn.relu6)
    out = tf.layers.dense(layer5, 3 )
    pred = tf.cast(tf.argmax(out,dimension=1),tf.int32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=out))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    sess = tf.Session()  # control training and others
    sess.run(tf.global_variables_initializer())  # initialize var in graph

    for step in range(500):
        # train and net output
        _, l, out_, = sess.run([train_op, loss, out], {tf_x: train_x, tf_y: train_y, dropout_pl:0.5})
        if step % 5:
            print(l)
        if step % 20 :
            pred_y_ = sess.run(pred, {tf_x: test_x, tf_y: test_y, dropout_pl:1})
            test_y_ = np.argmax(test_y, axis=1)
            # print(pred_y,test_y)
            print(classification_report(test_y_, pred_y_), get_accuacy(test_y_, pred_y_))

def get_LSTM(train_x, train_y, train_len, test_x, test_y, test_len, word2id):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_len = np.array(train_len)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_len = np.array(test_len)
    tf_x = tf.placeholder(tf.int32, (None, len(train_x[0])))  # input x
    tf_y = tf.placeholder(tf.float32, (None, 3))
    sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
    dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    embeddings = random_embedding(word2id,300)
    with tf.variable_scope("Embedding_lookup_layer"):
        _word_embeddings = tf.Variable(embeddings,
                                       dtype=tf.float32,
                                       name="_word_embeddings")
        lookup = tf.nn.embedding_lookup(params=_word_embeddings, ids=tf_x, name="word_embeddings")
    # word_embeddings_ = tf.nn.dropout(lookup, 0.5)
    with tf.variable_scope("Bi-LSTM_layer"):
        cell_fw = LSTMCell(300)
        cell_bw = LSTMCell(300)
        # time_major=False,所以input是batch*step*
        (output_fw_seq, output_bw_seq), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=lookup,
            sequence_length=sequence_lengths,
            dtype=tf.float32)
        output = tf.concat([output_state_fw[1], output_state_bw[1]], axis=-1)

        # output = output * tf.layers.dense(output, 300 * 2, activation=tf.nn.tanh, name='dense',
        #                                   kernel_initializer=tf.truncated_normal_initializer)
        # output = tf.unstack(tf.transpose(output, [1, 0, 2]))[-1]



        with tf.variable_scope('drop_out'):
            output = tf.nn.dropout(output, dropout_pl)
    layer2 = tf.layers.dense(output, 1000)
    layer3 = tf.layers.dense(layer2, 100)
    out = tf.layers.dense(layer3, 3)
    pred = tf.cast(tf.argmax(out,dimension=1),tf.int32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=out))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    sess = tf.Session()  # control training and others
    sess.run(tf.global_variables_initializer())  # initialize var in graph

    for step in range(500):
        # train and net output
        _, l, out_, = sess.run([train_op, loss, out], {tf_x: train_x, tf_y: train_y, sequence_lengths: train_len, dropout_pl:0.5})
        if step % 5:
            print(l)
        if step % 20 :
            pred_y_ = sess.run(pred, {tf_x: test_x, tf_y: test_y, sequence_lengths: test_len, dropout_pl:1})
            test_y_ = np.argmax(test_y, axis=1)
            print(test_y, pred_y_)
            # print(pred_y,test_y)
            print(classification_report(test_y_, pred_y_), get_accuacy(test_y_, pred_y_))



if __name__ == '__main__':
    print("get vec of all words...")
    plat_word = get_data()
    x, y = get_x_y(plat_word)
    train_x, test_x, train_y, test_y = train_test_split(x,y)

    train_x, train_len = pad_seq(train_x)
    test_x, test_len = pad_seq(test_x,max_len=len(train_x[0]))
    count =0
    _test_x = []
    _test_y = []
    _test_len = []
    for x, y,l in zip(test_x, test_y,test_len):
        if y == [0,1]:
            if count > 57:
                continue
            else:
                count += 1
                _test_x.append(x)
                _test_y.append(y)
                _test_len.append(l)
        else:
            _test_x.append(x)
            _test_y.append(y)
            _test_len.append(l)
    test_x = _test_x
    test_y = _test_y
    test_len = _test_len
    word2id = read_dictionary(os.path.join('../../data_path/word2id.pkl'))
    # get_dense(train_x, train_y, test_x, test_y, word2id)
    get_LSTM(train_x, train_y, train_len, test_x, test_y, test_len, word2id)






