#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
import os
import requests
from sklearn.externals import joblib
import re
import pymongo
from sklearn import tree
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import json

def get_words():
    words = []
    with open('words.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            word = line.strip()
            words.append(word)
    return words


def load_word2vec_model():
    model = Word2Vec.load('../data_path/Word2vec_model.pkl')
    return model


def get_vec(model, words):
    vec_sums = []
    for word in words:
        vec_sum = 0
        flag = False
        count = 0
        for char_ in word:
            if char_.isdigit():
                char_ = '<NUM>'
            if char_ not in model.wv.vocab.keys():
                continue
            vec = model[char_]
            count += 1
            if not flag:
                vec_sum = vec
            else:
                vec_sum += vec
                # print(vec_sum.shape())
        if count != 0:
            vec_sums.append(vec_sum / count)
    return np.array(vec_sums)


def model(x, k=500):
    model = KMeans(n_clusters=k, random_state=9)
    model.fit(x)
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
                    plat_word[key] += eval(eva_ent.split("ALL: ")[1])
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
    all_pingtai = requests.get("http://www.wdzj.com/wdzj/html/json/dangan_search.json").json()
    plats = set()
    for pingtai in all_pingtai:
        plats.add(pingtai['platName'])
    collection = []
    with open("problem.txt", "r", encoding="utf8") as f:
        for line in f.readlines():
            collection.append(json.loads(line.strip()))

    problem_plats = set()
    for i in collection:
        if i["problemTime"] < "2018-07-30":
            problem_plats.add(i["platName"])
    # problem_plats = collection.distinct('platName')
    for prob in problem_plats:
        plats.add(prob)
    return plats, problem_plats


def get_x_y(model, plat_word, k_cluster):
    plat_vec = {}
    plats, problemPlat = get_platAndproblemPlat()
    d = {}
    for key, value in plat_word.items():
        x = get_vec(word2vec_modle, value)
        y = model.predict(x)
        for a, b in zip(value,y):

            if not d.get(b,None):
                d[b] = set()
                d[b].add(a)
            else:
                d[b].add(a)
        vec = np.zeros(k_cluster)
        for dim in y:
            vec[dim] += 1
        if key not in plats:
            continue
        if key in problemPlat:
            plat_vec[key] = [vec, 1]
        else:
            plat_vec[key] = [vec, 0]
    for i in d.values():
        print(i)
    return plat_vec

def get_vec_y(plat_word):
    plat_vec = {}
    plats, problemPlat = get_platAndproblemPlat()
    for key, value in plat_word.items():
        x = get_vec(word2vec_modle, value)
        vec = np.zeros(len(x[0]))
        for i in x:
            vec += i
        vec = vec/len(x)
        if key not in plats:
            continue
        if key in problemPlat:
            plat_vec[key] = [vec, 1]
        else:
            plat_vec[key] = [vec, 0]
    return plat_vec

def get_svm(train_x, train_y):
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    return clf

def get_decision(train_x, train_y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y)
    return clf
def get_accuacy(y_t,y_pre):
    count = 0
    for a, b in zip(y_t, y_pre):
        if a==b:
            count += 1
    return count/len(y_t)
def get_dense(train_x, train_y, test_x, test_y, k_cluster):
    tf_x = tf.placeholder(tf.float32, (None, k_cluster))  # input x
    tf_y = tf.placeholder(tf.float32, (None,))
    layer1 = tf.layers.dense(tf_x, 1000, activation=tf.nn.relu6)
    layer2 = tf.layers.dense(layer1, 1500, activation=tf.nn.relu6)
    layer3 = tf.layers.dense(layer2, 1000, activation=tf.nn.relu6)
    layer4 = tf.layers.dense(layer3, k_cluster, activation=tf.nn.relu6)
    layer5 = tf.layers.dense(layer4, k_cluster, activation=tf.nn.relu6)
    out = tf.layers.dense(layer5, 1, activation=tf.nn.sigmoid)
    out = tf.reshape(out, [-1])
    a = tf.cast(tf.greater(out, 0.5), tf.int32)
    loss = tf.losses.mean_squared_error(tf_y, out)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    sess = tf.Session()  # control training and others
    sess.run(tf.global_variables_initializer())  # initialize var in graph

    for step in range(130):
        # train and net output
        _, l, pred, = sess.run([train_op, loss, out], {tf_x: train_x, tf_y: train_y})
        # if step % 5:
        #     print(l)

    a_, pred_y = sess.run([a, out], {tf_x: test_x, tf_y: test_y})
    print("1", classification_report(test_y, a_), get_accuacy(test_y, a_))
def get_train_and_test(plat_vec):
    x = []
    y = []
    for key, [x_, y_] in plat_vec.items():
        x.append(x_)
        y.append(y_)
    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    print(y.sum(), len(y))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    return train_x, test_x, train_y, test_y

if __name__ == '__main__':
    k_cluster = 150
    if not os.path.exists("kmeans_model.m"):
        print("get words...")
        words = get_words()
        print("get word2vec model ...")
        word2vec_modle = load_word2vec_model()
        print("get vec of all words...")
        x = get_vec(word2vec_modle, words)
        print(x.shape)
        print("train kmeans model...")
        k_model = model(x, k_cluster)
        print("dump kmeans model...")
        joblib.dump(k_model, "kmeans_model.m")
    else:
        print("get word2vec model ...")
        word2vec_modle = load_word2vec_model()
        print("load kmeans model")
        k_model = joblib.load("kmeans_model.m")
    plat_word = get_data()
    plat_vec = get_x_y(k_model, plat_word, k_cluster)
    # plat_vec = get_vec_y(plat_word)
    train_x, test_x, train_y, test_y = get_train_and_test(plat_vec)
    svm_model = get_svm(train_x, train_y)
    print("svm result is")
    pred_y = svm_model.predict(test_x)
    print(classification_report(test_y, pred_y))
    dec_model = get_decision(train_x, train_y)
    print("dec result is")
    pred_y = dec_model.predict(test_x)
    print(classification_report(test_y, pred_y))
    print("dense result is")
    # get_dense(train_x, train_y, test_x, test_y, k_cluster)



