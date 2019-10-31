#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
import requests
import  tensorflow as tf
import numpy as np
a = np.array([[4],[4],[4]])
b = np.array([[2],[1],[2]])
with tf.Session():

    print(tf.losses.mean_squared_error(a,b).eval())
