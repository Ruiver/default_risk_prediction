#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
count = 0
c = 0
with open("negative","r", encoding="utf8") as f:
    for line in f.readlines():
        with open("all_label", 'a+', encoding="utf8") as a:
            if c <count:
                c += 1
                continue
            print(line.strip())
            i = input("请输入：")
            a.write(line.replace('\n','') + '\t' + str(i)+'\n')
            c += 1

