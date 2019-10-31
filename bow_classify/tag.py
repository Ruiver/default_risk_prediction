#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'
import redis
def get_entity(tag_seq, char_seq):
    global h
    ENT = []
    EVA = []
    ALL = []
    temp_tags = ""
    temp_chars = ""
    for n, (cha, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == '3' or tag == '6' or tag == '1' or tag == '4' or tag == '0':
            if temp_chars:
                if temp_tags == '3':
                    ENT.append(temp_chars)
                    ALL.append(temp_chars)
                elif temp_tags == '6':
                    EVA.append(temp_chars)
                    ALL.append(temp_chars)
                elif len(temp_tags) > 1 and temp_tags[0] == '1' and temp_tags.count('2') == (len(temp_tags) - 1):
                    ENT.append(temp_chars)
                    ALL.append(temp_chars)
                elif len(temp_tags) > 1 and temp_tags[0] == '4' and temp_tags.count('5') == (len(temp_tags) - 1):
                    EVA.append(temp_chars)
                    ALL.append(temp_chars)

            temp_chars = ''
            temp_tags = ''

            if tag == '3'  :
                if cha == '标' or cha == '群':
                    ENT.append(cha)
                    ALL.append(cha)
            elif tag == '6':
                if cha == '雷':
                    EVA.append(cha)
                    ALL.append(cha)
            elif tag != '0':
                temp_chars = cha
                temp_tags = tag
        elif tag == '2':
            if temp_chars:
                if temp_tags[0] == '1' and temp_tags.count('2') == (len(temp_tags) - 1):
                    temp_tags += tag
                    temp_chars += cha
                elif len(temp_tags) > 1 and temp_tags[0] == '4' and temp_tags.count('5') == (len(temp_tags) - 1):
                    EVA.append(temp_chars)
                    ALL.append(temp_chars)
                    temp_chars = ''
                    temp_tags = ''
        elif tag == '5':
            if temp_chars:
                if temp_tags[0] == '4' and temp_tags.count('5') == (len(temp_tags) - 1):
                    temp_tags += tag
                    temp_chars += cha
                elif len(temp_tags) > 1 and temp_tags[0] == '1' and temp_tags.count('2') == (len(temp_tags) - 1):
                    ENT.append(temp_chars)
                    ALL.append(temp_chars)
                    temp_chars = ''
                    temp_tags = ''
    return ENT, EVA, ALL
with open('../data_path/all', encoding='utf8') as f:
    count = 0
    error_count = 0
    words_s = set()
    for line in f.readlines():
        sen = list(line.strip()[:800])
        label = len(line.strip()) * ["O"]
        if len(label) == 0:
            continue
        data_,label_ = line.strip().split('\t')
        label_list = []
        tag = [label for label in label_]
        demo_sent = list(data_.strip())

        ENT, EVA, ALL = get_entity(tag, demo_sent)
        print(ALL)
        for eva in EVA:
            words_s.add(eva)
        with open('tag/tag.txt','w', encoding='utf8') as f:
            for eva in words_s:
                f.write(eva+'\n')

