#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Zhang Shuai'

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


result = open('../result.txt', 'r', encoding='utf8')
result_new = open('result_new.txt', 'w', encoding='utf8')
words_handle = open('words.txt', 'w', encoding='utf8')
with open('../content.txt', encoding='utf8') as f:
    count = 0
    error_count = 0
    r = result.readlines()
    print(len(r))
    words_s = set()
    for line in f.readlines():
        if '\2' in line or '\1' in line:
            result_new.write(line)
        else:
            sen = list(line.strip()[:800])
            label = len(line.strip()) * ["O"]
            if len(label) == 0:
                continue
            # result_new.write(line)
            data_,label_ = r[count].strip().split('\t')
            label_list = []
            tag = [label for label in label_]
            demo_sent = list(data_.strip())
            try:
                ENT, EVA, ALL = get_entity(tag, demo_sent)
                for eva in EVA:
                    words_s.add(eva)
                result_new.write(data_.strip() + '\t' + label_.strip()+ '\t' +'ENT: {} EVA: {}  ALL: {}\n'.format(ENT, EVA, ALL))
            except:
                error_count += 1
                print(data_.strip())
                print(label_.strip())
            count += 1
    for word in words_s:
        words_handle.write(word+'\n')
    print(error_count)





