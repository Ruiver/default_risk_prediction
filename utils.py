import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    if  temp_chars:
        if len(temp_tags) > 1 and temp_tags[0] == '4' and temp_tags.count('5') == (len(temp_tags) - 1):
            EVA.append(temp_chars)
            ALL.append(temp_chars)
        elif len(temp_tags) > 1 and temp_tags[0] == '1' and temp_tags.count('2') == (len(temp_tags) - 1):
                ENT.append(temp_chars)
                ALL.append(temp_chars)

    return ENT, EVA, ALL

def get_ENT_entity(tag_seq, char_seq):
    length = len(char_seq)
    ENT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'S-ENT':
            ENT.append(char)
            del ent
            continue
        if tag == 'B-ENT':
            if 'ent' in locals().keys():
                ENT.append(ent)
                del ent
            ent = char
            if i+1 == length:
                ENT.append(ent)
        if tag == 'I-ENT':
            ent += char
            if i+1 == length:
                ENT.append(ent)
        if tag not in ['I-ENT', 'B-ENT']:
            if 'ent' in locals().keys():
                ENT.append(ent)
                del ent
            continue
    return ENT


def get_EVA_entity(tag_seq, char_seq):
    length = len(char_seq)
    EVA = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'S-EVA':
            EVA.append(char)
            del eva
            continue
        if tag == 'B-EVA':
            if 'eva' in locals().keys():
                EVA.append(eva)
                del eva
            eva = char
            if i+1 == length:
                EVA.append(eva)
        if tag == 'I-EVA':
            eva += char
            if i+1 == length:
                EVA.append(eva)
        if tag not in ['I-EVA', 'B-EVA']:
            if 'eva' in locals().keys():
                EVA.append(eva)
                del eva
            continue
    return EVA




def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
