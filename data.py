import sys, pickle, os, random
import numpy as np
from gensim.models import Word2Vec
import multiprocessing

## tags, BIO
tag2label = {"O": 0,
             "B-ENT": 1, "I-ENT": 2,
             "S-ENT": 3, "B-EVA": 4,
             "I-EVA": 5, "S-EVA": 6
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    num = 0
    for line in lines:
        if line != '\n':
            [char, label] = line.replace("\n", "").split("\t")
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """

    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


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


def random_embedding(word2id, embedding_dim, data_path):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    if os.path.isfile('data_path/Word2vec_model.pkl'):
        print('load the w2v_model')
        model = Word2Vec.load('data_path/Word2vec_model.pkl')
    else:
        with open(data_path, 'r', encoding='utf8') as f:
            data = []
            for line in f.readlines():
                sentence = []
                for word in line.split('\t')[0]:
                    # 这里不需要UNK，因为这里是训练word2vec模型的，文本中是没有UNK这种的，别个sen2id搞混了
                    if word.isdigit():
                         word = '<NUM>'
                    sentence.append(word)
                data.append(sentence)
            model = Word2Vec(data, size=embedding_dim,
                             workers=multiprocessing.cpu_count())
            print('saving the w2v.model')
            model.save('data_path/Word2vec_model.pkl')
    embedding_mat = np.zeros((len(word2id), embedding_dim))
    for word, id in word2id.items():
        if word != '<UNK>' and word != '<PAD>' and word != '<UNK>':
            embedding_mat[id] = model[word]
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        #seq后面没必要接[:max_len]呀，不知道为什么这么做
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels
