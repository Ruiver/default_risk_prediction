import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding, vocab_build


## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=25, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo/all/all_2')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


## get char embeddings
if not os.path.exists(os.path.join('.', args.train_data, 'word2id.pkl')):
    vocab_build(os.path.join('.', args.train_data, 'word2id.pkl'), os.path.join('.', args.train_data, 'train_data'), 5)
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))

if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim, os.path.join('.', args.train_data, 'all_test'))
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)
    test_data = read_corpus(test_path)
    test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config, on_train=True)
    model.build_graph()

    # hyperparameters-tuning, split train/dev
    dev_data = test_data; dev_size = len(dev_data)
    train_data = train_data; train_size = len(train_data)
    print("train data: {0}\ndev data: {1}".format(train_size, dev_size))

    # ckpt_file = r'.\data_path_save\1527663228\checkpoints\model-17136'
    # paths['model_path'] = ckpt_file
    # print(ckpt_file)
    model.train(train=train_data, dev=dev_data)

    # ## train model on the whole training data
    # print("train data: {}".format(len(train_data)))
    # model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    # ckpt_file = r'.\data_path_save\1527697768\checkpoints\model-19992'
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print('ckpt_file:',ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                ENT, EVA, ALL = get_entity(tag, demo_sent)
                print('ENT: {}\nEVA: {}\nALL: {}\n'.format(ENT, EVA, ALL))
elif args.mode == 'all':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print('ckpt_file:',ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        result = open('result.txt', 'w',encoding='utf8')
        with open('content.txt', encoding='utf8') as f:
            count = 0
            error_count = 0
            for line in f.readlines():
                try:
                    if '\2' in line or '\1' in line:
                        result.write(line)
                    else:
                        count += 1
                        demo_sent = list(line.strip())
                        demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                        tag = model.demo_one(sess, demo_data)
                        ENT, EVA, ALL = get_entity(tag, demo_sent)
                        # print('ENT: {} EVA: {}\n'.format(ENT, EVA))
                        tag = [str(i) for i in tag]
                        result.write(str(line.strip()) + '\t' + "".join(tag)+ '\t' +'ENT: {} EVA: {} ALL: {}\n'.format(ENT, EVA, ALL))
                except Exception as e:
                    error_count += 1
                    print(line)
                    print(tag)
                    print(e)
                    print("count is ", count, "error count is", error_count)
                    print('---------------------------')
elif args.mode == 'all_2':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print('ckpt_file:',ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        result = open('result.txt', 'w',encoding='utf8')
        with open('content.txt', encoding='utf8') as f:# Modify here to use all posts and comments
            count = 0
            error_count = 0
            data =[]
            for line in f.readlines():
                if '\2' in line or '\1' in line:
                    # result.write(line)
                    pass
                else:
                    count += 1
                    sen = list(line.strip()[:800])
                    label = len(line.strip())*["O"]
                    if len(label) == 0:
                        # print(line.strip(),len(line),)
                        # print('$$$$$$$$$$$$$$$$$$')
                        continue
                    data.append((sen, label))
            tags = model.demo_many(sess, data)
            for (sen, _), tag in zip(data, tags):
                result.write("".join(sen) + '\t' + "".join([str(t) for t in tag]) + '\n')
                # print("".join(sen) + '\t' + "".join([str(t) for t in tag]))


