import pickle
import dgl
import numpy as np
import torch
import random
import os
import logging
from collections import defaultdict as ddict


def get_g(tri_list):
    triples = np.array(tri_list)
    g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
    g.edata['rel'] = torch.tensor(triples[:, 1].T)
    return g


def get_g_bidir(triples, args):
    g = dgl.graph((torch.cat([triples[:, 0].T, triples[:, 2].T]),
                   torch.cat([triples[:, 2].T, triples[:, 0].T])))
    g.edata['type'] = torch.cat([triples[:, 1].T, triples[:, 1].T + args.num_rel])
    return g


def get_hr2t_rt2h(tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    return hr2t, rt2h

def get_hr2t_rt2h_sup_que(sup_tris, que_tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in sup_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    for tri in que_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    que_hr2t = dict()
    que_rt2h = dict()
    for tri in que_tris:
        h, r, t = tri
        que_hr2t[(h, r)] = hr2t[(h, r)]
        que_rt2h[(r, t)] = rt2h[(r, t)]

    return que_hr2t, que_rt2h


def get_indtest_test_dataset_and_train_g(args):
    data = pickle.load(open(args.data_path, 'rb'))['ind_test_graph']
    num_ent = len(np.unique(np.array(data['train'])[:, [0, 2]]))

    hr2t, rt2h = get_hr2t_rt2h(data['train'])

    from datasets import KGEEvalDataset
    test_dataset = KGEEvalDataset(args, data['test'], num_ent, hr2t, rt2h)

    g = get_g_bidir(torch.LongTensor(data['train']), args)

    return test_dataset, g


def get_posttrain_train_valid_dataset(args):
    data = pickle.load(open(args.data_path, 'rb'))['ind_test_graph']
    num_ent = len(np.unique(np.array(data['train'])[:, [0, 2]]))

    hr2t, rt2h = get_hr2t_rt2h(data['train'])

    from datasets import KGETrainDataset, KGEEvalDataset
    train_dataset = KGETrainDataset(args, data['train'],
                                    num_ent, args.posttrain_num_neg, hr2t, rt2h)

    valid_dataset = KGEEvalDataset(args, data['valid'], num_ent, hr2t, rt2h)

    return train_dataset, valid_dataset


def get_num_rel(args):
    data = pickle.load(open(args.data_path, 'rb'))
    num_rel = len(np.unique(np.array(data['train_graph']['train'])[:, 1]))

    return num_rel


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    data_tuple = pickle.loads(data)
    return data_tuple


def set_seed(seed):
    dgl.seed(seed)
    dgl.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


class Log(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        # file handler
        log_file = os.path.join(log_dir, name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # console handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        fh.close()
        sh.close()

    def get_logger(self):
        return self.logger

class FileLog(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        # file handler
        log_file = os.path.join(log_dir, name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

        fh.close()

    def get_logger(self):
        return self.logger