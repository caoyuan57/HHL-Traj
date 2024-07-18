from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from pars_args import args
import math

class MyData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.idx = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.label[idx], self.idx[idx])
        return tuple_


def load_traindata(train_file):
    data = np.load(train_file, allow_pickle=True)
    x = data['train_list']
    y_idx = data['train_y']
    return x, y_idx

def load_valdata(val_file):
    data = np.load(val_file, allow_pickle=True)
    x = data['val_list']
    y_idx = data['val_y']
    return x, y_idx

def load_testdata(test_file):
    data = np.load(test_file, allow_pickle=True)
    x = data['test_list']
    y_idx = data['test_y']
    return x, y_idx


def PE(idx):
    if idx % 2 == 0:
        return math.sin(idx/pow(1000,2*idx/100))/4
    else:
        return math.cos(idx/pow(1000,2*idx/100))/4
    
def construct_H(sums, datacell, batch_size):
    H = np.zeros((batch_size, sums))
    H = H
    for k, da in enumerate(datacell):
        for idx,d in enumerate(da):
            if d != -1:
                H[k][d]=H[k][d]+1#+PE(idx)
    # H[batch_size] = 1
    return torch.Tensor(H)

def generate_G_from_H(H):
    np.seterr(divide='ignore',invalid='ignore')
    H = H
    DV = torch.sum(H, axis=0)
    DE = torch.sum(H, axis=1)
    DV2 = torch.pow(DV.type(torch.float), -0.5)
    DV2[torch.isinf(DV2)] = 0
    DV2[torch.isnan(DV2)] = 0

    invDE = torch.pow(DE.type(torch.float), -1)
    invDE[torch.isinf(invDE)] = 0
    invDE[torch.isnan(invDE)] = 0

    return DV2,invDE
def TrainValueDataLoader(train_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        dataa = [torch.LongTensor(sq[0]) for sq in data_tuple]
        datab = [torch.LongTensor(sq[1]) for sq in data_tuple]
        dataa_length = [len(sq) for sq in dataa]
        datab_length = [len(sq) for sq in datab]
        Ha = construct_H(args.nodes,dataa,len(dataa))
        Hb = construct_H(args.nodes,datab,len(datab))
        HaDV2,HainvDE = generate_G_from_H(Ha)
        HbDV2,HbinvDE = generate_G_from_H(Hb)
        dataa = rnn_utils.pad_sequence(dataa, batch_first=True, padding_value=0)
        datab = rnn_utils.pad_sequence(datab, batch_first=True, padding_value=0)
        return Ha,Hb,HaDV2,HainvDE,HbDV2,HbinvDE,dataa,datab,dataa_length,datab_length

    a, b = load_traindata(train_file)
    data_ = MyData(a, b)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset

def ValValueaDataLoader(val_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        idx_list = torch.tensor([sq[2] for sq in data_tuple])
        data_length = [len(sq) for sq in data]
        Ha = construct_H(args.nodes,data,len(data))
        HaDV2,HainvDE = generate_G_from_H(Ha)
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        
        return idx_list,Ha,HaDV2,HainvDE,data,data_length

    val_x, val_y = load_valdata(val_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset


def ValValuebDataLoader(val_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        idx_list = torch.tensor([sq[2] for sq in data_tuple])
        data_length = [len(sq) for sq in data]
        Ha = construct_H(args.nodes,data,len(data))
        HaDV2,HainvDE = generate_G_from_H(Ha)
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        
        return idx_list,Ha,HaDV2,HainvDE,data,data_length

    val_x, val_y = load_valdata(val_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset


def TestValueaDataLoader(test_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        idx_list = torch.tensor([sq[2] for sq in data_tuple])
        data_length = [len(sq) for sq in data]
        Ha = construct_H(args.nodes,data,len(data))
        HaDV2,HainvDE = generate_G_from_H(Ha)
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        
        return idx_list,Ha,HaDV2,HainvDE,data,data_length

    val_x, val_y = load_testdata(test_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset


def TestValuebDataLoader(test_file, batchsize):
    def collate_fn_neg(data_tuple):
        data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
        data = [torch.LongTensor(sq[1]) for sq in data_tuple]
        idx_list = torch.tensor([sq[2] for sq in data_tuple])
        data_length = [len(sq) for sq in data]
        H = construct_H(args.nodes,data,len(data))
        HDV2,HainvDE = generate_G_from_H(H)
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        return idx_list, H, HDV2, HainvDE,data,data_length

    val_x, val_y = load_testdata(test_file)
    data_ = MyData(val_x, val_y)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset
