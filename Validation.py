import numpy as np
import torch
import random
import os
import logging
from logg import setup_logger
from pars_args import args
from HHLTraj.HHLTraj_model import HHLTraj
from tqdm import tqdm
from Data_Loader import load_valdata
from Data_Loader import ValValueaDataLoader
from Data_Loader import ValValuebDataLoader
import torch.nn.functional as F
def cosine_similarity(matrix1, matrix2):
    # 归一化输入矩阵
    matrix1_normalized = F.normalize(matrix1, p=2, dim=1)
    matrix2_normalized = F.normalize(matrix2, p=2, dim=1)

    # 计算余弦相似度
    similarity = torch.matmul(matrix1_normalized, matrix2_normalized.t())

    return similarity

# def recall(s_emb, train_emb, label, device, K=[1, 5, 10, 20, 50]):
#     r = cosine_similarity(s_emb,train_emb)
#     label_r = torch.argsort(r, dim=1, descending=True)
#     recall = torch.zeros((s_emb.shape[0], len(K)), device=device)
#     for idx, la in enumerate(label):
#         for idx_k, k in enumerate(K):
#             if la in label_r[idx, :k]:
#                 recall[idx, idx_k:] = 1
#                 break
#     return recall

def recall(s_emb, train_emb, label, device, K=[1, 5, 10, 20, 50]):
    # 计算余弦相似度
    r = cosine_similarity(s_emb,train_emb)
    
    # 找到前K个最相似的索引
    _, label_r = torch.topk(r, max(K), dim=1, largest=True)
    
    # 初始化召回率结果
    recall = torch.zeros((s_emb.shape[0], len(K)), device=device)
    
    # 将真实标签扩展为与label_r相同形状，以便向量化比较
    expanded_label = label.view(-1, 1).expand(-1, max(K)).to(device)
    # 检查真实标签是否在前K个索引中
    for idx_k, k in enumerate(K):
        recall[:, idx_k] = torch.any(expanded_label[:, :k] == label_r[:, :k], dim=1).float()

    return recall


def eval_model():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    train_x, _ = load_valdata(args.val_file)

    emb_train = torch.zeros((len(train_x), args.latent_dim),
                            device=device, requires_grad=False)

    K = [1, 5, 10, 20, 50]
    model = HHLTraj(args, device, batch_first=True).to(device)

    rec = torch.zeros((train_x.shape[0], len(K)),
                      device=device, requires_grad=False)

    data_loader_train = ValValueaDataLoader(args.val_file, args.batch_size)
    data_loader_val = ValValuebDataLoader(args.val_file, args.batch_size)

    max_rec_v = -1
    max_epoch = -1

    for epoch in range(0, 50):
        model_name = 'epoch_' + str(epoch) + '.pt'

        model_f = '%s/%s' % (args.save_folder, model_name)
        if not os.path.exists(model_f):
            continue

        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f, map_location=device))
        model.eval()

        pbar = tqdm(data_loader_train)

        for batch_id, (idx_list,Ha,HaDV2,HainvDE,data,data_length) in enumerate(pbar):
            last_output_emba = model((Ha, HaDV2, HainvDE,data,data_length))
            # emb_train[idx_list, :] = last_output_emba.detach()
            emb_train[idx_list, :] = torch.sign(last_output_emba).detach()

        pbar = tqdm(data_loader_val)

        for batch_id, (idx_list,H, HDV2, HinvDE,data,data_length) in enumerate(pbar):
            last_output_emb = model((H, HDV2, HinvDE,data,data_length))
            last_output_emb = torch.sign(last_output_emb)
            
            # print(torch.sign(last_output_emb).detach())
            # last_output_emb = torch.nn.Tanh(last_output_emb)
            rec[idx_list, :] = recall(
                last_output_emb, emb_train, idx_list, device, K).detach()

        rec_ave = rec.mean(axis=0)
        for recs in rec_ave:
            logging.info('%.4f' % recs)

            if recs > max_rec_v:
                max_rec_v = recs
                max_epoch = epoch

    logging.info(str(max_epoch))


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('HHLTraj_eva.log')
    eval_model()
