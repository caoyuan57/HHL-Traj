import logging

import os
import numpy as np
import torch
import torch.optim as optim
from pars_args import args
from tqdm import tqdm
import torch.nn.functional as F
torch.set_printoptions(precision=4,sci_mode=False)
def cosine_similarity(matrix1, matrix2):
    # # 归一化输入矩阵
    # matrix1 = F.normalize(matrix1, dim=1)
    # matrix2 = F.normalize(matrix2, dim=1)

    # 计算余弦相似度
    similarity = torch.mm(matrix1, matrix2.t())
    # similarity = similarity / 2
    similarity = torch.clamp(similarity, min = 0.001, max= 50)
    return similarity
class Trainer:
    def __init__(self, model, train_data_loader, val_data_loader, n_epochs, lr,
                 save_epoch_int, model_folder, device):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(self.device)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _pass(self, data, train=True):
        self.optim.zero_grad()

        Ha,Hb,HaDV2,HainvDE,HbDV2,HbinvDE,dataa,datab,dataa_length,datab_length= data
        dataa = dataa.to(self.device)
        datab = datab.to(self.device)

        last_output_a_emb = self.model((Ha,HaDV2,HainvDE,dataa,dataa_length))
        
        last_output_b_emb = self.model((Hb,HbDV2,HbinvDE,datab,datab_length))
        # predictiond = F.cosine_similarity(last_output_a_emb, last_output_b_emb)
        prediction = cosine_similarity(last_output_a_emb, last_output_b_emb)
        #cosine_similarity
        # print(prediction)
        row_sum = torch.exp(prediction).sum(dim=1)
        d_sum = torch.exp(prediction).sum(dim=0)
        # print(row_sum)
        L1 = torch.exp(prediction.diag())/row_sum
        L1 = -L1.log().sum()
        L2 = torch.exp(prediction.diag())/d_sum
        L2 = -L2.log().sum()
        L3 = (1 - abs(last_output_a_emb)).sum()/args.latent_dim

        # loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        # loss1 = loss.mean()
        # loss2 = config["alpha"] * (1 - u.abs()).abs().mean()
        
        # L4 = abs(last_output_a_emb.sum()/args.latent_dim)
        # L5 =  - (predictiond.sigmoid().log().sum())
        # print(L1)
        # print(prediction.diag())
        # if torch.any(torch.isnan(L1)) == True:
        #     print(L1)
        #     print(prediction.diag())
        #     print(torch.exp(prediction.diag()))
        #     print(row_sum)
        loss = L1 + L2 + L3
        if train:
            torch.backends.cudnn.enabled = False
            loss.backward()
            self.optim.step()

        return loss.item()

    def _train_epoch(self):
        self.model.train()
        
        losses = []
        pbar = tqdm(self.train_data_loader)
        for data in pbar:
            loss = self._pass(data)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % loss)

        return np.array(losses).mean()

    def _val_epoch(self):
        self.model.eval()

        losses = []

        pbar = tqdm(self.val_data_loader)
        for data in pbar:
            loss = self._pass(data, train=False)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % loss)

        return np.array(losses).mean()

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch()
            logging.info(
                '[Epoch %d/%d] [training loss: %f]' %
                (epoch, self.n_epochs, train_loss)
            )
            
            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file)

