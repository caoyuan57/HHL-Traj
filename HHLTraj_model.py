import torch
import torch.nn as nn
import logging
from pars_args import args
import torch.nn.functional as F
import numpy as np
from layer import HGNN_conv
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

class HHLTraj(nn.Module):
    def __init__(self, args, device, batch_first=True):
        super(HHLTraj, self).__init__()
        self.nodes = args.nodes
        self.latent_dim = args.latent_dim
        self.device = device
        self.batch_first = batch_first
        self.dropout = 0.5
        self.hgc1 = HGNN_conv(128, 128)
        self.hgc2 = HGNN_conv(128, 128)
        # self.hgc3 = HGNN_conv(128, 128)
        # self.hgc4 = HGNN_conv(128, 128)
        # self.hgc5 = HGNN_conv(128, 128)
        # self.hgc6 = HGNN_conv(128, 128)
        # self.conv = nn.Linear(128, 128)
        self.hash_layer = nn.Tanh()
        logging.info('Initializing model: latent_dim=%d' % self.latent_dim)
        # self.gru = nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim,
        #             num_layers=1, batch_first=True)
        
        # self.hc1 = nn.Linear()
        self.lstm_list = nn.ModuleList([
            nn.GRU(input_size=self.latent_dim, hidden_size=self.latent_dim,
                    num_layers=1, batch_first=True)
            for _ in range(args.lstm_layers)
        ])

        self.poi_features = torch.randn(
            self.nodes, self.latent_dim).to(self.device)
    

    def forward(self, fps):
        Hbat,DV2,invDE,data,data_length = fps
        Hbat = Hbat.to(self.device)
        DV2 = DV2.to(self.device)
        invDE = invDE.to(self.device)
        
        # W = torch.ones(len(data)+1)
        # W[len(data)] = 0.01
        # W = W.to(self.device)
        # # print(self.poi_features)
        G1 = torch.mm(invDE.diag_embed(),Hbat)
        G = torch.mm(DV2.diag_embed(),Hbat.T)
        # G = torch.mm(G,W.diag_embed())
        G1 = torch.mm(G1,DV2.diag_embed())
        G = torch.mm(G,G1)
        # G2 = torch.mm(invDE.diag_embed(),Hbat)
        embedding_poi = self.poi_features
        # embedding_poi = F.relu(self.hgc3(embedding_poi, G)) + embedding_poi
        embedding_poi = F.relu(self.hgc1(embedding_poi, G)) + embedding_poi
        # embedding_poi = F.relu(self.hgc4(embedding_poi, G)) + embedding_poi
        # embedding_poi = F.relu(self.hgc5(embedding_poi, G))# + embedding_poi
        # embedding_poi = F.relu(self.hgc6(embedding_poi, G)) + embedding_poi
        # embedding_poi = F.relu(self.hgc2(embedding_poi, G)) + embedding_poi

        embedding_poi =  self.hgc2(embedding_poi, G)
        batch_emb = embedding_poi[data]
        batch_emb_pack = rnn_utils.pack_padded_sequence(
                batch_emb, data_length, batch_first=self.batch_first)
        # print(batch_emb_pack[0].size())

        for lstm in self.lstm_list[:-1]:
            out_emb, _ = lstm(batch_emb_pack)
            out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(
                out_emb, batch_first=self.batch_first)
            out_emb_pad = batch_emb + F.relu(out_emb_pad)
            batch_emb_pack = rnn_utils.pack_padded_sequence(
                out_emb_pad, out_emb_len, batch_first=self.batch_first)
            
        out_emb, _ = self.lstm_list[-1](batch_emb_pack)
        out_emb_pad, out_emb_len = rnn_utils.pad_packed_sequence(
            out_emb, batch_first=self.batch_first)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8,batch_first=True,device=self.device)
        # transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,device=self.device)
        # out_emb = transformer_encoder(batch_emb)
        # print(out_emb.shape)
        # out_emb_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(out_emb, batch_first=True)

        # out_emb, _ = self.gru(batch_emb_pack)
        # out_emb_pad, _ = torch.nn.utils.rnn.pad_packed_sequence(out_emb, batch_first=True)
        

        idx = (torch.LongTensor(data_length) - 1).view(-1, 1).expand(
                len(data_length), out_emb_pad.size(2))
        time_dimension = 1 if self.batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        if out_emb_pad.is_cuda:
            idx = idx.cuda(out_emb_pad.data.get_device())
        last_output_emb = out_emb_pad.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        # printcccc
        # last_output_emb = torch.Tensor(last_output_emb)
        # last_output_emb = self.conv(last_output_emb)
        last_output_emb = self.hash_layer(last_output_emb)
        return last_output_emb
        
