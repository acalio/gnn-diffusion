"""Training the model
"""
import os
import time
import tqdm
from dgl_diffusion.data import CascadeDataset
from dgl_diffusion.model import InfluenceDecoder, InfluenceLayer, InfEncDec
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl_diffusion.util import get_activation, get_optimizer, construct_negative_graph
from collections import OrderedDict

NETWORK_PATH = "/home/antonio/git/gnn-diffusion/data/networks/nethept/graph_ic.inf"
CASCADE_PATH = "/home/antonio/git/gnn-diffusion/data/cascades/nethept/prova.txt"

def main():
    seq_dict = OrderedDict([
        ('linear1', nn.Linear(124*2, 124)),
        ('relu1', nn.ReLU()),
        ('linear2', nn.Linear(124, 1)),
        ('sigmoid2', nn.Sigmoid())
    ])

    model = InfEncDec(124, 124, 124, 'relu', 'relu', seq_dict)


    data = CascadeDataset(NETWORK_PATH, CASCADE_PATH)

    net = InfEncDec(124, 124, 124, 'relu', 'sigmoid', seq_dict)
    learning_rate = 0.0001
    opt = get_optimizer('adam')(net.parameters(), lr=learning_rate)

    # contruct negative graph
    target_graph = data.get_target_negative_graph(5)
    labels = target_graph.edata['w']

    # embeddings
    embeddings = nn.Embedding(data.enc_graph.number_of_nodes(), 124)
    feat = embeddings.weight
    nn.init.xavier_uniform_(feat)

    max_epoch = 100
    with tqdm.trange(max_epoch) as pbar:
        for epoch in pbar:
            net.train()
            predictions = net(data.enc_graph, feat)
            # total scores
            loss = F.mse_loss(predictions, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            pbar.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)


if __name__ == '__main__':
    main()
