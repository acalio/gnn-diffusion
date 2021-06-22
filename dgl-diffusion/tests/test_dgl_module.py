import itertools
import os
from dgl_diffusion.util import load_cascades
from dgl_diffusion.data import  CascadeDataset, CascadeDatasetBuilder
import torch as th
import torch.nn as nn
import pytest
import dgl
from dgl_diffusion.model import *
from dgl_diffusion.persistance import PManager
from dgl_diffusion.util import *
from collections import OrderedDict
import pandas as pd

DATA_HOME = "/home/antonio/Garbage/data/"
INPUT_SIZE = 6

def setup(tw=1):
    builder = CascadeDatasetBuilder()
    graph_path = os.path.join(DATA_HOME, "networks/jazz/jazz.csv")
    cascade_path = os.path.join(DATA_HOME, "cascades/jazz/jazz_srange_1_10_degree_ic.txt")
    builder.graph_path = graph_path
    builder.cascade_path = cascade_path
    builder.max_cascade = 10
    builder.strategy = 'counting'
    builder.edge_weights_normalization = False
    builder.add_self_loops = True
    builder.training_size = 0.7
    builder.validation_size = 0.2
    builder.test_size = 0.1
    data = builder.build(time_window=tw)
    graph = data.enc_graph
    # create embedding layer
    embeddings = nn.Embedding(data.enc_graph.number_of_nodes(), INPUT_SIZE)
    feat = embeddings.weight
    nn.init.xavier_uniform_(feat)
    
    return data, embeddings


@pytest.mark.skip("already tested")
def test_split():
    data, _ = setup(1)
    dec_graph = data.dec_graph
    train_dec_graph = data.training_dec_graph
    val_dec_graph = data.validation_dec_graph
    test_dec_graph = data.test_dec_graph
    
    assert train_dec_graph.number_of_nodes() == dec_graph.number_of_nodes()
    print(dec_graph.number_of_edges())
    print(train_dec_graph.number_of_edges())
    print(val_dec_graph.number_of_edges())
    print(test_dec_graph.number_of_edges())

    
# @pytest.mark.skip("already tested")
def test_only_positive():
    data, embeddings = setup(3)
    feat = embeddings.weight
    graph = data.enc_graph
    w = data.training_inf_graph.edata['w']
    u, v = data.training_inf_graph.edges()
    encoder = InfluenceEncoder([INPUT_SIZE, 12, 6], "tanh", "tanh", .2, "cpu")
    decoder = InfluenceDecoder(get_architecture([12, 24, 1], ["tanh", "relu"]))
    net = InfEncDec(encoder, decoder)
    loss_fn = get_loss("mse", "mean")
    evaluation_fn = get_loss("mse", "mean")
    opt = get_optimizer("adam")(list(net.parameters())+list(embeddings.parameters()), lr=0.01)
    for i in range(5000):
        net.train()
        pred, _ = net(data.enc_graph, feat, data.training_inf_graph)
        pred = pred.squeeze()
        loss_value = loss_fn(pred, w)
        opt.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(itertools.chain(
                net.parameters(), embeddings.parameters()), 1)
        opt.step()
        if (i+1)%100==0:
            print(f'{loss_value.item()}, {evaluation_fn(pred, w).item()}')
        if (i+1)%200==0:
            net.eval()
            val_pred, _ = net(data.enc_graph, feat, data.validation_inf_graph)
            val_pred = val_pred.squeeze()
            val_loss  = loss_fn(val_pred, data.validation_inf_graph.edata['w'])
            print(f"\033[91mVal-loss:{val_loss.item()}\033[0m'")


    print(pred[:20], w[:20])


@pytest.mark.skip("already tested")
def test_negative_positive():
    data, embeddings = setup()
    feat = embeddings.weight
    graph = data.enc_graph
    neg_graph = data.get_negative_graph(1)
    encoder = InfluenceEncoder(6, 24, 6, "relu", "relu", "cpu")
    decoder = InfluenceDecoder(get_architecture([12, 24, 1], ["sigmoid", "relu"]))
    net = InfEncDec(encoder, decoder)
    loss_fn = get_loss("mse", "mean")
    opt = get_optimizer("adam")(list(net.parameters())+list(embeddings.parameters()), lr=0.001)
    net.train()
    w = data.dec_graph.edata['w']
    u, v = data.dec_graph.edges()
    print(w[:10], u[:10], v[:10])
    labels = th.cat([w, th.zeros(neg_graph.number_of_edges())])
    assert labels.shape[0]==(w.shape[0] + neg_graph.number_of_edges())
    print(f"pos_graph edges: {data.dec_graph.number_of_edges()}\nNegative Graph {neg_graph.number_of_edges()}")
    for i in range(100):
        pos_pred, neg_pred = map(lambda t: t.squeeze(), net(data.enc_graph, feat, data.dec_graph, neg_graph))
        pred = th.cat([pos_pred, neg_pred])

        loss_value = loss_fn(pred, labels)
        opt.zero_grad()
        loss_value.backward()
        nn.utils.clip_grad_norm_(itertools.chain(
                net.parameters(), embeddings.parameters()), 1)
        opt.step()
        if (i+1)%10==0:
            print(loss_value.item())

    print(pred[:20], w[:20])


