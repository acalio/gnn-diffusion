import os
from dgl_diffusion.util import load_cascades
from dgl_diffusion.data import  CascadeDataset, CascadeDatasetBuilder
import torch as th
import torch.nn as nn
import pytest
import dgl
from dgl_diffusion.model import *
from dgl_diffusion.persistance import PManager
from dgl_diffusion.util import get_architecture
from collections import OrderedDict
import pandas as pd

DATA_HOME = "/home/antonio/git/gnn-diffusion/data/"


def setup():
    builder = CascadeDatasetBuilder()
    graph_path = os.path.join(DATA_HOME, "networks/jazz/jazz.csv")
    cascade_path = os.path.join(DATA_HOME, "cascades/jazz/jazz_srange_1_10_degree_ic.txt")
    builder.graph_path = graph_path
    builder.cascade_path = cascade_path
    builder.max_cascade = 10
    builder.strategy = 'counting'
    
    data = builder.build(time_window=2)
    graph = data.enc_graph

    # create embedding layer
    embeddings = nn.Embedding(data.enc_graph.number_of_nodes(), 12)
    feat = embeddings.weight
    nn.init.xavier_uniform_(feat)

    return data, embeddings


@pytest.mark.skip("already tested")
def test_graphConv():
    data, feat = setup()
    graph = data.enc_graph
    conv = InfluenceGraphConv(124, 124, "cpu")
    out_feat = conv(data.enc_graph, feat)
    nxg = dgl.to_networkx(graph, edge_attrs=['w'])
    weights = [(u, d['w'].item()) for u, _, d in nxg.in_edges(2492, data=True)]
    feat_2492 = feat[2492].detach().clone()
    feat_2492 = feat_2492 @ conv.weight
    messages = th.zeros_like(feat_2492)
    for u, w in weights:
        messages = messages + feat[u] @ conv.weight

    # something has happened
    assert th.norm(feat[2492]-out_feat[2492]).item() > 0
    assert th.norm(messages-out_feat[2492]).item() < 0.00001  # as expected


@pytest.mark.skip("already tested")
def test_infLayer():
    data, feat = setup()
    graph = data.enc_graph
    ilayer = InfluenceEncoder(124, 124, 124, "relu", "relu", "cpu")
    conv = ilayer.conv_layer

    out_feat = ilayer(data.enc_graph, feat)

    nxg = dgl.to_networkx(graph, edge_attrs=['w'])
    weights = [(u, d['w'].item()) for u, _, d in nxg.in_edges(2492, data=True)]
    feat_2492 = feat[2492].detach().clone()
    feat_2492 = feat_2492 @ conv.weight
    messages = th.zeros_like(feat_2492)
    for u, w in weights:
        messages = messages + feat[u] @ conv.weight

    # something has happened
    assert th.norm(feat[2492]-out_feat[2492]).item() > 0


@pytest.mark.skip("already tested")
def test_decoder():
    data, feat = setup()
    graph = data.enc_graph
    ilayer = InfluenceEncoder(124, 124, 124, "relu", "relu", "cpu")
    seq_dict = OrderedDict([
        ('linear1', nn.Linear(124*2, 124)),
        ('relu1', nn.ReLU()),
        ('linear2', nn.Linear(124, 1)),
        ('sigmoid2', nn.Sigmoid())
    ])
    seq = nn.Sequential(seq_dict)
    deco = InfluenceDecoder(seq)
    conv = ilayer.conv_layer

    out_feat = ilayer(data.enc_graph, feat)

    pred = deco(graph, out_feat)
    print(pred.squeeze().shape)

#@pytest.mark.skip("already tested")
def test_encoder_decoder():
    data, embeddings = setup()
    feat = embeddings.weight
    embeddings.zero_grad()
    graph = data.enc_graph
    # encoder = InfluenceEncoder(12, 12, 12, "relu", "relu", "cpu")
    decoder = InfluenceDecoder(get_architecture([24, 12, 1], ["relu", "sigmoid"]))
    pred = decoder(graph, feat)
    print(pred.is_leaf)
    print(feat.grad)
    loss = pred.sum() + 100
    loss.backward()
    print(feat.grad)
    assert False
    # net = InfEncDec(encoder, decoder)
    # pred = encoder(graph, feat).sum()
    # assert pred.backward()



    
