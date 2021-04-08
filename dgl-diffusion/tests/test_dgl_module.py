import os
from dgl_diffusion.util import load_cascades
from dgl_diffusion.data import  CascadeDataset
import torch as th
import torch.nn as nn
import pytest
import dgl
from dgl_diffusion.model import *
from dgl_diffusion.persistance import PManager
from collections import OrderedDict
import pandas as pd

DATA_HOME = "/home/antonio/git/gnn-diffusion/data/"


def setup():
    graph_path = os.path.join(DATA_HOME, "networks/nethept/graph_ic.inf")
    cascade_path = os.path.join(DATA_HOME, "cascades/nethept/prova1.txt")
    data = CascadeDataset(graph_path, cascade_path)
    graph = data.enc_graph

    # create embedding layer
    embeddings = nn.Embedding(data.enc_graph.number_of_nodes(), 124)
    feat = embeddings.weight
    nn.init.xavier_uniform_(feat)

    return data, feat


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

@pytest.mark.skip("already tested")
def test_encoder_decoder():
    data, feat = setup()
    graph = data.enc_graph
    ilayer = InfluenceEncoder(124, 124, 124, "relu", "relu", "cpu")
    seq_dict = OrderedDict([
        ('linear1', nn.Linear(124*2, 124)),
        ('relu1', nn.ReLU()),
        ('linear2', nn.Linear(124, 1)),
        ('sigmoid2', nn.Sigmoid())
    ])

    net = InfEncDec(124, 124, 124, 'relu', 'relu', seq_dict)

    pred = net(graph, feat)
    print(pred.squeeze().shape)


def test_persistence():
    p = PManager("/home/antonio/Garbage/")
    p.hash("prova")
    p.hash("prova", "provata")
    l = [1, 2, 3]
    arch = ["relu", "ciao"]

    def info(f):
        f.write("Layers\n")
        f.write(str(l))
        f.write("Arch")
        f.write(str(arch))
    p.persist(("plist", info))
    
