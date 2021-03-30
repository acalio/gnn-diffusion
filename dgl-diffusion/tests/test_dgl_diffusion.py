from dgl_diffusion.util import load_cascades
from dgl_diffusion.data import CascadeDataset
import torch as th
import pytest
import dgl


NETWORK_PATH = "/home/antonio/git/gnn-diffusion/data/networks/nethept/graph_ic.inf"
CASCADE_PATH = "/home/antonio/git/gnn-diffusion/data/cascades/nethept/prova.txt"


@pytest.mark.skip("already tested")
def test_version():
    cascades = load_cascades(CASCADE_PATH)
    # print(cascades)
    assert True


@pytest.mark.skip("already tested")
def test_counting_weight():
    cascades = [
        [[1], [2, 3], [4, 5]],
        [[2], [1, 3], [4, 5]]
    ]
    c = CascadeDataset(
    )
    coord = c.counting_weight(cascades)

    for x in coord:
        print(f'{x}:{coord[x]}')
        print("="*50)


@pytest.mark.skip("already tested")
def test_tempdiff_weight():
    cascades = [
        [[1], [2, 3], [4, 5]],
        [[2], [1, 3], [4, 5]]
    ]
    c = CascadeDataset(NETWORK_PATH, CASCADE_PATH)
    coord = c.tempdiff_weight(cascades)
    for x in coord:
        print(f'{x}:{coord[x]}')
        print("="*50)


@pytest.mark.skip("already tested")
def test_graph_creation_counting():
    cascades_path = "/home/antonio/git/gnn-diffusion/data/cascades/nethept/prova1.txt"
    cascades = load_cascades(cascades_path)
    c = CascadeDataset(NETWORK_PATH, CASCADE_PATH)

    # find the max in the cascade
    max_node = 0
    for cascade in cascades:
        for t, tlist in enumerate(cascade):
            max_node = max(max_node, max(tlist))

    assert c.train_enc_graph.nodes()[-1].item() == max_node
    # convert to networkx
    g = dgl.to_networkx(c.train_enc_graph, edge_attrs=['w'])
    # print(g.edges(data=True))


@pytest.mark.skip("already tested")
def test_graph_creation_tempdiff():
    cascades = load_cascades(CASCADE_PATH)

    c = CascadeDataset(NETWORK_PATH, CASCADE_PATH)
    # find the max in the cascade
    max_node = 0
    for cascade in cascades:
        for t, tlist in enumerate(cascade):
            max_node = max(max_node, max(tlist))

    assert c.train_enc_graph.nodes()[-1].item() == max_node
    # convert to networkx
    g = dgl.to_networkx(c.train_enc_graph, edge_attrs=['w'])
    print(g.edges(data=True))


def test_graph_negative():
    c = CascadeDataset(NETWORK_PATH, CASCADE_PATH)
    neg_graph = c.get_target_negative_graph(5)
    src_tensor, dst_tensor, eid_tensor = neg_graph.edges("all")
    nxg = dgl.to_networkx(c.dec_graph, edge_attrs=['w'])
    for src, dst, eid in zip(src_tensor, dst_tensor, eid_tensor):
        src, dst, eid = map(lambda x : x.item(), (src, dst, eid))
        try:
            assert nxg[src][dst][0]['w'].item() == neg_graph.edata['w'][eid].item()
        except KeyError:
            assert neg_graph.edata['w'][eid].item() == 0



