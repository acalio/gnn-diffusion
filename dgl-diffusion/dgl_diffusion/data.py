from os.path import splitext
from dgl_diffusion.util import load_cascades
from collections import defaultdict
from numpy.linalg import norm
import pickle
import torch as th
import dgl
import networkx as nx
from functools import reduce
from collections import deque
from copy import copy


class CascadeDataset:
    """Cascade dataset.
    This class is responsible for loading the
    raw data related to the cascades of a particular
    graph.

    Parameters
    ----------
    graph_path : str
      path to the influence graph

    cascade_path : str
      path for loading the cascades

    strategy : {'counting', 'tempdiff'}, default='counting'
      strategy for the determining the graph edge weights

      - 'counting' each edge weight w_{uv} denotes the number of times
        u is active before v in the same cascade

      - 'tempdiff' each edge weight w_{uv} denotes the norm of the
        following vector:
        .. math::
             \Delta_{uv} = \langle \delta^{1}_{uv}, \delta^{2}_{uv}, \dots, \delta^{m}_{uv} \rangle

         where,
        .. math::
            \delta^{i}_{uv} denotes the temporal difference between u and v activation
    Attributes
    ----------
    enc_graph : dgl.Graph
      graph reconstructed starting from the cascades provided as input

    dec_graph : dgl.Graph
      original influence graph (ground truth)
    """

    def __init__(self, graph_path,
                 cascade_path,
                 strategy='counting',
                 max_cascade=-1,
                 cascade_randomness=False,
                 save_cascade = None,
                 **kwargs):
        # get the strategy for the weights initialization
        strategy_fn = {
            'counting': self.counting_weight,
            'tempdiff': self.tempdiff_weight}[strategy]

        # check if the file is in pickle format
        _, cascade_format = splitext(cascade_path)
        if cascade_format == ".pickle":
            #load from pickle
            with open(cascade_path,'rb') as f:
                cascades = pickle.load(f)
        else:
            # load from the text file
            cascades = load_cascades(cascade_path, max_cascade=max_cascade,
                                 randomness=cascade_randomness)

        if save_cascade is not None:
            # save the cascades in pickle format
            with open(save_cascade, 'wb') as f:
                pickle.dump(cascades, f)
                
            
        # create the enc graph
        coordinates_dict = strategy_fn(cascades, **kwargs)
        self.enc_graph = self.get_graph(coordinates_dict)

        # read the influence graph
        inf_graph = nx.read_weighted_edgelist(
            graph_path, nodetype=int, create_using=nx.DiGraph())
        self.dec_graph = dgl.from_networkx(inf_graph, edge_attrs=['weight'])
        # rename edata weight to w
        self.dec_graph.edata['w'] = self.dec_graph.edata['weight']
        del self.dec_graph.edata['weight']

    def counting_weight(self, cascades, time_window=0):
        """ Counting strategy

        The correlations between any pair of nodes
        activation is considered only within a particular
        time frame.

        For example, if a node is active at time t, it will be
        accountable for any further activation up until
        t + time_window

        If time_window is 0 then any node is accountable
        for every subsequent activations

        Parameters
        ----------
        cascades : list of cascades 
          each cascade is a list of list of int (node indexes)

        time_window : int, optional, default 0
          time window

        Returns
        -------
        coordinates_dict : dict of dict
          each entry <u,v> denotes the weight of the corresponding edge
        """
        coordinates_dict = defaultdict(lambda: defaultdict(int))

        def inc(u, v):
            """Increment the counter
            at the given coordinates
            """
            coordinates_dict[u][v] += 1

        time_aware = time_window != 0
        # iterate over every cascade
        for cascade in cascades:
            # set the initial set of active nodes
            active_nodes = copy(cascade[0])
            # list containing the size of the last time_window activations
            backward_window = deque(
                [time_aware*len(active_nodes)] + [0]*(time_window-1))
            # iterate for every time step of this cascade
            for tlist in cascade[1:]:
                # number of nodes activated in the last time_window time steps
                back_limit = reduce(lambda x, y: x + y, backward_window)

                _ = [inc(u, v) for u in active_nodes[-back_limit:]
                     for v in tlist]
                # add the nodes to the set of active_nodes
                _ = [active_nodes.append(v) for v in tlist]

                backward_window.pop()
                backward_window.appendleft(time_aware*len(tlist))

        return coordinates_dict

    def tempdiff_weight(self, cascades, time_window=0):
        """Temporal difference strategy

        The correlations between any pair of nodes
        activation is considered only within a particular
        time frame.

        For example, if a node is active at time t, it will be
        accountable for any further activation up until
        t + time_window

        If time_window is 0 then any node is accountable
        for every subsequent activations

        Parameters
        ----------
        cascades : list of cascades 
          each cascade is a list of list of int (node indexes)

        time_window : int, optional, default 0
          time window

        Returns
        -------
         dict of dict
          each entry <u,v> denotes the weight of the corresponding edge
        """
        coordinates_dict = defaultdict(lambda: defaultdict(list))

        def append(u, v, t):
            """Add a new temporal difference value
            at the given coordinates
            """
            coordinates_dict[u][v].append(t)

        time_aware = time_window != 0
        # iterate over every cascade
        for cascade in cascades:
            # set the initial set of active nodes -
            # activation time step is 0
            active_nodes = {x: 0 for x in cascade[0]}
            # list containing the size of the last time_window activations
            active_nodes_list = copy(cascade[0])
            backward_window = deque(
                [time_aware*len(active_nodes_list)] + [0]*(time_window-1))
            for t, tlist in enumerate(cascade[1:]):
                # number of nodes activated in the last time_window time steps
                back_limit = reduce(lambda x, y: x + y, backward_window)
                t = t + 1
                # iterate for every time step of this cascade
                _ = [append(u, v, t-active_nodes[u])
                     for u in active_nodes_list[-back_limit:] for v in tlist]
                # add the nodes to the set of active_nodes
                active_nodes.update({v: t for v in tlist})
                active_nodes_list.extend(tlist)

                backward_window.pop()
                backward_window.appendleft(time_aware*(len(tlist)))

        # for each pair <x,y> compute the norm of the corresponding list
        def compute_norm(d): return {v: norm(l) for v, l in d.items()}
        return {u: compute_norm(udict) for u, udict in coordinates_dict.items()}

    def get_graph(self, coordinates_dict, normalize=False):
        """ Create a DGL graph from the coordinates
        dictionary.

        Each entry of the dictionary corresponds to
        an edge of the graph

        If normalize is true then weights are normalized:
        for any node the sum of its incoming edges sum up to 1

        Parameters
        ----------
        coordinates_dict : dict of dict
          each entry <u,v> denotes the weight of the corresponding edge

        normalize : bool, default True
          apply edge weight normalization

        Returns
        -------
        graph: dgl.Graph
          the dgl graph
        """
        src, dst, weights = edges = [], [], []

        def batch_append(u, v, w):
            edges[0].append(u)
            edges[1].append(v)
            edges[2].append(w)

        _ = [batch_append(u, v, w) for u in coordinates_dict for v,
             w in coordinates_dict[u].items()]
        # create the graph
        graph = dgl.graph((src, dst))
        # add the edge weights
        graph.edata['w'] = th.tensor(weights, dtype=th.float)

        if normalize:
            in_degrees = graph.in_degrees()

            def normalize_weights(v):
                v_in_edges = graph.in_edges(v, "eid")
                graph.edata['w'][v_in_edges] = graph.edata['w'][v_in_edges]\
                  .sum()/in_degrees[v]

            _ = [normalize_weights(v.item()) for v in graph.nodes()]

        return graph

    def get_target_graph(self):
        """ Create the target graph by combining the
        enc_graph and dec_graph.

        More specifically, for every (u,v) in self.enc_graph,

          w_{uv} = w_{uv}^{dec_graph}  if (u,v) in self.dec_graph
          w_{uv} = 0 if (u,v) not in self.dec_graph

        Returns
        -------
        neg_graph : dgl.Graph
        """
        src, dst, weights = edges = [], [], []

        def batch_append(u, v, w):
            edges[0].append(u)
            edges[1].append(v)
            edges[2].append(w)

        src_tensor, dst_tensor = self.enc_graph.edges()
        neg_edge_cnt = 0
        for u, v in zip(src_tensor, dst_tensor):
            # check if we need to add more edges
            try:
                eid = self.dec_graph.edge_ids(u, v)
                w = self.dec_graph.edata['w'][eid].item()
            except dgl.DGLError:
                w = 0
                neg_edge_cnt += 1

            batch_append(u, v, w)

        target_graph = dgl.graph((src, dst))
        target_graph.edata['w'] = th.tensor(weights, dtype=th.float)
        print(neg_edge_cnt)
        return target_graph
