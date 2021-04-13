from os.path import splitext, basename
from dgl_diffusion.util import load_cascades, nx_to_dgl, edges_to_dgl
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

    def __init__(self):
        self._enc_graph = None
        self._dec_graph = None

    @property
    def enc_graph(self):
        return self._enc_graph

    @enc_graph.setter
    def enc_graph(self, enc_graph):
        self._enc_graph = enc_graph

    @property
    def dec_graph(self):
        return self._dec_graph

    @dec_graph.setter
    def inf_graph(self, dec_graph):
        self._dec_graph = dec_graph

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
        print(edges)
        # create the graph
        graph = edges_to_dgl(src, dst, weights)

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

        return edges_to_dgl(src, dst, weights)

class CascadeDatasetBuilder:
    """Class responsible for 
    building the dataset. 
    There are two main options:
    
    i) Create the dataset by providing 
       the influence graph and the cascade file

    ii) Create the dataset by providing two 
       edgelist files: 
       - the influence graph, i.e., the dec_graph
       - the cascade graph, i.e., the enc_graph

    """
    def __init__(self):
        # init everything to None
        self._graph_path = None
        self._enc_graph_path = None
        self._cascade_path = None
        self._strategy = None
        self._max_cascade = -1
        self._cascade_randomness = False
        self._save_cascade = None
        self._edge_weights_normalization = False

    @property
    def graph_path(self):
        return self._graph_path

    @graph_path.setter
    def graph_path(self, graph_path):
        self._graph_path = graph_path

    @property
    def enc_graph_path(self):
        return self._enc_graph_path

    @enc_graph_path.setter
    def enc_graph_path(self, enc_graph_path):
        self._enc_graph_path = enc_graph_path

    @property
    def cascade_path(self):
        return self._cascade_path

    @cascade_path.setter
    def cascade_path(self, cascade_path):
        self._cascade_path = cascade_path

    @property
    def strategy(self):
        return self._strategy
        
    @strategy.setter
    def strategy(self, strategy):
        if strategy not in ("counting", "tempdiff"):
            raise ValueError(f"Strategy {strategy} unknown")
        self._strategy = strategy

    @property
    def max_cascade(self):
        return self._max_cascade

    @max_cascade.setter
    def max_cascade(self, max_cascade):
        self._max_cascade = max_cascade

    @property
    def cascade_randomness(self):
        return self._cascade_randomness

    @cascade_randomness.setter
    def cascade_randomness(self, cascade_randomness):
        self.cascade_randonness = cascade_randomness

    @property
    def save_cascade(self):
        return self._save_cascade

    @save_cascade.setter
    def save_cascade(self, save_cascade):
        self._save_cascade = save_cascade

    @property
    def edge_weights_normalization(self):
        return self._edge_weights_normalization

    @edge_weights_normalization.setter
    def edge_weights_normalization(self, edge_normalization):
        self._edge_weights_normalization = edge_normalization

    def build(self, **kwargs) -> CascadeDataset:
        d = CascadeDataset()

        # load/create the encoded graph            
        if self._cascade_path:
            _, cascade_format = splitext(self._cascade_path)
            if cascade_format == ".pickle":
                # load from pickle
                with open(self._cascade_path, 'rb') as f:
                    cascades = pickle.load(f)

            else:
                strategy_fn = {
                    'counting': d.counting_weight,
                    'tempdiff': d.tempdiff_weight
                }[self._strategy]

                cascades = load_cascades(self.cascade_path,
                                         max_cascade=self._max_cascade,
                                         randomness=self._cascade_randomness)

                coordinates_dict = strategy_fn(cascades, **kwargs)
                enc_graph = d.get_graph(coordinates_dict, self._edge_weights_normalization)
                if self._save_cascade:
                    # save the cascade
                    with open(self._save_cascade) as f:
                        pickle.dump(cascades, f)

        elif self._enc_graph_path:
            # the encoded graph is provided in edgelist format
            nx_enc_graph = nx.read_weighted_edgelist(self._enc_graph_path,
                                                     create_using=nx.DiGraph(), 
                                                     nodetype=int)
            
            # enc_graph = dgl.from_networkx(nx_enc_graph, edge_attrs=["weight"])
            # enc_graph.edata['w'] = enc_graph.edata['weight']
            
            # del enc_graph.edata['weight']
            enc_graph = nx_to_dgl(nx_enc_graph)

        else:
            raise ValueError("You must specifiy the encoded graph")

        # load the influence graph
        nx_inf_graph = nx.read_weighted_edgelist(self._graph_path,
                                                 create_using=nx.DiGraph(),
                                                 nodetype=int)
        
        inf_graph = nx_to_dgl(nx_inf_graph)
        # inf_graph = dgl.from_networkx(nx_inf_graph, edge_attrs=["weight"])
        # inf_graph.edata['w'] = inf_graph.edata['weight']
        # del inf_graph.edata['weight']

        d.enc_graph = enc_graph
        d.inf_graph = inf_graph

        return d
        
            
            

            

            
                

                

            


