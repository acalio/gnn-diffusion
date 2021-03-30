import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl


def load_cascades(path):
    """Function for loading the cascades

    Parameters
    ----------
    path : str
      path to the file containing the cascades

    Returns
    -------
    list of cascades
       each cascade is as list of list of int

    Example
    -------
    Here is an example of a single cascade
    [[0,1], [2,3,4]]
    This function returns a list of the above lists
    [
      [[0, 1], [2,3,4,5]], # cascade 1
      [[2,3,4,5], [10, 11]], # cascade 2
      ...
      ...
      ...
    ]
    """
    cascades = []
    with open(path) as f:
        for line in f:
            time, active_list_str = line.split(':')
            if time == '0':
                #  a new cascade is starting
                cascades.append([])
            # transform the active_list string into
            # a list of int
            active_list = [x for x in map(int, active_list_str.split())]
            cascades[-1].append(active_list)
    return cascades


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def construct_negative_graph(graph, k):
    src, dst = graph.edges()
    # each source node is replicated k times
    neg_src = src.repeat_interleave(k)
    # drak k random samples for each edge
    neg_dst = th.randint(0, graph.num_nodes(), (len(src)*k))

    # neg_src and neg_dst contain the edges of the negative graph
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())
