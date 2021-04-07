from collections import OrderedDict
import torch as th
import torch.nn as nn
import torch.optim as optim
from numpy.random import rand
import dgl


def load_cascades(path, max_cascade = None, randomness=False):
    """Function for loading the cascades.
    The file is always read line by line.
    If max_cascade is set, then only the first max_cascade
    cascades are read.
    If randomness is set, once the a cascades is read from the file,
    it has .5 chance to be loaded into the final result
    

    Parameters
    ----------
    path : str
      path to the file containing the cascades

    max_cascade : int, default -1
      number of cascades to load from the file.
      -1 means all the cascades are loaded

    randomness : bool, defualt False
      if True each cascade has 0.5 chance to be loaded

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
            # decide if this cascade has to be ignored
            if randomness and rand() <= .5:
                del cascades[-1]
            # check if we have reached the max no. of cascades
            if max_cascade and len(cascades) == max_cascade:
                break

    return cascades


def get_activation(act, **kwargs):
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
        try:
            act_fn = {
                "leaky": nn.LeakyReLU,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "softsign": nn.Softsign,
                "softmax": nn.Softmax
            }[act](**kwargs)
            return act_fn
        except KeyError:
            raise NotImplementedError
    else:
        return act


def get_loss(loss):
    """Get the loss function

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if isinstance(loss, str):
        try:
            loss_fn = {
                "mse": nn.MSELoss,
                "mae": nn.L1Loss
            }[loss]()
            return loss_fn
        except KeyError:
            raise NotImplementedError
    else:
        return loss


def get_optimizer(opt):
    """Get the opimizer

    Parameters
    ----------
    opt : string
      name of the optimizer

    Returns
    -------
    pytorch optimizer
    """
    try:
        optimizer = {
            "sgd": optim.SGD,
            "adam": optim.Adam
        }[opt]
        return optimizer
    except KeyError:
        raise NotImplementedError


def construct_negative_graph(graph, k):
    """Construct a negative graph

    It adds k negative edges for each
    node in the graph

    Parameters
    ----------
    graph : dgl graph
      the graph

    k : int
      number of negative edges to add for each node

    Returns
    ------
    dgl graph
      graph with negative edges
    """
    src, dst = graph.edges()
    # each source node is replicated k times
    neg_src = src.repeat_interleave(k)
    # drak k random samples for each edge
    neg_dst = th.randint(0, graph.num_nodes(), (len(src)*k))

    # neg_src and neg_dst contain the edges of the negative graph
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def get_architecture(units, activations):
    """Get the layers sizes and
    the activation function and return
    an ordered dict representing a feed forward
    neural network

    Parameters
    ----------
    units : list of int
      number of units for each layer

    activations: list of string or callable
      activation function for each layer

    Returns
    -------
     orderd dict
       ordered dict for initializing a feed forward NN
    """
    dimensions = [(e, units[i+1])
                  for i, e in enumerate(units[:-1])]
    layers = [("linear_%i" % i, nn.Linear(*t))
              for i, t in enumerate(dimensions)]
    # create the activations
    activations = [("%s_%i" % (act, i), get_activation(act))
                   for i, act in enumerate(activations)]
    seq_dict = OrderedDict()
    layer_turn = True
    while layers or activations:
        try:
            if layer_turn:
                name, layer = layers.pop(0)
            else:
                name, layer = activations.pop(0)
            seq_dict[name] = layer
            layer_turn = not layer_turn
        except IndexError:
            print("\033[91mIncompatible number of layers "
                  "and activation functions\033[0m'")

    return seq_dict



def evaluate(model, metric, graph, features, labels, mask):
    """Evaluate the performance of the model

    Parameters
    ----------
    model : th.Module
      model to be evaluated

    metric : th.Module
      loss function

    graph : dgl graph
      the graph

    features : th.Tensor (N,d)
      nodes features

    labels : th.Tensor (N,)
      ground truth - edge wise

    mask : th.Tensor (E,)
      boolean mask 

    Returns
    -------
    float
      value of the evaluation metric
    """
    model.eval()
    with th.no_grad():
        # comute predictionsl
        pred = model(graph, features)
        # apply the mask
        pred = pred[mask]
        # compute the loss function
        loss_value = metric(pred.squeeze(), labels)
        return loss_value.item()

