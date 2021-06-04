"""NN modules"""
import torch as th
import torch.nn as nn
import dgl.function as fn
from dgl_diffusion.util import get_activation


class InfluenceGraphConv(nn.Module):
    """Graph convolution module

    Parameters
    ----------
    in_features : int
      input feature size

    out_features : int
      output feature size

    weight : bool, optional
      If True, apply a linear layer before message passing.
      Otherwise use the weight matrix provided by the caller
    Attributes
    ----------
    """
    def __init__(self,
                 in_features,
                 out_features,
                 dropout=0,
                 weight=True,
                 device=None):
        super(InfluenceGraphConv, self).__init__()
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_features, out_features))
        else:
            self.register_parameter('weight', None)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.reset_parameters()

    
    def reset_parameters(self):
        """Initialize learnable parameters"""
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        """Compute the graph convolution

        Parameters
        ----------
        graph : dgl.Graph
          input graph
        feat : torch.Tensor
          input features
        weight : torch.Tensor of shape (in_feature, out_features), optional
          external weight tensor

        Returns
        -------
        rst : torch.Tensor
          output features
        """
        if weight is None:
            weight = self.weight

        with graph.local_scope():
            if isinstance(feat, tuple):
                # dst features are discarded 
                feat, _ = feat[0] @ weight
                feat_src = feat_dst = feat
            elif graph.is_block:
                # mini-batch training
                feat_src = feat @ weight
                feat_dst = feat[:graph.number_of_dst_nodes()] @ weight
            else:
                # full graph training
                feat = feat @ weight
                feat_src = feat_dst = feat


            cu = graph.srcdata['cu'].to(self.device)
            cv = graph.dstdata['cv'].to(self.device)
            
            # apply normalization plus node-wise dropout
            graph.srcdata['h'] = feat_src * self.dropout(cu)
            graph.dstdata['h'] = feat_dst 

            graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                             fn.sum(msg='m', out='h'))

            
            rst = graph.dstdata['h']
            # apply normalization
            rst = rst * cv
            return rst


class InfluenceEncoder(nn.Module):
    """NN
    Attributes
    ----------
    conv_layers : nn.ModuleList
      list of convoulutional layers

    out_layer : nn.Module
      output layer 

    agg_act : string or callable
      activation function to  apply
      on the node embeddings returned
      by the convolution
    
    out_act : string or callable
      activation function to apply
      on the output layer

    Parameters
    ----------
    units : list of int
      number of units for each convolutional layer and
      the output layer. More specifically, each pair 
      at index (i, i+1) represents the input and output 
      units of a convolutional layer, with the only
      exception of the last pair (len(units)-2, len(units)-1)
      which denotes the shape of the output layer

    agg_act : callable, str, optional
      activation function for the output of the convolution

    out_act : callable, str, optional
      activation function of the output layer
    
    dropout : float, optional
      dropout probability for the convolutional layers

    device : str, optional
    """
    def __init__(self, units, agg_act, out_act, dropout=0, device='cpu'):
        super(InfluenceEncoder, self).__init__()
        dimensions = [(e, units[i+1]) for i, e in enumerate(units[:-1])]
        self.conv_layers = nn.ModuleList([
            InfluenceGraphConv(*d, dropout=dropout) for d in dimensions[:-1]
        ])

        self.out_layer = nn.Linear(*dimensions[-1])

        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)

        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, feat=None):
        """Forward function

        Parameters
        ----------
        graph : dgl.Graph or a list of graphs/blocks
          the graph
        feat : torch.Tensor
          node features, if None use an identity matrix

        Returns
        -------
        torch.Tensor
          new node features
        """
        
        graphs = graph if isinstance(graph, list) else [graph]*len(self.conv_layers)

        # apply the convolution
        for i, (layer, g) in enumerate(zip(self.conv_layers, graphs)):
            feat = self.agg_act(layer(g, feat))
        # apply the output layer and return
        return self.out_act(self.out_layer(feat))



class InfluenceDecoder(nn.Module):
    """Influence Decoder

    Given a graph G, for each edge (i,j) in G
    we compute the influence weight as:

      w_{ij} = MLP(concat(i_embedding, j_embedding))

    Parameters
    ----------
    sequential_model : nn.Module
      feed forward neural network

    Attributes
    ----------
    decoder  : nn.Module
      feed forward neural network

    """

    def __init__(self, seq_dict, device='cpu'):
        super(InfluenceDecoder, self).__init__()
        self.decoder = nn.Sequential(seq_dict)

    def forward(self, graph, feat):
        """Forward function

        Parameters
        ----------
        graph : dgl.Graph
          the graph

        feat : th.Tensor
          node embeddings. Shape (|V|, D)

        Returns
        -------
        th.Tensor
          edge influence scores
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            graph.apply_edges(self.concat_message_fn)
            edges = graph.edata['cat_h']
            out = self.decoder(edges)
        return out

    def concat_message_fn(self, edges):
        return {'cat_h': th.cat([edges.src['h'],
                                 edges.dst['h']], 1)}

class InfEncDec(nn.Module):
    def __init__(self, encoder, decoder,  device="cpu"):
        super(InfEncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graph, feat, inf_graph, neg_graph=None):
        h = self.encoder(graph, feat)
        return self.decoder(inf_graph, h), self.decoder(neg_graph, h) \
            if neg_graph else self.decoder(inf_graph, h),
        



# """NN modules"""
# import torch as th
# import torch.nn as nn
# import dgl.function as fn
# from dgl_diffusion.util import get_activation


# class InfluenceGraphConv(nn.Module):
#     """Graph convolution module

#     Parameters
#     ----------
#     in_features : int
#       input feature size

#     out_features : int
#       output feature size

#     weight : bool, optional
#       If True, apply a linear layer before message passing.
#       Otherwise use the weight matrix provided by the caller
#     Attributes
#     ----------
#     """

#     def __init__(self,
#                  in_features,
#                  out_features,
#                  weight=True,
#                  device=None):
#         super(InfluenceGraphConv, self).__init__()
#         self._in_feats = in_features
#         self._out_feats = out_features
#         self.device = device

#         if weight:
#             self.weight = nn.Parameter(th.Tensor(in_features, out_features))
#         else:
#             self.register_parameter('weight', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         """Initialize learnable parameters"""
#         if self.weight is not None:
#             nn.init.xavier_uniform_(self.weight)

#     def forward(self, graph, feat, weight=None):
#         """Compute the graph convolution

#         Parameters
#         ----------
#         graph : dgl.Graph
#           input graph
#         feat : torch.Tensor
#           input features
#         weight : torch.Tensor of shape (in_feature, out_features), optional
#           external weight tensor

#         Returns
#         -------
#         rst : torch.Tensor
#           output features
#         """
#         with graph.local_scope():
#             if isinstance(feat, tuple):
#                 feat, _ = feat  # dst features are discarded

#             if weight is None:
#                 weight = self.weight

#             feat = feat @ weight

#             graph.srcdata['h'] = feat
#             graph.update_all(fn.u_mul_e('h', 'w', 'm'),
#                              fn.sum(msg='m', out='h'))
#             rst = graph.dstdata['h']


#             return rst


# class InfluenceEncoder(nn.Module):
#     """NN
#     Attributes
#     ----------
#     Parameters
#     ----------
#     in_units : int
#       size of input feature

#     hid_units : int
#       size of hidden feature. The representation
#       provided by the convolution

#     out_units : int
#       size of output feature
#     agg_act : callable, str, optional
#       activation function for the output of the convolution

#     out_act : callable, str, optional
#       activation function of the output layer

#     device : str, optional
#     """

#     def __init__(self, in_units, hid_units, out_units, agg_act, out_act, device='cpu'):
#         super(InfluenceEncoder, self).__init__()

#         self.conv_layer = InfluenceGraphConv(in_units, hid_units)
#         self.conv_layer_ = InfluenceGraphConv(hid_units, hid_units)

#         self.out_layer = nn.Linear(hid_units, out_units)

#         self.agg_act = get_activation(agg_act)
#         self.out_act = get_activation(out_act)

#         self.device = device
#         self.reset_parameters()

#     def reset_parameters(self):
#         """Initialize the model parameters"""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, graph, feat=None):
#         """Forward function

#         Parameters
#         ----------
#         graph : dgl.Graph
#           the graph
#         feat : torch.Tensor, optional
#           node features, if None use an identity matrix

#         Returns
#         -------
#         torch.Tensor
#           new node features
#         """
#         # apply the convolution
#         feat = self.conv_layer(graph, feat)  # nodes x embedding size
#         # apply the non-linearity
#         # feat = self.agg_act(feat)

#         # feat = self.conv_layer_(graph, feat)
#         feat = self.agg_act(feat)
#         # apply the output layer and return
#         return self.out_act(self.out_layer(feat))


# class InfluenceDecoder(nn.Module):
#     """Influence Decoder

#     Given a graph G, for each edge (i,j) in G
#     we compute the influence weight as:

#       w_{ij} = MLP(concat(i_embedding, j_embedding))

#     Parameters
#     ----------
#     sequential_model : nn.Module
#       feed forward neural network

#     Attributes
#     ----------
#     decoder  : nn.Module
#       feed forward neural network

#     """

#     def __init__(self, seq_dict):
#         super(InfluenceDecoder, self).__init__()
#         self.decoder = nn.Sequential(seq_dict)
#         self.reset_parameters()

#     def forward(self, graph, feat):
#         """Forward function

#         Parameters
#         ----------
#         graph : dgl.Graph
#           the graph

#         feat : th.Tensor
#           node embeddings. Shape (|V|, D)

#         Returns
#         -------
#         th.Tensor
#           edge influence scores
#         """
#         with graph.local_scope():
#             graph.ndata['h'] = feat
#             graph.apply_edges(self.concat_message_fn)
#             edges = graph.edata['cat_h']
#             out = self.decoder(edges)
#         return out

#     def concat_message_fn(self, edges):
#         return {'cat_h': th.cat([edges.src['h'],
#                                  edges.dst['h']], 1)}

#     def reset_parameters(self):
#         """Initialize the model parameters"""
#         for p in self.decoder.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#             else:
#                 nn.init.uniform_(p, a=0.0, b=1.0)



# class InfEncDec(nn.Module):
#     def __init__(self, encoder, decoder,  device="cpu"):
#         super(InfEncDec, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, graph, feat, inf_graph, neg_graph=None):
#         h = self.encoder(graph, feat)
#         return self.decoder(inf_graph, h), self.decoder(neg_graph, h) \
#             if neg_graph else self.decoder(inf_graph, h),
