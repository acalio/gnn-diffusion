"""
Training the model
"""
import time
import os
import sys
import click
import tqdm
import ast
from dgl_diffusion.data import CascadeDatasetBuilder
from dgl_diffusion.model import InfluenceDecoder, InfluenceEncoder, InfEncDec
import torch as th
import torch.nn as nn
from dgl_diffusion.util import get_optimizer, get_loss, get_architecture, construct_negative_graph, evaluate, MetricLogger, dgl_to_nx
from dgl_diffusion.persistance import PManager
from numpy.random import permutation
from collections import OrderedDict
import dgl
import networkx as nx

class ListParser(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

@click.command()
@click.argument('netpath', type=click.Path(exists=True))
@click.argument('cascade-path', type=click.Path(exists=True), default=None)
@click.option('--epochs', '-e', type=int, default=100)
@click.option('--optimizer', '-o', type=click.Choice(["adam", "sgd"]), default="adam")
@click.option('--learning-rate', '-lr', type=float, default=0.001)
@click.option("--loss", '-l', type=click.Choice(["mse", "mae", "huber", "lgcos"]), default="mse")
@click.option("--device", type=int, default=-1)
@click.option("--encoder-units", cls=ListParser, required=True)
@click.option("--encoder-agg-act", type=str, required=True)
@click.option("--encoder-out-act", type=str, required=True)
@click.option("--encoder-weight/--no-encoder-weight", default=True)
@click.option("--decoder-units", cls=ListParser, required=True)
@click.option("--decoder-act", cls=ListParser, required=True)
@click.option("--cascade-strategy", type=click.Choice(["counting", "tempdiff"]),
              default="counting")
@click.option("--cascade-time-window", type=int, default=0)
@click.option("--test-size", type=float, default=0.2)
@click.option("--validation-size", type=float, default=0.2)
@click.option("--validation-interval", type=int, default=25)
@click.option("--max-cascade", type=int, default=None)
@click.option("--cascade-randomness", type=bool, default=False)
@click.option("--save-cascade", type=click.Path(), default=None)
@click.option("--training-log-interval", type=int, default=3)
@click.option("--data-repo", type=click.Path(), default=None)
@click.option("--results-repo", type=click.Path(), default=None)
@click.option("--force", type=bool, default=False)
@click.option("--normalize-weights/--no-normalize-weights", default=False)
@click.option("--evaluation-metric", type=click.Choice(["mse", "mae"]), default="mse")
def main(netpath,
         cascade_path,
         epochs,
         optimizer,
         learning_rate,
         loss,
         device,
         encoder_units,
         encoder_agg_act,
         encoder_out_act,
         encoder_weight,
         decoder_units,
         decoder_act,
         cascade_strategy,
         cascade_time_window,
         test_size,
         validation_size,
         validation_interval,
         max_cascade,
         cascade_randomness,
         save_cascade,
         training_log_interval,
         data_repo,
         results_repo,
         force,
         normalize_weights,
         evaluation_metric):

    # create the encoder
    encoder = InfluenceEncoder(
        *encoder_units, encoder_agg_act, encoder_out_act, device)

    # create the decoder
    decoder = InfluenceDecoder(get_architecture(
        decoder_units, decoder_act))

    # create the encoder decoder model
    net = InfEncDec(encoder, decoder)

    # read the data
    builder = CascadeDatasetBuilder()
    builder.graph_path = netpath
    load_kws = dict()
    data_pm, already_in_repo = None, False
    if data_repo:
        # check if the enc_graph is already
        # available in the data repository
        data_pm = PManager(data_repo, force)
        # create the hash 
        data_pm.hash(("infgraph",os.path.basename(netpath)),
                     ("cascade",os.path.basename(cascade_path)),
                     ("cascade_strategy",cascade_strategy),
                     ("cascade_time_window",cascade_time_window),
                     ("max_cascade", max_cascade))

        # check if the folder for the given
        # parameters already exists
        cascade_folder = os.path.join(data_repo, data_pm.hex())
        if os.path.exists(cascade_folder):
            print("Loading graph from data repository")
            # load the enc_graph
            builder.enc_graph_path = os.path.join(cascade_folder, "enc_graph.edgelist")
            already_in_repo = True

    if not (data_repo and already_in_repo):
        # specify the parameters to read the cascades
        builder.cascade_path = cascade_path
        builder.strategy = cascade_strategy
        builder.max_cascade = max_cascade
        builder.save_cascade = save_cascade
        builder.edge_weights_normalization = normalize_weights
        load_kws['time_window'] = cascade_time_window

    data = builder.build(**load_kws)
    
    # initialize the optimizer
    opt = get_optimizer(optimizer)(net.parameters(), lr=learning_rate)

    # contruct the target graph with negatie links
    target_graph = data.get_target_graph()

    # split into training, validation and test set
    # check if the split sizes are correct
    if sum((validation_size, test_size)) > 1:
        print("\033[91mThe size of training, validation"
              "and test size mustbe at most 1\033[0m'")
        sys.exit(-1)

    edge_permutation = permutation(data.enc_graph.edges("eid"))
    edges = data.enc_graph.number_of_edges()
    training_size = int((1 - validation_size - test_size)*edges)
    validation_size = int(validation_size * edges)

    # create the mask
    training_mask = th.zeros(edge_permutation.shape, dtype=th.bool)
    training_mask[edge_permutation[:training_size]] = True

    validation_mask = th.zeros_like(training_mask, dtype=th.bool)
    validation_mask[edge_permutation[training_size:training_size+validation_size]] = True

    test_mask = th.zeros_like(training_mask, dtype=th.bool)
    test_mask[edge_permutation[training_size+validation_size:]] = True

    # save into the graph
    data.enc_graph.edata['train_mask'] = training_mask
    data.enc_graph.edata['val_mask'] = validation_mask
    data.enc_graph.edata['test_mask'] = test_mask

    # save into the target graph
    training_labels = target_graph.edata['w'][training_mask]
    validation_labels = target_graph.edata['w'][validation_mask]
    test_labels = target_graph.edata['w'][test_mask]
    
    # preare the logger
    training_logger = MetricLogger(['iter', 'time',  loss, evaluation_metric], ['%d','%d', '%.4f', '%.4f'])
    validation_logger = MetricLogger(['iter',  evaluation_metric], ['%d', '%.4f'])
    test_logger = MetricLogger(['iter', evaluation_metric], ['%d', '%.4f'])

    # define the loss function
    loss_fn = get_loss(loss)

    # get the evauation metric
    evaluation_fn = get_loss(evaluation_metric)

    # embeddings
    embeddings = nn.Embedding(
        data.enc_graph.number_of_nodes(), encoder_units[0])
    feat = embeddings.weight
    nn.init.xavier_uniform_(feat)
    with tqdm.trange(epochs) as pbar:
        postfix_dict = OrderedDict({'loss': 'nan', f'val-{evaluation_metric}': 'nan'})
        for epoch in pbar:
            epoch_start = time.time()
            net.train()
            predictions = net(data.enc_graph, feat)
            predictions = predictions[training_mask].squeeze()
            # total scores
            loss_value = loss_fn(predictions.squeeze(), training_labels)
            opt.zero_grad()
            loss_value.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            postfix_dict['loss'] = '%.03f' % loss_value.item()
            pbar.set_postfix(postfix_dict)
            
            if epoch % training_log_interval == 0:
                # compute the evaluation metric
                metric_value = evaluate(net,
                                        evaluation_fn,
                                        data.enc_graph,
                                        feat,
                                        training_labels,
                                        training_mask)
                
                training_logger.log(**{
                    'iter': epoch,
                    'time': (epoch_start - time.time()),
                    loss: loss_value,
                    evaluation_metric: metric_value})
            
            if epoch % validation_interval == 0:
                metric_value = evaluate(net,
                                        evaluation_fn,
                                        data.enc_graph,
                                        feat,
                                        validation_labels,
                                        validation_mask)
                
                validation_logger.log(**{
                    'iter': epoch,
                    evaluation_metric: metric_value})

                postfix_dict[f'val-{evaluation_metric}'] = '%.03f' % metric_value
                pbar.set_postfix(postfix_dict)

    # persist results
    if results_repo:
        pm = PManager(results_repo, force)
        # generate the hash
        pm.hash(("infgraph", os.path.basename(netpath)),
                ("cascade", os.path.basename(cascade_path)), 
                ("epochs", epochs),
                ("loss", loss),
                ("learning_rate", learning_rate),
                ("encoder_units", encoder_units),
                ("encoder_agg_acc", encoder_agg_act),
                ("encoder_out_act",encoder_out_act),
                ("decoder_units", decoder_units),
                ("decoder_act", decoder_act),
                ("cascade_strategy",  cascade_strategy),
                ("cascade_time_window",cascade_time_window),
                ("max_cascade", max_cascade),
                ("cascade_randomness", cascade_randomness))


        training_loss_logger_df = training_logger.close()
        validation_loss_logger_df = validation_logger.close()

        pm.persist(
            ("target_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(target_graph), f), "wb"),
            ("enc_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(data.enc_graph), f), "wb"),
            ("train_logger.csv", lambda f: training_loss_logger_df.to_csv(f, index=None ), "wb"),
            ("validation_logger.csv", lambda f: validation_loss_logger_df.to_csv(f, index=None ), "wb"))
        pm.close()

    # save the encoded graph
    if data_pm:
        data_pm.persist(
            ("target_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(target_graph), f), "wb"),
            ("enc_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(data.enc_graph), f), "wb"))
        data_pm.close()
        
if __name__ == '__main__':
    main()
