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
from dgl_diffusion.util import get_optimizer, get_loss, get_architecture, evaluate, MetricLogger, dgl_to_nx, train_val_test_split_, maybe_to_cpu
from dgl_diffusion.persistance import PManager
from numpy.random import permutation, choice
from collections import OrderedDict
import dgl
import networkx as nx
import itertools


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
@click.option("--loss", '-l', type=click.Choice(["mse", "mae", "huber", "lgcos", "kl"]), default="mse")
@click.option("--device", type=str, default="cpu")
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
@click.option("--force/--no-force", type=bool, default=False)
@click.option("--normalize-weights/--no-normalize-weights", default=True)
@click.option("--evaluation-metric", type=click.Choice(["mse", "mae"]), default="mse")
@click.option("--negative-positive-ratio", type=float, default=None)
@click.option("--loss-reduction", type=click.Choice(["sum", "mean"]), default="mean")
@click.option("--evaluation-metric-reduction", type=click.Choice(["sum", "mean"]), default="mean")
@click.option("--clip-at", type=float, default=1.0)
@click.option("--sampler-fanout", cls=ListParser, default="[]",)
@click.option("--batch-size", type=int, default=1024)
def main(netpath, cascade_path, epochs,
         optimizer, learning_rate, loss,
         device, encoder_units, encoder_agg_act,
         encoder_out_act, encoder_weight, decoder_units,
         decoder_act, cascade_strategy, cascade_time_window,
         test_size, validation_size, validation_interval,
         max_cascade, cascade_randomness, save_cascade,
         training_log_interval, data_repo, results_repo,
         force, normalize_weights, evaluation_metric,
         negative_positive_ratio,
         loss_reduction,
         evaluation_metric_reduction,
         clip_at):

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
    # set the training/validation/test set segments size
    builder.training_size = 1.0 - validation_size - test_size
    builder.validation_size = validation_size
    builder.test_size = test_size
    load_kws = dict()
    data_pm, already_in_repo = None, False
    if data_repo:
        # check if the enC_graph is already
        # available in the data repository
        data_pm = PManager(data_repo, force)
        # create the hash
        data_pm.hash(("infgraph", os.path.basename(netpath)),
                     ("cascade", os.path.basename(cascade_path)),
                     ("cascade_strategy", cascade_strategy),
                     ("cascade_time_window", cascade_time_window),
                     ("normalize_weights", normalize_weights),
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

    # split into training, validation and test set
    # check if the split sizes are correct
    if sum((validation_size, test_size)) > 1:
        print("\033[91mThe size of training, validation"
              "and test size mustbe at most 1\033[0m'")
        sys.exit(-1)


    neg_training_ids, neg_validation_ids, neg_test_ids = train_val_test_split_(negative_graph.number_of_edges(),
                                                                               (1-validation_size-test_size, validation_size, test_size))

    pos_training_ids, pos_validation_ids, pos_test_ids = train_val_test_split_(data.dec_graph.number_of_edges(),
                                                                               (1-validation_size-test_size, validation_size, test_size))
    # prepare the labels
    pos_labels = data.dec_graph.edata['w']
    neg_labels = negative_graph.edata['w']

    # preare the logger
    training_logger = MetricLogger(['iter', 'time', loss, evaluation_metric,
                                    f'pos-{evaluation_metric}', f'neg-{evaluation_metric}'],
                                   ['%d', '%d', '%.4f', '%.4f', '%.4f', '%.4f'])

    validation_logger = MetricLogger(['iter',  evaluation_metric, f'pos-{evaluation_metric}', f'neg-{evaluation_metric}'],
                                     ['%d', '%.4f', '%.4f', '%.4f'])
    test_logger = MetricLogger(['iter', evaluation_metric, f'pos-{evaluation_metric}', f'neg-{evaluation_metric}'],
                               ['%d', '%.4f', '%.4f', '%.4f'])

    # create training/validation/test labels merging the positive and negative graphs
    training_labels = th.cat([pos_labels[pos_training_ids],
                              neg_labels[neg_training_ids]], 0)
    validation_labels = th.cat([pos_labels[pos_validation_ids],
                                neg_labels[neg_validation_ids]], 0)
    test_labels = th.cat([pos_labels[pos_test_ids],
                          neg_labels[neg_test_ids]], 0)

    # embeddings
    embeddings = nn.Embedding(
        data.enc_graph.number_of_nodes(), encoder_units[0])

    feat = embeddings.weight
    nn.init.xavier_uniform_(feat)
    # Create the encoder
    encoder = InfluenceEncoder(
        encoder_units, encoder_agg_act, encoder_out_act, device)

    # create the decoder
    decoder = InfluenceDecoder(get_architecture(
        decoder_units, decoder_act), device)

    # create the encoder decoder model
    net = InfEncDec(encoder, decoder).to(th.device(device))

    # define the loss function
    loss_fn = get_loss(loss, loss_reduction)

    # get the evauation metric
    evaluation_fn = get_loss(evaluation_metric, evaluation_metric_reduction)
    
    # initialize the optimizer
    opt = get_optimizer(optimizer)(itertools.chain(
        net.parameters(), embeddings.parameters()), lr=learning_rate)

    with tqdm.trange(epochs) as pbar:
        postfix_dict = OrderedDict({'loss': 'nan', f'val-{evaluation_metric}': 'nan'})
        for epoch in pbar:
            net.train()
            pos_predictions, _ = net(data.enc_graph, feat, data.dec_graph)  # , negative_graph)

            # concatenate the prediction
#            predictions = th.cat([pos_predictions[pos_training_ids],
#                                  neg_predictions[neg_training_ids]], 0)
            predictions = pos_predictions[pos_training_ids]
            # total scores
            loss_value = loss_fn(predictions, training_labels)
            opt.zero_grad()
            loss_value.backward()
#            nn.utils.clip_grad_norm_(itertools.chain(
#                net.parameters(), embeddings.parameters()), clip_at)
            opt.step()
            postfix_dict['loss'] = '%.03f' % loss_value.item()
            pbar.set_postfix(postfix_dict)
            if epoch % training_log_interval == 0:
                # compute the evaluation metric
                pos_metric_value = evaluate(net,
                                            data.enc_graph,
                                            feat,
                                            data.dec_graph,
                                            pos_labels[pos_training_ids],
                                            pos_training_ids,
                                            evaluation_fn)

                neg_metric_value = evaluate(net,
                                            data.enc_graph,
                                            feat,
                                            negative_graph,
                                            neg_labels[neg_training_ids],
                                            neg_training_ids,
                                            evaluation_fn)

                training_logger.log(**{
                    'iter': epoch,
                    'time': (epoch_start - time.time()),
                    loss: loss_value.item(),
                    evaluation_metric: .5 * (pos_metric_value+neg_metric_value),
                    f'pos-{evaluation_metric}': pos_metric_value,
                    f'neg-{evaluation_metric}': neg_metric_value
                })

                if switched:
                    net.decoder.to(th.device('cuda'))
                    net.decoder.device = th.device('cuda')

            if epoch % validation_interval == 0:
                switched = maybe_to_cpu(net.decoder)
                pos_metric_value = evaluate(net.decoder, data.dec_graph, feat,
                                            pos_labels[pos_validation_ids], pos_validation_ids, evaluation_fn)
                
                neg_metric_value = evaluate(net.decoder, negative_graph, feat,
                                            neg_labels[neg_validation_ids], neg_validation_ids, evaluation_fn)
                        
                metric_value = .5*(pos_metric_value+neg_metric_value)
                validation_logger.log(**{
                    'iter': epoch,
                    evaluation_metric: metric_value,
                    f'pos-{evaluation_metric}': pos_metric_value,
                    f'neg-{evaluation_metric}': neg_metric_value
                })

                postfix_dict[f'val-{evaluation_metric}'] = '%.03f' % metric_value
                pbar.set_postfix(postfix_dict)

    net.eval()
    # compute the performance on the test set
    pos_test_metric = evaluate(net,
                               data.enc_graph,
                               feat,
                               data.dec_graph,
                               pos_labels[pos_test_ids],
                               pos_test_ids,
                               evaluation_fn)
    neg_test_metric = evaluate(net,
                               data.enc_graph,
                               feat,
                               negative_graph,
                               neg_labels[neg_test_ids],
                               neg_test_ids,
                               evaluation_fn)

    print(f"Test Set {evaluation_metric}:{.5*(pos_test_metric+neg_test_metric)}")
    print(f"\t(Positive) Test Set {evaluation_metric}:{pos_test_metric}")
    print(f"\t (Negative) Test Set {evaluation_metric}:{neg_test_metric}")
    print(feat[0])
    # compute some prediction on training set over the entire graph
    with th.no_grad():
        pos_src, pos_dst, pos_eid = data.dec_graph.edges(form="all")
        neg_src, neg_dst, neg_eid = negative_graph.edges(form="all")
        pos_pred = net(data.enc_graph, feat, data.dec_graph)[0].squeeze()
        # print(pos_pred[:30])
        # print(data.dec_graph.edata['w'][:30])
        print(f"{evaluation_metric}:{evaluation_fn(pos_pred, data.dec_graph.edata['w'])}")
#        neg_pred = net.decoder(negative_graph, feat).squeeze()
#        print(neg_pred[:30], neg_pred.min())
#        print(f"Accuracy:{evaluation_fn(neg_pred, negative_graph.edata['w'])}")
        for _ in range(50):
            i = choice(pos_eid, 1).item()

            print(pos_src[i].item(), pos_dst[i].item(),
                  data.dec_graph.edata['w'][i].item(), pos_pred[i])
#            print(neg_src[i].item(), neg_dst[i].item(), negative_graph.edata['w'][i].item())

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
                ("encoder_out_act", encoder_out_act),
                ("decoder_units", decoder_units),
                ("decoder_act", decoder_act),
                ("cascade_strategy",  cascade_strategy),
                ("cascade_time_window", cascade_time_window),
                ("normalize_weights", normalize_weights),
                ("max_cascade", max_cascade),
                ("cascade_randomness", cascade_randomness))

        training_loss_logger_df = training_logger.close()
        validation_loss_logger_df = validation_logger.close()

        pm.persist(
            # ("negative_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(negative_graph), f), "wb"),
            ("enc_graph.edgelist", lambda f: nx.write_weighted_edgelist(
                dgl_to_nx(data.enc_graph), f), "wb"),
            ("train_logger.csv", lambda f: training_loss_logger_df.to_csv(f, index=None), "wb"),
            ("validation_logger.csv", lambda f: validation_loss_logger_df.to_csv(f, index=None), "wb"))
        pm.close()

    # save the encoded graph
    if data_pm:
        data_pm.persist(
            # ("target_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(target_graph), f), "wb"),
            ("enc_graph.edgelist", lambda f: nx.write_weighted_edgelist(dgl_to_nx(data.enc_graph), f), "wb"))
        data_pm.close()


if __name__ == '__main__':
    main()
