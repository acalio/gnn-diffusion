import click
import networkit as nk
from cascade_generator.diffusion import IndependendCascade
from cascade_generator.generator import CascadeGenerator
from numpy.random import randint

@click.command()
@click.argument('graph', type=click.Path(exists=True))
@click.argument('model', type=click.Choice(['ic', 'lt']))
@click.option('--runs', '-r', type=int, default=1)
@click.option('--size', '-s', type=int, default=1)
@click.option('--size-range', nargs=2, type=click.Tuple([int, int]), default=(None, None))
@click.option('--output', '-o', type=click.Path(), default=None)
@click.option('--mode', '-m', type=click.Choice(["append", "new"]),
              default='append')
@click.option('--strategy', '-s', type=click.Choice(['uniform', 'degree', 'random-walk']),
              default='uniform')
@click.option('--beta', '-b', type=float, default=1.0)
@click.option('--separator', type=str, default=' ')
@click.option('--first-node', type=int, default=0)
@click.option('--comment-prefix', type=str, default='#')
def main(graph,
         model,
         runs,
         size,
         size_range,
         output,
         mode,
         strategy,
         beta,
         separator,
         first_node,
         comment_prefix):
    # initialize the graph reader
    reader = nk.graphio.EdgeListReader(
        separator, first_node, comment_prefix, True, True)
    # read the graph
    g = reader.read(graph)
    # print some info on the graph
    nk.overview(g)

    # initialize the diffusion model
    if model == "ic":
        dmodel = IndependendCascade()
    else:
        raise ValueError("LT not available")

    if size_range[0] is not None:
        # select the initial seed sets size uniformly at random
        # between size_range[0] and size_range[1]
        size = randint(size_range[0], size_range[1], runs).tolist()

    cascade_generator = CascadeGenerator(g, dmodel)
    # get the cascades
    print("="*50 + "\nStart cascade generation with the following params")
    print(f"\truns:{runs}\n\tstrategy:{strategy}")
    cascades = cascade_generator(size, runs,
                                 seed_selection_strategy=strategy, beta=beta)

    if output is None:
        # print on the screen
        print(cascades)
        return

    # save the cascades to a file
    # each cascade is saved according to the following format
    # <time-step>: v1 v2 v3 ... white space separated list of nodes
    m = 'a+' if mode == 'append' else 'w+'
    with open(output, m) as f:
        for cascade in cascades:
            for t, active_list in enumerate(cascade):
                # convert the list of active nodes into a string
                active_string = " ".join(map(str, active_list))
                f.write(f"{t}:{active_string}\n")


if __name__ == '__main__':
    main()
