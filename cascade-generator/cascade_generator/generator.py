import numpy.random as rn
from networkit.centrality import DegreeCentrality
import tqdm


class CascadeGenerator:
    """Cascade Generator.

    This class is in  charge of running a diffusion
    process and to collect the output of the propagation.

    Parameters
    ----------
    influence_graph: networkit Graph object
      influence graph

    diffusion_model: DiffusionModel
      diffusion model to generate the cascades
    """

    def __init__(self, influence_graph, diffusion_model):
        self.influence_graph = influence_graph
        self.diffusion_model = diffusion_model

    def __call__(self, k, num_of_runs=1, seed_selection_strategy='degree'):
        """ Method to generate the cascades

        Parameters
        ----------
        k: int or list-like of int of shape (num_of_runs,)
            number of initial seeds of the propagation.
            If k is a single integer than each simulation has
            the same number of initial seeds

        num_of_runs: int, default=1
            number of propagation process to run.

        seed_selection: {'degree', 'uniform'}, default='degree'
           strategy to randomly select the seeds of the propagation

           - 'degree' nodes are selected according to their out degree.
             Nodes with high out-degree are more likely to be selected

           - 'uniform' nodes are selected according to a uniform distribution.
             Each node has the same probability 1/N to be selected, where N
             is the number of nodes in the graph

        Return
        ------
        cascades: list of cascades
          a list containing every cascade generated
        """
        if isinstance(k, tuple) or isinstance(k, list):
            runs = k
        else:
            runs = [k] * num_of_runs

        # compute the probability distribution
        if seed_selection_strategy == 'uniform':
            probs = None
        elif seed_selection_strategy == 'degree':
            probs = DegreeCentrality(self.influence_graph).run().scores()
            # normalize the probability so they sum up to 1
            probs = [d/sum(probs) for d in probs]
        else:
            raise ValueError("Unknown selection strategy: %s" %
                             seed_selection_strategy)

        # initialize the list of cascades
        cascades = []
        # get the list of node ids
        nodes = []
        self.influence_graph.forNodes(lambda v: nodes.append(v))

        # get the seeds of the propagation
        for k in tqdm.tqdm(runs):
            seeds = rn.choice(nodes, k, replace=False, p=probs)
            _, cascade = self.diffusion_model(self.influence_graph, seeds)
            cascades.append(cascade)

        return cascades
