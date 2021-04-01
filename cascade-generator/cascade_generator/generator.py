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

    def __call__(self, k, num_of_runs=1, seed_selection_strategy='degree',
                 beta=1.0):
        """ Method to generate the cascades

        Parameters
        ----------
        k: int or list-like of int of shape (num_of_runs,)
            number of initial seeds of the propagation.
            If k is a single integer than each simulation has
            the same number of initial seeds

        num_of_runs: int, default=1
            number of propagation process to run.

        seed_selection: {'degree', 'uniform', 'random-walk' }, default='degree'
           strategy to randomly select the seeds of the propagation

           - 'degree' nodes are selected according to their out degree.
             Nodes with high out-degree are more likely to be selected

           - 'uniform' nodes are selected according to a uniform distribution.
             Each node has the same probability 1/N to be selected, where N
             is the number of nodes in the graph

           - 'random-walk' nodes are selected according to a random walk
             strategy. This strategy requires the selection probability
             \beta which denotes the probability
             of a node to be inserted into the seed
             set once it has been reached by the random walker.

             More specifically, the process starts from a node v,
             selected uniformly at random, which is added to
             the seed set. Given v, the random walker jumps on
             one of v's out-neighbors (each neighbor is equally likely to
             be selected), call it u. u will be inserted into the
             seed set with probability \beta.
             The random walker proceeds until the seed set reaches the
             desired size

        beta: float, optional
          random walk selection probability

        Return
        ------
        cascades: list of cascades
          a list containing every cascade generated
        """
        if isinstance(k, tuple) or isinstance(k, list):
            runs = k
        else:
            runs = [k] * num_of_runs

        # determine the seed selection strategy
        if seed_selection_strategy in ('uniform', 'degree'):
            if seed_selection_strategy == 'uniform':
                probs = None
            else:
                probs = DegreeCentrality(self.influence_graph).run().scores()
                # normalize the probability so they sum up to 1
                probs = [d/sum(probs) for d in probs]

            def choice(nodes, k): return rn.choice(nodes, k, replace=False, p=probs)

        elif seed_selection_strategy == 'random-walk':
            def choice(nodes, k): return self.random_walk(nodes, k, beta)
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
            seeds = choice(nodes, k)
            _, cascade = self.diffusion_model(self.influence_graph, seeds)
            cascades.append(cascade)

        return cascades

    def random_walk(self, nodes, k, beta):
        """Random walk selection strategy

        Parameters
        ----------
        nodes: list of nodes
          nodes of the graph
        k: int
          size of the seed set
        beta: float
          selection probability

        Returns
        -------
        list of int
          seed set
        """
        def start():
            # start the random walker
            start_node = rn.choice(nodes)
            return start_node, set([start_node])

        current_node, seeds = start()
        while len(seeds) < k:
            try:
                # get v's neigbors
                neigh = [u for u in self.influence_graph.iterNeighbors(current_node)]
                # select one of the neigbors
                next_node = rn.choice(neigh)
                current_node = next_node
                if rn.rand() <= beta:
                    seeds.add(next_node)
            except ValueError:
                # some of the nodes has no out-neighbors
                # discard the entire selection and start from
                # the beginning
                current_node, seeds = start()

        return seeds
