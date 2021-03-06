from numpy.random import rand
from collections import defaultdict


class DiffusionModel:

    def __call__(self, graph, seeds):
        """ Run the diffusion model on input graph,
        starting from the provided seeds.


        Parameters
        ----------
        graph : networkit.Graph
          Diffusion graph
        seeds : sequence-like object  of int
          Seed nodes of the propagation

        Raises
        ------
        NotImplementedError
          if the subclass does not override this method

        Returns
        -------
        list of int
          list of active nodes at the end of the propagation process
        list of list of int
          list of nodes activation for each time step

        """
        raise NotImplementedError


class IndependendCascade(DiffusionModel):
    """ Independent Cascade model.

    This model runs a diffusion process according to the
    rules of the independent cascade model
    """

    def __init__(self):
        super().__init__()

    def __call__(self, graph, seeds):
        # dict containing all the active nodes regardless
        # their activation time step
        active_dict = defaultdict(bool, {s: True for s in seeds})
        # active nodes for each time step
        active_time_list = [[x for x, _ in active_dict.items()]]

        def try_activate(u, v, w):
            if not active_dict[v] and rand() <= w:
                active_dict[v] = True
                active_time_list[-1].append(v)

        while True:
            # get the last activated nodes
            last_activated = active_time_list[-1]
            # append a new to list for the nodes
            # that will be activated in this iteration
            active_time_list.append([])
            for v in last_activated:
                graph.forEdgesOf(v, lambda u, v, w, eid: try_activate(u, v, w))

            # if no activation happened in the last iteration then exit
            if len(active_time_list[-1]) == 0:
                # remove the list and break the loop
                active_time_list.pop()
                break

        active_list = [v for v, _ in filter(lambda x:x[1],
                                            active_dict.items())]
        return active_list, active_time_list


class LinearThresholdModel(DiffusionModel):
    """ Linear Threshold Model model.

    This model runs a diffusion process according to the
    rules of the linear threshold model
    """

    def __init__(self):
        super().__init__()

    def __call__(self, graph, seeds):
        # dict containing all the active nodes regardless
        # their activation time step
        active_dict = defaultdict(bool, {s: True for s in seeds})
        # active nodes for each time step
        active_time_list = [[x for x, _ in active_dict.items()]]
        # select the nodes thresholds
        node_thresholds = rand(graph.upperNodeIdBound()).tolist()
        # nodes perceived influence
        node_perceived_influence = defaultdict(float)

        def try_activate(u, v, w):
            if not active_dict[v]:
                if node_perceived_influence[v] + w >= node_thresholds[v]:
                    active_dict[v] = True
                    active_time_list[-1].append(v)
                else:
                    node_perceived_influence[v] += w

        while True:
            # get the last activated nodes
            last_activated = active_time_list[-1]
            # append a new to list for the nodes
            # that will be activated in this iteration
            active_time_list.append([])
            for v in last_activated:
                graph.forEdgesOf(v, lambda u, v, w, eid: try_activate(u, v, w))

            # if no activation happened in the last iteration then exit
            if len(active_time_list[-1]) == 0:
                # remove the list and break the loop
                active_time_list.pop()
                break

        active_list = [v for v, _ in filter(lambda x:x[1],
                                            active_dict.items())]
        return active_list, active_time_list
