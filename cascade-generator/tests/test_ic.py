import unittest
import networkit as nk
from cascade_generator.diffusion import IndependendCascade, LinearThresholdModel
from cascade_generator.generator import CascadeGenerator


class ICTest(unittest.TestCase):

    def setUp(self):
        # create a graph
        self.g = nk.Graph(5, weighted=True, directed=True)
        # add connections
        self.g.addEdge(0, 1, 1)
        self.g.addEdge(1, 2, 1)
        self.g.addEdge(2, 3, 1)
        self.g.addEdge(3, 4, 0)

    @unittest.skip
    def testPropagation1(self):
        """Test the propagation"""
        ic = IndependendCascade()
        nodes, tlist = ic(self.g, (0,))
        self.assertEqual(tlist, [[0], [1], [2], [3]])
        self.assertEqual(nodes, [0, 1, 2, 3])

    @unittest.skip
    def testPropagation2(self):
        ic = IndependendCascade()
        nodes, tlist = ic(self.g, (0, 1))
        self.assertEqual(tlist, [[0, 1], [2], [3]])
        self.assertEqual(nodes, [0, 1, 2, 3])

    @unittest.skip
    def testGenerator(self):
        ic = IndependendCascade()
        gen = CascadeGenerator(self.g, ic)
        cascades = gen((1, 2,), seed_selection_strategy='degree')
        print(cascades)
        self.assertTrue(len(cascades), 2)

    @unittest.skip
    def testLT(self):
        lt = LinearThresholdModel()
        nodes, tlist = lt(self.g, (0, 1))
        self.assertEqual(tlist, [[0, 1], [2], [3]])
        self.assertEqual(nodes, [0, 1, 2, 3])

    def testGeneratorRandomWalker(self):
        ic = IndependendCascade()
        gen = CascadeGenerator(self.g, ic)
        nodes = []
        self.g.forNodes(lambda v: nodes.append(v))
        s = gen.random_walk(nodes, 10, 0.5)
        print(s)



