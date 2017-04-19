import numpy as np
import tensorflow as tf
from tfsdp.utils import weight_variable, bias_variable

class BallTree:
    def __init__(self, data):
        self.data = data
        self.path_length = int(np.ceil(np.log2(data.shape[0])))
        self.num_leaves = 2**self.path_length
        self.num_nodes = self.num_leaves - 1
        self.paths = np.zeros((len(data), self.path_length), dtype=int)
        self.splits = np.zeros_like(self.paths)
        self.build_tree(np.arange(len(data)), 0, 0)

    def build_tree(self, subset, level, node_id):
        # Base case -- empty subset
        if len(subset) == 1:
            return node_id

        # Get the relevant points
        subdata = self.data[subset]

        # Add this node to the path of each point
        self.paths[subset,level] = node_id
        node_id += 1

        # Split the data somehow (default is highest variance)
        splitmask = self.split_mask(subdata)
        left = subset[~splitmask]
        right = subset[splitmask]
        self.splits[subset, level] = splitmask

        # print 'level {0} left: {1} right: {2} splits: {3}'.format(level, left, right, self.splits[subset,level])

        node_id = self.build_tree(left, level+1, node_id)
        node_id = self.build_tree(right, level+1, node_id)
        return node_id

    def split_mask(self, subdata):
        # Get the dimension with highest variance
        dim = np.argmax(subdata.var(axis=0))
        
        # Split in the middle
        splitval = np.median(subdata[:,dim])
        splitmask = subdata[:,dim] > splitval
        return splitmask

class OrdinalTree(BallTree):
    def __init__(self, data):
        data = np.arange(len(data))
        BallTree.__init__(self, data)

    def split_mask(self, subdata):
        splitmask = subdata > np.median(subdata)
        return splitmask

class FastUnivariateSDP:
    '''A dyadic decomposition model with trend filtering on the logits. This
    model smooths only a local region around the target node, making it much
    more scalable to larger spaces. This model is sometimes referred to as
    trendfiltering-multiscale elsewhere in the code. '''
    def __init__(self, input_layer, input_layer_size, num_classes, 
                    k=2, lam=0.005, neighbor_radius=5,
                    scope=None, **kwargs):

        self.tree = OrdinalTree(np.arange(num_classes))
        self.paths = tf.constant(self.tree.paths, tf.int32)
        self.signs = tf.constant(self.tree.splits * 2 - 1, tf.float32)
        self.neighborhoods = tf.constant(np.array([np.arange(max(0,x - neighbor_radius),
                                                    max(0,x - neighbor_radius) + 1 + neighbor_radius*2) 
                                            for x in np.arange(num_classes)], dtype=int), tf.int32)
        
        # Local trend filtering setup
        self._k = k
        self._neighbor_radius = neighbor_radius
        self._lam = lam
        
        with tf.variable_scope(scope or type(self).__name__):
            # Bit weird but W is transposed for compatibility with tf.gather
            # See the _compute_sampled_logits function for reference:
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py
            self._W = weight_variable([self._num_nodes, self._input_layer_size])
            self._b = bias_variable([self._num_nodes])

    def build(self, input_layer, labels):
        labels = tf.argmax(self._labels, 1)
        nodes = tf.gather(self.paths, labels)
        signs = tf.gather(self.signs, labels)
        W = tf.transpose(tf.gather(self._W, nodes), [0,2,1])
        b = tf.gather(self._b, nodes)
        logits = signs * (tf.matmul(input_layer, W) + b)
        logprobs = -tf.log(1 + tf.exp(logits))
        self._train_loss = logprobs
        self._test_loss = logprobs


    def dist(self):
        return self._density

    @property
    def labels(self):
        return None

    @property
    def density(self):
        return self._density

    @property
    def output(self):
        return None

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def test_loss(self):
        return self._test_loss







