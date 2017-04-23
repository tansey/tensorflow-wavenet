import numpy as np
import tensorflow as tf
from tfsdp.utils import weight_variable, bias_variable, trend_filtering_penalty

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

def build_neighborhoods(num_classes, neighbor_radius):
    neighborhood_size = 2 * neighbor_radius + 1
    neighborhoods = np.zeros((num_classes, neighborhood_size), dtype=int)
    for x in np.arange(num_classes):
        if x < neighbor_radius:
            start = 0
            end = neighborhood_size
        elif (x + neighbor_radius) >= num_classes:
            start = num_classes - neighborhood_size
            end = num_classes
        else:
            start = x - neighbor_radius
            end = x + neighbor_radius + 1
        neighborhoods[x] = np.arange(start, end)
    return neighborhoods

class FastUnivariateSDP:
    '''A dyadic decomposition model with trend filtering on the logits. This
    model smooths only a local region around the target node, making it much
    more scalable to larger spaces. This model is sometimes referred to as
    trendfiltering-multiscale elsewhere in the code. '''
    def __init__(self, #input_layer_size,
                    num_classes, 
                    k=2, lam=0.005, neighbor_radius=5,
                    scope=None, **kwargs):

        self.tree = OrdinalTree(np.arange(num_classes))
        self.paths = tf.constant(self.tree.paths, tf.int32)
        self.splits = tf.constant(self.tree.splits, tf.int32)
        self.signs = tf.constant(-(self.tree.splits * 2 - 1), tf.float32)
        self.neighborhoods = tf.constant(build_neighborhoods(num_classes, neighbor_radius), tf.int32)
        self.neighborhood_size = neighbor_radius*2+1

        # Local trend filtering setup
        self._k = k
        self._neighbor_radius = neighbor_radius
        self._lam = lam
        
        # with tf.variable_scope(scope or type(self).__name__):
        #     # Bit weird but W is transposed for compatibility with tf.gather
        #     # See the _compute_sampled_logits function for reference:
        #     # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py
        #     self._W = weight_variable([self.tree.num_nodes, input_layer_size])
        #     self._b = bias_variable([self.tree.num_nodes])

    def build(self, logits, labels):
        # Convert from 1-hot
        labels = tf.argmax(labels, 1)

        # Build the loss functions
        logprobs = self._logprobs(logits, labels)
        self._train_loss = -tf.reduce_mean(logprobs)
        self._test_loss = -tf.reduce_mean(logprobs)

        if self._lam > 0:
            regularizer = self._trend_filtering(logits, labels)
            # regularizer = tf.Print(regularizer, [self._train_loss, regularizer], message='Loss vs. Regularizer:')
            self._train_loss += self._lam * regularizer

    def _logprobs(self, logits, labels):
        nodes = tf.gather(self.paths, labels)
        signs = tf.gather(self.signs, labels)
        logprobs = tf.map_fn(lambda (l, n, s): -tf.reduce_sum(tf.log(1 + tf.exp(s * tf.gather(l, n))), axis=-1), [logits, nodes, signs], dtype=tf.float32, parallel_iterations=100)
        return logprobs

    def _trend_filtering(self, logits, labels):
        neighbors = tf.gather(self.neighborhoods, labels) # [batchsize, neighborhood_size]
        logprobs = self._logprobs(logits, neighbors)
        return trend_filtering_penalty(logprobs, self.neighborhoods.get_shape()[1], self._k)

    def _build_density(self, input_layer):
        W = tf.transpose(self._W)
        b = self._b
        logits = tf.matmul(input_layer, W) + b
        left_probs = 1. / (1. + tf.exp(-logits))

        probs = tf.map_fn(lambda x: tf.reduce_prod(tf.where(self.signs < 0, tf.gather(x, self.paths), 1 - tf.gather(x, self.paths)), axis=1), left_probs)
        probs = probs / tf.reduce_sum(probs, axis=1, keep_dims=True)
        return probs

    @property
    def density(self):
        return self._density

    @property
    def train_loss(self):
        return self._train_loss

    @property
    def test_loss(self):
        return self._test_loss







