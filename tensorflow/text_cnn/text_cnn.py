import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    # TODO: Implement a CNN for sentiment classification
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # TODO: Add placeholders for the input, label and dropout
        # self.input_x should be the name of the placeholder for the input

        # self.pretrained_embeddings is a matrix which is a lookup table - for every word in the vocabulary
        # it contains a low dimensional word vector representation
        self.pretrained_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), trainable=True)


        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # embedding_lookup is the operation which does the actual embedding i.e. it replaces each word (represented
            # as the index of the word in the vocabulary) with its corresponding embedding from the lookup matrix
            # self.embedded_tokens will be a 3-dim tensor of shape [None, sequence_length, embedding_size] where None
            # will depend on the batch size
            self.embedded_tokens = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_x)
            # conv2d expects a 4-dim tensor of size [batch size, width, height, channels]. self.embedded_tokens doesn't
            # contain the dimension for the channel so we expand the tensor to have one more dimension manually
            self.embedded_tokens_expanded = tf.expand_dims(self.embedded_tokens, -1)

        # TODO: Create a convolution + maxpool layer for each filter size
            # the filter should be a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
            # filter_height represents how many words the filter cover
            # filter_width is the same as the embedding size
            # in_channels is 1, and out_channels is the num_filters
            # for NLP tasks the stride is typically [1, 1, 1, 1] and in_channels=1

        # TODO: Combine all the pooled features

        # TODO: (Optional) Add dropout and L2 regularization

        # TODO: Define Mean cross-entropy loss

        # TODO: Define Accuracy
