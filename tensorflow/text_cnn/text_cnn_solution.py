import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # self.pretrained_embeddings is a matrix which is a lookup table - for every word in the vocabulary
        # it contains a low dimensional word vector representation
        self.pretrained_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), trainable=True)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = 0.0

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
            print(self.embedded_tokens_expanded.get_shape())

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # the filter should be a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
                # filter_height represents how many words the filter cover
                # filter_width is the same as the embedding size
                # in_channels is 1, and out_channels is the num_filters
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # for NLP tasks the stride is typically [1, 1, 1, 1] and in_channels=1
                conv = tf.nn.conv2d(
                    self.embedded_tokens_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # shape of conv [None, sequence_length, embedding_size, num_filters]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, axis=3)
        print(self.h_pool.get_shape())
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")