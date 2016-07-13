import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn

class TextRNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, cell_size = 128, num_layers = 1):
      
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.pretrained_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], dtype=tf.float32, minval=-1.0, maxval=1.0), trainable=True)


        # Define our cell
        single_cell = tf.nn.rnn_cell.BasicRNNCell(cell_size)
        
        # Possibly multi layer
        cell = single_cell
        if num_layers > 1:
          cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # Embedding layer 
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Input of shape [batch_size, sequence_length, embedding_size]
            self.embedded_tokens = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_x)
            
        
        # Prepare the input
        # the tensorflow rnn module requires sequences of [batch_size, input_dim] tensors,
        # so we need to split along the `sequence_length` dimension. Every split will be of 
        # size [batch_size, 1, embedding_size] so we squeeze out the singleton dimension
        input_seq = []
        for slot in tf.split(1, sequence_length, self.embedded_tokens):
          input_seq.append(tf.squeeze(slot, squeeze_dims = [1]))
          
        # Define the RNN given the cell and the inputs
        # We do not need the outputs at every layer, just the final encoder state
        _, encoder_state = rnn.rnn(cell, input_seq, dtype=tf.float32)
        
        # Add dropout
        with tf.name_scope("dropout"):
            encoder_state_drop = tf.nn.dropout(encoder_state, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[cell_size * num_layers, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #l2_loss += tf.nn.l2_loss(W)
            #l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(encoder_state_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) #+ l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")