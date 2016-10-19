#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from text_cnn import TextCNN
from tensorflow.contrib import learn
from gensim import models

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0001)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Data Parameters
tf.flags.DEFINE_string("training_file_pos", "twitter-datasets/train_pos.txt", "Path and name for the training file (pos examples)")
tf.flags.DEFINE_string("training_file_neg", "twitter-datasets/train_neg.txt", "Path and name for the training file (neg examples)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helper.load_data_and_labels(FLAGS.training_file_pos, FLAGS.training_file_neg)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))


# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# It's better to use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # TODO: Define your training procedure
        # TODO: Write your code here for creating summaries for the variables and the CNN loss and accuracy
        # TODO: Define a checkpoint directory


        # Vocabulary directory. Tensorflow assumes this directory already exists so we need to create it
        vocab_path = os.path.join(os.path.curdir, "vocabulary")
        vocab_dir = os.path.abspath(vocab_path)
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        # Write the vocabulary
        vocab_processor.save(os.path.join(vocab_path, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        # Initialize the embedding matrix using pretrained embeddings
        # Override the random initialization
        # Use 50-dimensional GloVe vectors
        model = models.Word2Vec.load_word2vec_format('word-embeddings/glove_model.txt', binary=False)
        # Alternatively, use 300-dimensional word2vec vectors - caveat: requires more memory
        #model = models.Word2Vec.load_word2vec_format('word-embeddings/GoogleNews-vectors-negative300.bin', binary=True)
        
        my_embedding_matrix = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        for word in vocab_processor.vocabulary_._mapping:
            id = vocab_processor.vocabulary_._mapping[word]
            if word in model.vocab:
                my_embedding_matrix[id] = model[word]  
            else:
                my_embedding_matrix[id] = np.random.uniform(low=-0.25, high=0.25, size=FLAGS.embedding_dim)

        pretrained_embeddings = tf.placeholder(tf.float32, [None, None], name="pretrained_embeddings")
        set_x = cnn.pretrained_embeddings.assign(pretrained_embeddings)
        sess.run(set_x, feed_dict={pretrained_embeddings: my_embedding_matrix})

        def train_step(x_batch, y_batch, print_loss = False):
            """
            A single training step
            """
            # TODO: Implement the training function


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            # TODO: Implement the function for evaluation on the dev set


        # Generate batches
        batches = data_helper.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop
        for i, batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, i % 20 == 0)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                print ("\nCreating checkpoint")
                # TODO: Save your model in the checkout directory

