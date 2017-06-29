import tensorflow as tf
import numpy as np


class CNN(object):
    def __init__(self, patch_size, num_filters_fist_layer, num_filters_second_layer,
                 size_fully_connected_layer, num_classes=10, image_size=784):
        # Placeholders for input of images, labels and dropout rate
        self.x = tf.placeholder(tf.float32, shape=[None, image_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # creates and returns a weight variable with given shape initialized with
        # a truncated normal distribution with stddev of 0.1
        def weight_variable(shape, nameVar):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=nameVar)

        # creates and returns a bias variable with given shape initialized with
        # a constant of 0.1
        def bias_variable(shape, nameVar):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=nameVar)

        # computes a 2D convolution for the input data x and the filter W
        # uses a stride of one and is zero padded so the output is the same size as the input
        # input shapes:
        # x is the input tensor - should be a 4-D tensor of shape [batch_size, in_height, in_width, in_channels]
        # W is the filter tensor - should be a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        # strides is a 4-D tensor that defines how the filter slides over the input tensor in each of the 4 dimensions
        # padding - if it is set to "SAME" it means that zero padding on every side of the input is introduced to
        # make the shapes match if needed such that the filter is centered at all the pixels of the image according
        # to the strides.
        # ex. if strides=[1, 1, 1, 1] and padding='SAME' the filter is centered at every pixel from the image
        # padding - if it is set to "VALID" it means that there is no padding.
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



        # performs max pooling over 2x2 blocks
        # input shapes:
        # x is the input tensor - should be a 4-D tensor of shape [batch_size, in_height, in_width, in_channels]
        # ksize has the same dimensionality as the input tensor. It defines the patch size. It extracts the max
        # value out of each such patch. Here the patch we define is a 2x2 block
        # strides is a 4-D tensor that defines how the patch slides over the input tensor
        # if the padding is "SAME" there is padding, if it is "VALID" there is no padding
        # For the SAME padding, the output height and width are computed as:
        #     out_height = ceil(float(in_height) / float(strides1))
        #     out_width = ceil(float(in_width) / float(strides[2]))
        # For the VALID padding, the output height and width are computed as:
        #     out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
        #     out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        # example: if x is an image of shape [2,3] and has 1 channel (so the input shape is [1, 2, 3, 1])
        # , we max pool with 2x2 kernel and the stride is 2
        # if the pad is VALID (valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID'))
        #  the output is of shape [1, 1, 1, 1]
        # if the pad is SAME we pad the image to the shape [2, 4];
        # (same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')) the output is of shape [1, 1, 2, 1]
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Create a convolution + maxpool layer for the first layer

        # define the filter
        self.W_conv1 = weight_variable([patch_size, patch_size, 1, num_filters_fist_layer], "filter_layer1")
        b_conv1 = bias_variable([num_filters_fist_layer], "bias_layer1")
        # reshape the data to a 4D tensor to fit into the convolution
        # the second and third dimensions correspond to image width and height,
        #  and the final dimension corresponds to the number of color channels
        # the first dimension is for the batch size; when we have -1 for one dimension when reshaping
        # it will dynamically calculate that dimension
        # example: if x is of shape [a, b*c, d] and we run tf.reshape([-1, b, c, d]), the first dimension will be "a"
        # this is useful when the batch size varies
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        # apply convolution, add the bias, apply relu, and then max pooling
        h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + b_conv1)
        # print h_conv1.get_shape() # the shape is [-1, 28, 28, 32]
        h_pool1 = max_pool_2x2(h_conv1)
        # print h_pool1.get_shape() # the shape is [-1, 14, 14, 32]

        # Create a second layer of convolution + maxpool
        # define the filter
        self.W_conv2 = weight_variable(
            [patch_size, patch_size, num_filters_fist_layer, num_filters_second_layer], "filter_layer2")
        b_conv2 = bias_variable([num_filters_second_layer], "bias_layer2")
        # apply convolution, add the bias, apply relu, and then max pooling
        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2) + b_conv2)
        # print h_conv2.get_shape() # the shape is [-1,  14, 14, num_filters_fist_layer] i.e. [-1,  14, 14, 64]
        h_pool2 = max_pool_2x2(h_conv2)
        # print h_pool2.get_shape() # the shape is [-1,  14, 14, num_filters_second_layer] i.e. [-1, 7, 7, 64]

        # Create a densely connected layer
        W_fc1 = weight_variable([7 * 7 * 64, size_fully_connected_layer], "W_fc1")
        b_fc1 = bias_variable([size_fully_connected_layer], "b_fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # the shape of h_fc1 is [-1, size_fully_connected_layer]

        # Add dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([size_fully_connected_layer, num_classes], "W_fc2")
        b_fc2 = bias_variable([num_classes], "b_fc2")
        l2_loss = 0.0
        l2_loss += tf.nn.l2_loss(W_fc2)
        l2_loss += tf.nn.l2_loss(b_fc2)
        self.y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1])) + 0.00001 * l2_loss

        # here the regularization rate is fixed, you should make this a hyperparameter
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))