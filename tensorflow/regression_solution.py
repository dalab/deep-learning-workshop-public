import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt # import if you want to plot


# 100 epochs should end up at about 0.8 accuracy 

if __name__ == "__main__":
  # Parameters
  nrEpochs = 100
  initial_learningRate = 0.1
  datsetPath = '/home/schmiflo/Data/TOy/Fourclass/fourclass_scale'
  
  # Load the dataset, shuffle it and split it into train and test
  data = np.genfromtxt(datsetPath, delimiter=' ')
  tf.set_random_seed(42)
  np.random.seed(50)
  np.random.shuffle(data)
  test_data, train_data = np.split(data, [100])
  print("%d train datapoints, %d test datapoints" % (len(train_data), len(test_data)))

  summary_writer =  tf.train.SummaryWriter("/tmp/regression")
  # Optional: Plot if pyplot is installed (comment-in the import)
  # data_to_plot = data
  # dataPos = data_to_plot[data_to_plot[:,0] == +1]
  # dataNeg = data_to_plot[data_to_plot[:,0] == -1]
  # print("%d %d" % (len(dataPos), len(dataNeg)))
  # plt.plot(dataPos[:,1], dataPos[:,2], 'go')
  # plt.plot(dataNeg[:,1], dataNeg[:,2], 'bo')
  # plt.show()
   
  
  # Inputs are our startingpoint for the graph
  x = tf.placeholder(tf.float32, [1,2], "input")
  y_gold = tf.placeholder(tf.float32, [1,2], "label")

  # Layer 1
  init_width = 1.0 
  W = tf.Variable(tf.random_uniform([2,2], -init_width, init_width), name="W")
  b = tf.Variable(tf.zeros([1,2]), name="bias")
  y_1 = tf.sigmoid(tf.matmul(x, W) + b) 
  
  # Layer 2
  W2 = tf.Variable(tf.random_uniform([2,2], -init_width, init_width), name="W2")
  b2 = tf.Variable(tf.zeros([1,2]), name="bias2")
  y_2 = tf.sigmoid(tf.matmul(y_1, W2) + b2)

  # Softmax weights
  W_softmax = tf.Variable(tf.random_uniform([2,2], -init_width, init_width), name="softmax_weights")
  b_softmax = tf.Variable(tf.zeros([1,2]), name="softmax_bias")
  y_predicted = tf.matmul(y_2, W_softmax) + b_softmax

  # Our loss is cross-entropy
  loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_predicted, y_gold), name = "loss")

  # Use SGD with linear decaying lr from initial_learningRate to 0
  global_step = tf.Variable(0, name="global_step")
  total_nr_step = nrEpochs * len(train_data)
  lr = initial_learningRate * (1.0 - tf.cast(global_step, tf.float32) / total_nr_step)
  
  # Minimizer (keep track of the steps the optimizer took)
  optimizer = tf.train.GradientDescentOptimizer(lr)
  update_step = optimizer.minimize(loss, global_step)

  # Initialize
  init = tf.initialize_all_variables()
  

  with tf.Session() as session:
    # Init all variables and write the graph so that we can 
    # inspect it in tensorboard
    init = session.run(init)
    summary_writer.add_graph(session.graph)
    summary_writer.flush()
    
    # Training loop
    for epoch in range(nrEpochs):
      test_loss = 0.0 # Test loss
      correct = 0     # Count for classification accuracy

      # Compute the test error before the epoch
      for j in range(len(test_data)):
            # Feed datapoint and label as numpy data
            datapoint = [test_data[j,1:]]
            # We need an one-hot encoding
            label = [[0,1]] if test_data[j,0] == 1 else [[1,0]]
            feed_dict = {x : datapoint, y_gold : label}
            # We only want the loss and the prediction - no update step
            step_loss, pred = session.run([loss, y_predicted], feed_dict = feed_dict)
            test_loss += step_loss
            # Did we predict correctly?
            if pred[0][0] > pred[0][1] and label[0][0] == 1 or pred[0][0] < pred[0,1] and label[0][1] == 1:
              correct += 1
           
   
      # Train by 
      train_loss = 0.0
      for i in range(len(train_data)):
        # Feed datapoint and label as numpy data
        datapoint = [train_data[i,1:]]  
        label = [[0,1]] if train_data[i,0] == +1 else [[1,0]]
        feed_dict = {x : datapoint, y_gold : label}
        # Run the update and keep track of the training loss
        step_loss, _, current_lr = session.run([loss, update_step, lr], feed_dict = feed_dict)
        train_loss += step_loss
       
     
      print("epoch %d:\t train loss = %f8\t test loss = %f8\t(lr=%f6)" % (epoch, train_loss / len(train_data), test_loss / len(test_data), current_lr))
      print("Accuracy is %f" % (float(correct) / len(test_data)))
  