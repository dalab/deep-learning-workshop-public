import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt # import if you want to plot

if __name__ == "__main__":
  # Parameters
  nrEpochs = 100
  learningRate = 0.01
  datsetPath = 'fourclass_scale.txt'
  logdirPath = "/tmp/regression"
  
  # Load the dataset, shuffle it and split it into train and test
  data = np.genfromtxt(datsetPath, delimiter=' ')
  tf.set_random_seed(42)
  np.random.seed(50)
  np.random.shuffle(data)
  test_data, train_data = np.split(data, [100])
  print("%d train datapoints, %d test datapoints" % (len(train_data), len(test_data)))

  # The summary write allows to save to graph definition 
  # for tensorboard later
  summary_writer =  tf.train.SummaryWriter(logdirPath)
  # Optional: Plot if pyplot is installed (comment-in the import)
  # data_to_plot = data
  # dataPos = data_to_plot[data_to_plot[:,0] == +1]
  # dataNeg = data_to_plot[data_to_plot[:,0] == -1]
  # print("%d %d" % (len(dataPos), len(dataNeg)))
  # plt.plot(dataPos[:,1], dataPos[:,2], 'go')
  # plt.plot(dataNeg[:,1], dataNeg[:,2], 'bo')
  # plt.show()
   
  
  # Inputs are our startingpoint for the graph
  #TODO Define input placeholders

  # Multiply inputs with softmax weights
  #TODO Define weights and multiply

  # Our loss is cross-entropy
  #TODO Define the classification loss between the groundtruth labels
  #     and our prediction

  # Use SGD with fixed learning rate
  #TODO Create an Optimizer for the loss
  
  
  

  with tf.Session() as session:
    # Init all variables
    #TODO Initialize variable
    
    # Write graph
    summary_writer.add_graph(session.graph)
    summary_writer.flush()
    
    # Training loop
    for epoch in range(nrEpochs):
      test_loss = 0.0 # Test loss
      correct = 0     # Count for classification accuracy

      # Compute the test error before the epoch
      for j in range(len(test_data)):
        #TODO Compute predictions, accumulate the loss for every datapoint in the dev set
        # and count how many times we have been right 

           
   
      # Train by 
      train_loss = 0.0
      for i in range(len(train_data)):
        # Perform an update step for every datapoint and accumulate the training loss
       
     
      print("epoch %d:\t train loss = %f8\t test loss = %f8" % (epoch, train_loss / len(train_data), test_loss / len(test_data)))
      print("Accuracy is %f" % (float(correct) / len(test_data)))
  
