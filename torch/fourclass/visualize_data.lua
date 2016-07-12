require 'torch'
require 'math'
require 'gnuplot'

local loader = require 'data_loader'
local train = require 'train'

torch.manualSeed(1)
data = loader.load_data()

local opt = {
  model_type = 'linear', -- Exercise: change the type of the model here, see options in create_model
  training_iterations = 100, -- note: the code uses *batches*, not *minibatches*, now.
  print_every = 25,          -- how many iterations to skip between printing the loss
}

-- Exercise 1: visualize the data
-- TODO: Modify the code to plot the data separately for each class

x = data.inputs:select(2,1)
y = data.inputs:select(2,2)

gnuplot.figure()
gnuplot.plot(x, y,'+')
-- gnuplot.plot({x, y,'+'}, {x, y,'+'}) -- passes several sets of parameters

