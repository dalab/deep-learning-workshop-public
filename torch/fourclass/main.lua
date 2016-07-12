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

-- train sigmoid and requ versions
model_sigmoid, losses_sigmoid = train(opt, data)
-- TODO: uncomment once you implement softsign
--opt.model_type = 'softsign'
--model_softsign, losses_softsign = train(opt, data)

--------------------------------------------------------
-- EVALUATION STUFF: YOU CAN IGNORE ALL THIS CODE
-- NOTE: though we don't have a test set, but we'll plot the training loss curves
-- We won't know if we overfit, but we can see how flexible our model is.

models = { 
    --softsign = model_softsign,  -- TODO: uncomment once you implement softsign
    sigmoid = model_sigmoid 
}
for model_name, model in pairs(models) do
  -- classification error on train set
  local log_probs = model:forward(data.inputs)
  local _, predictions = torch.max(log_probs, 2)
  print(string.format('# correct for %s:', model_name))
  print(torch.mean(torch.eq(predictions:long(), data.targets:long()):double()))

  -- plot the predictions
  x = data.inputs:select(2,1)
  y = data.inputs:select(2,2)

  i1 = predictions:eq(1)

  x1 = x[i1]
  y1 = y[i1]

  i2 = predictions:eq(2)
  x2 = x[i2]
  y2 = y[i2]

  print('Number of points: class 1 = ' .. x1:size()[1] .. ' class 2 = ' .. x2:size()[1])

  gnuplot.figure()
  gnuplot.plot({x1, y1,'+'}, {x2, y2,'+'})

end
