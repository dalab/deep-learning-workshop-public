require 'nn'
require 'softsign'

function create_model(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  local n_inputs = 2
  local n_classes = 2

  
  local model = nn.Sequential()

  if opt.model_type == 'linear' then
    --  linear -> softmax
    model:add(nn.Linear(n_inputs, n_classes))
  elseif opt.model_type == 'sigmoid' then
    --  linear -> sigmoid -> softmax
    model:add(nn.Linear(n_inputs, n_classes))
    model:add(nn.Sigmoid())
  elseif opt.model_type == 'softsign' then
    model:add(nn.Linear(n_inputs, n_classes))
    model:add(nn.softsign())
  elseif opt.model_type == 'sigmoid_2layers' then
    -- TODO: add something here
  else
    error('undefined model_type ' .. tostring(opt.model_type))
  end

  model:add(nn.LogSoftMax())

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.ClassNLLCriterion()

  return model, criterion
end

return create_model
