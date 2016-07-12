-- Similar to mnist-loader from last practical, loads 2d dataset train/test split
-- has load_train function that returns table like { inputs = ... , targets = ... }, a tensor for inputs and targets.

require 'torch'

local loader = {}

local function isempty(s)
  return s == nil or s == ''
end

local function label_str_to_index(label_str)
  -- print('label_str ' .. label_str)
  if label_str == '+1' then
    return 1
  else
    return 2
  end
end

function loader.load_data()
  -- load
  local data = {}
  data.inputs = {}
  data.targets = {}
  data.targets_by_name = {}

  local f = torch.DiskFile("fourclass_scale.txt", "r")
  f:quiet()

  local line =  f:readString("*l")
  while line ~= '' do
      -- print('line ' .. line)
      label, f1_str, f2_str = string.match(line, '([^,]+) ([^,]+) ([^,]+)')
      if(isempty(f1_str)) then
        f1 = 0
      else
        f1 = tonumber(string.sub(f1_str, 3, -1))
      end
      if(isempty(f2_str)) then
        f2 = 0
      else
        f2 = tonumber(string.sub(f2_str, 3, -1))
      end
      -- print('label = ' .. label .. ' f1 = ' .. f1 .. ' f2 = ' .. f2)      
      data.inputs[#data.inputs + 1] = {f1, f2}
      data.targets[#data.targets + 1] = label_str_to_index(label)
      line = f:readString("*l")
  end

  data.inputs = torch.Tensor(data.inputs)
  data.targets = torch.Tensor(data.targets)

  -- shuffle the dataset
  local shuffled_indices = torch.randperm(data.inputs:size(1)):long()
  -- creates a shuffled *copy*, with a new storage
  data.inputs = data.inputs:index(1, shuffled_indices):squeeze()
  data.targets = data.targets:index(1, shuffled_indices):squeeze()

  print('--------------------------------')
  print('Loaded. Sizes:')
  print('inputs', data.inputs:size())
  print('targets', data.targets:size())
  print('--------------------------------')

  return data
end

return loader

