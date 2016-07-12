----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
-- Modified by Aurelien Lucchi
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------

function isdir(fn)
    return path.exists(fn)
end

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -a,--strategy          (default 0)           standard | increasing
   -d,--rand          (default 0)           randomize if set to 1
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 1)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   -y,--type          (default "float")           float | cuda | cudann
]]

-- fix seed
torch.manualSeed(1)

-- additional parameters
opt.nEpochs = 1

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

print('Optimization ' .. opt.optimization)
print('LearningRate ' .. opt.learningRate)
print('CoefL2 ' .. opt.coefL2)
print('batchSize ' .. opt.batchSize)
print('nEpochs ' .. opt.nEpochs)


-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

log_dir = opt.save
print("log_dir = " .. log_dir)
os.execute("mkdir " .. log_dir)


----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

maxLogStep = 10 -- how often do we compute the loss
nEpochs = opt.nEpochs

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolution(1, 32, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolution(32, 64, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model:add(nn.Reshape(64*2*2))
      model:add(nn.Linear(64*2*2, 200))
      model:add(nn.Tanh())
      model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   print('Using CUDA')
   require 'cunn'
   require 'cudnn'   
   model = model:cuda()
   criterion = criterion:cuda()
elseif opt.type == 'cudnn' then
   cudnn.convert(model, cudnn)
end

----------------------------------------------------------------------

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 500
   nbTestingPatches = 100
   print('<warning> only using 200 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

-- produce a permutation of a vector containing numbers from 1 to n
function permute(n)
	local t = {}
	for i = 1, n do
	   t[i] = i
	end

	if opt.rand then
		for i = 1, n - 100 do
		   local j = math.random(i, n)
		   t[i], t[j] = t[j], t[i]
		end
	end

	return t
end

-- Save one of the images to disk
image.save(log_dir .. '/trainData10.png', trainData.data[10])

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(log_dir, 'train.log'))
testLogger = io.open(paths.concat(log_dir, "test.txt"), "w")
trainLoggerFile = io.open(paths.concat(log_dir, "train.txt"), "w")

-- training function
function train(dataset, dataset_size)
   -- epoch tracker
   epoch = epoch or 1
   nsteps = nsteps or 0
   logStep = logStep or 0

   -- local vars
   local time = sys.clock()

   local perm = permute(dataset_size)

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. '/' .. dataset_size .. ']')
   for t = 1,dataset_size,opt.batchSize do
      -- Exercise: create mini batch by first allocation two tensors using the following instructions:
      -- local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      -- local targets = torch.Tensor(opt.batchSize)
      local sample = dataset[t]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()

      nsteps = nsteps + opt.batchSize
      logStep = logStep + opt.batchSize

      if opt.type == 'cuda' then
        input = input:cuda()
        target = target:cuda()
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local output = model:forward(input)
         local f = criterion:forward(output, target)

         -- estimate df/dW
         local df_do = criterion:backward(output, target)
         model:backward(input, df_do)

         -- L2 penalty:
         if opt.coefL2 ~= 0 then
            -- Exercise :add l2 penalty (use opt.coefL2)
            -- f = f + ...
            -- gradParameters:add( ... )
         end

         -- update confusion
         confusion:add(output, target)

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         x,fs = optim.sgd(feval, parameters, sgdState)
	 	 loss = fs[#fs]
	 	 print('loss ', loss)

         -- disp progress
         xlua.progress(t, dataset_size)

      else
         error('unknown optimization method')
      end

      if logStep >= maxLogStep then

	  	logStep = 0 -- reset

        -- appends a word test to the last line of the file
	    local loss = prediction(trainData, trainData:size())
	    print('loss (full set) ' .. loss)
        trainLoggerFile:write(nsteps .. ' ' .. loss .. '\n')
        trainLoggerFile:flush()
      end

   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset_size
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- save/log current net
   local filename = paths.concat(log_dir, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   -- print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test()
      -- test samples
      local loss = prediction(testData, testData:size())
      print('test loss (full set) ' .. loss)

      testLogger:write(nsteps .. ' ' .. loss .. '\n')
      testLogger:flush()
end

-- prediction function
function prediction(dataset, dataset_size)

   local total_loss = 0
   for t = 1,dataset_size,opt.batchSize do
      -- Exercise: create mini batch
      -- local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      -- local targets = torch.Tensor(opt.batchSize)
      -- load new sample
      local sample = dataset[t]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()

      if opt.type == 'cuda' then
        input = input:cuda()
        target = target:cuda()
      end

      -- test samples
      local output = model:forward(input)
      local loss = criterion:forward(output, target)
      total_loss = total_loss + loss
   end
   total_loss = (total_loss / dataset_size) * opt.batchSize

   return total_loss
end

----------------------------------------------------------------------
-- and train!
--

-- Compute loss
local loss = prediction(trainData, trainData:size())
print('loss (full set) at iteration 0 ' .. loss)
-- appends a word test to the last line of the file
trainLoggerFile:write('0 ' .. loss .. '\n')
trainLoggerFile:flush()

for t = 1,nEpochs do
   -- train/test
   train(trainData, trainData:size())
   test()

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      -- testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      -- testLogger:plot()
   end
end

-- closes the open file
io.close(trainLoggerFile)

print('Done!')