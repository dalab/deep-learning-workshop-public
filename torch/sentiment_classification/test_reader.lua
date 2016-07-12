----------------------------------------------------------------------
-- This script shows how to load the Glove vectors and compute
-- the neighbors to a given word.
--
-- Octavian Ganea, Aurelien Lucchi
----------------------------------------------------------------------


require 'torch'
require 'tds'
require 'nn'

run_unit_tests = 1
tensor_type = 'float'
word_vecs_norm = false

-- Loads pre-trained glove embeddings
w2v_txtfilename = 'glove.6B.50d.txt'
w2v_t7filename = w2v_txtfilename:sub(1,w2v_txtfilename:find('.txt') - 1) .. '.t7'
w2v_reader = 'glove_reader.lua'
word_vecs_size = 50
print('==> Loading Glove vectors of dim ' .. word_vecs_size .. ' :')

---------------------- Code: -----------------------

w2vutils = {}

if not paths.filep(w2v_t7filename) then
  print('  ---> t7 file NOT found. Loading w2v from the bin/txt file instead (slower).')
  w2vutils = require(w2v_reader)
else
  print('  ---> from t7 file.')
  w2vutils = torch.load(w2v_t7filename)
end

---------- Define additional functions -----------------
w2vutils.distance = function (self,vec,k)
  local k = k or 1
  local norm = vec:norm(2)
  vec:div(norm)
  local distances = torch.mv(self.M ,vec)
  distances , oldindex = torch.sort(distances,1,true)
  local returnwords = {}
  local returndistances = {}
  for i = 1,k do
    table.insert(returnwords, self.id2word[oldindex[i]])
    table.insert(returndistances, distances[i])
  end
  return returnwords
end

-- Query w2v table
w2vutils.seen_words = 0
w2vutils.unknown_words = 0

-- word -> id
w2vutils.getWordID = function (self,word)
  local id = self.word2id[word]
  self.seen_words  = self.seen_words  + 1
  if id == nil then
    self.unknown_words = self.unknown_words  + 1
    return unk_w_id
  end

  return id
end



-- word -> vec
w2vutils.getWordVec = function (self,word)
  local id = w2vutils:getWordID(word)
  return self.M[id]
end


-- Lookup table: ids -> tensor of vecs
local lookup_layer = nn.LookupTable
if tensor_type == 'cudafb' then
  lookup_layer = nn.LookupTableGPU
end
w2vutils.lookup = lookup_layer(w2vutils.M:size(1), word_vecs_size)

w2vutils.lookup.weight = w2vutils.M:double()
if string.find(tensor_type, 'cuda') then
  w2vutils.lookup = w2vutils.lookup:cuda()
end
if string.find(tensor_type, 'cudacudnn') then
  cudnn.convert(w2vutils.lookup, cudnn)
end

w2vutils.reinitLookup = function (self)
  self.lookup.weight = correct_type(w2vutils.M)
end


w2vutils.lookupWordVecs = function (self,word_id_tensor)
  return correct_type(self.lookup:forward(word_id_tensor))
end

-- Normalize word vectors to have norm 1 .
w2vutils.renormalize = function (self)
  if word_vecs_norm == 'true' then
    w2vutils.lookup.weight[unk_w_id]:mul(0)
    w2vutils.lookup.weight[unk_w_id]:add(1)
    w2vutils.lookup.weight:cdiv(w2vutils.lookup.weight:norm(2,2):expand(w2vutils.lookup.weight:size()))
    local x = correct_type(w2vutils.lookup.weight:norm(2,2):view(-1)) - correct_type(torch.ones(w2vutils.lookup.weight:size(1)))
    assert(x:norm() < 0.1, x:norm())
    assert(w2vutils.lookup.weight[100]:norm() < 1.001 and w2vutils.lookup.weight[100]:norm() > 0.99)
    w2vutils.lookup.weight[unk_w_id]:mul(0)
  end
end

w2vutils:renormalize()

print('Done reading w2v data. Word vocab size = ' .. w2vutils.M:size(1))



w2vutils.unk_words_stats = function(self)
  local perc = (self.unknown_words + 0.0) * 100.0 / self.seen_words 
  print('\nMissing word vectors = ' .. perc .. '%')
end

w2vutils.most_similar = function(self, word, k)
  local k = k or 1
  local v = w2vutils:getWordVec(word)
  neighbors = w2vutils:distance(v,k)
  print('Most similar to ' .. word .. ' : ')  
  print(neighbors)
end

--------------------- Unit tests ----------------------------------------
if (run_unit_tests) then
  print('\n')
  w2vutils:most_similar('cat', 5)
  w2vutils:most_similar('france', 5)
  w2vutils:most_similar('hello', 5)
end
