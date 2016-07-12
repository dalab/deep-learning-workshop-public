tds = require 'tds'

local vocab_size = 400000
local word_vecs_size = 50
local unk_w_id = 1 -- UNK word id

local word2id = tds.Hash()
local id2word = tds.Hash()
local M = torch.zeros(vocab_size + 1, word_vecs_size)

word2id['99eof99'] = unk_w_id 
id2word[unk_w_id] = '99eof99'

function split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

--Reading Contents
local id = 2
for line in io.lines(w2v_txtfilename) do
  local parts = split(line, " ")
  local word = parts[1]
  word2id[word] = id
  id2word[id] = word

  if id % 100000 == 0 then
    print('  Processed ' .. id .. ' lines')
  end
  for i=2, #parts do
    M[id][i-1] = tonumber(parts[i])
  end

  id = id + 1
end

glove = {}
glove.M = M
glove.word2id = word2id
glove.id2word = id2word

print('Writing t7 File for future usage. Next time Word2Vec loading will be faster!')
torch.save(w2v_t7filename, glove)

return glove
