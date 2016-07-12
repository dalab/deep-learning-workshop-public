require 'nn'

local softsign = torch.class('nn.softsign', 'nn.Module')

function softsign:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  -- ...something here...
  return self.output
end

function softsign:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- ...something here...
  return self.gradInput
end

