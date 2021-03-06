using Flux

"""
google uses a CIFG for word prediction
LSTM is built into Flux
now all we have to do is implement CIFG
damn I gotta figure this out myself
"""
 
function VGG16()
   Chain(
      Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(64),
      Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(64),
      MaxPool((2,2)),
      Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(128),
      Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(128),
      MaxPool((2,2)),
      Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(256),
      Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(256),
      Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(256),
      MaxPool((2,2)),
      Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(512),
      Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(512),
      Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(512),
      MaxPool((2,2)),
      Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(512),
      Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(512),
      Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
      BatchNorm(512),
      MaxPool((2,2)),
      flatten,
      Dense(512, 4096, relu),
      Dropout(0.5),
      Dense(4096, 4096, relu),
      Dropout(0.5),
      Dense(4096, 10)
   )
end
