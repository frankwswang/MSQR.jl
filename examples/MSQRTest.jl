push!(LOAD_PATH, abspath("src"))
using Yao
using MSQR
using MPSCircuit

# Basic parameters.
nBitT = 5
vBit = 2
rBit = 1
depth = 6
nMeasure = 15000
ϕ = 0

# Setup Training environments
## Gradient Optimization using ADAM algorithm(https://arxiv.org/abs/1412.6980v8).
regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
lnRate = "ADAM"
nTrain = 150

## Gradient Optimization using fixed step size(default size=0.1).
regTar2 = copy(regTar)
circuit2= deepcopy(circuit)
lnRate2 = 0.15
nTrain2 = 150

# Training Program.
## ADAM method.
par = MSQRpar(circuit, regTar, vBit, rBit)
MSQRtrain!(par, nMeasure, nTrain, learningRate = lnRate, show=true)

## default method.
par2 = MSQRpar(circuit2, regTar2, vBit, rBit)
MSQRtrain!(par2, nMeasure, nTrain2, learningRate = lnRate2, show=true)