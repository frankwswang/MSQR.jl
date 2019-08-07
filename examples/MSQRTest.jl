# push!(LOAD_PATH, abspath("src"))
# push!(LOAD_PATH, abspath("../MPSCircuit.jl/src"))
using Yao, MSQR, MPSCircuit

# Basic parameters.
nBitT = 4
vBit = 2
rBit = 1
depth = 2
nMeasure = 2000
regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, MPSGen.cBlocks)

# Setup Training environments
## Gradient Optimization using ADAM algorithm(https://arxiv.org/abs/1412.6980v8).
method1 = ("ADAM", 0.05)
nTrain1 = 20
## Gradient Optimization using fixed step size(default size=0.1).
method2 = ("default", 0.05)
nTrain2 = 20

# Training Program.
## ADAM based SGD method.
### MSQR training.
MSQRtrain!(regTar, deepcopy(circuit), nTrain1, nMeasure=nMeasure, GDmethod = method1, show=true, useCuYao=CUDA_ON)
### SWAP Test + SGD training.
SWAPtrain!(regTar, deepcopy(MPSGen.cExtend), nTrain1, nMeasure=nMeasure, GDmethod = method1, show=true, useCuYao=CUDA_ON)

## default SGD method.
### MSQR training.
MSQRtrain!(regTar, deepcopy(circuit), nTrain2, nMeasure=nMeasure, GDmethod = method2, show=true, useCuYao=CUDA_ON)
### SWAP Test + SGD training.
SWAPtrain!(regTar, deepcopy(MPSGen.cExtend), nTrain2, nMeasure=nMeasure, GDmethod = method2, show=true, useCuYao=CUDA_ON)