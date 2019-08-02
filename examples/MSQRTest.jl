push!(LOAD_PATH, abspath("src"))
push!(LOAD_PATH, abspath("../MPSCircuit.jl/src"))
using Yao
using CuYao
using CuArrays
CuArrays.allowscalar(false)
using MSQR
using MPSCircuit
using Random
Random.seed!(1234)

# Basic parameters.
nBitT = 4
vBit = 2
rBit = 1
depth = 2
nMeasure = 4000
regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)

# Setup Training environments
## Gradient Optimization using ADAM algorithm(https://arxiv.org/abs/1412.6980v8).
circuit1 = MScircuit(nBitT, vBit, rBit, MPSGen.cBlocks)
method1 = ("ADAM", 0.05)
nTrain1 = 2

# ## Gradient Optimization using fixed step size(default size=0.1).
# circuit2 = deepcopy(circuit1)
# method2 = ("default", 0.05)
# nTrain2 = 50

# Training Program.
## ADAM method.
MSQRtrain!(regTar, circuit1, nMeasure, nTrain1, GDmethod = method1, show=true, useCuYao=true)
# MSQRtrain!(regTar, circuit1, nMeasure, nTrain1, GDmethod = method1, show=true)

## default method.
# MSQRtrain!(regTar, circuit2, nMeasure, nTrain2, GDmethod = method2, show=true)