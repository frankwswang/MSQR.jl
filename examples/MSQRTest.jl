using Yao
#==#
# Author Tests only.  
push!(LOAD_PATH,abspath("src")) 
# =#
using MSQR
using MPSSwapTest
using MPSCircuit

# nBitT = 6
#  vBit = 2
#  rBit = 2
# depth = 4

nBitT = 3
vBit = 1
rBit = 1
depth = 3
ϕ = 0
nMeasure = 10000
learningRate = 0.1
nTrain = 500

regTar = rand_state(nBitT)
println("m1")
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
par = MSQRpar(circuit, regTar, vBit, rBit)
println("m2")
MSQRtrain(par, nMeasure, learningRate, nTrain, show=true)