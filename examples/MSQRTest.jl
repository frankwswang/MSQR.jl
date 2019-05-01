using Yao
#==#
# Author Tests only.  
push!(LOAD_PATH,abspath("src")) 
# =#
using MSQR
using MPSSwapTest
using MPSCircuit

nBitT = 2
vBit = 1
rBit = 1
depth = 3
nMeasure = 1000

ϕ = 0
learningRate = 0.1
nTrain = 200

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
MPSGen.cBlocks
par0 = zeros(length(MPSGen.cBlocks)*6)

typeof(par0) == Array{Float64,1}

MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit, dBlocksPar=par0)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
circuit.
# circuit2 = copy(circuit)
regTar2 = copy(regTar)
par = MSQRpar(circuit, regTar, vBit, rBit)
par2 = MSQRpar(circuit2, regTar2, vBit, rBit)
MSQRtrain(par, nMeasure, nTrain, learningRate = "ADAM", show=true)
MSQRtrain(par2, nMeasure, nTrain, show=true)