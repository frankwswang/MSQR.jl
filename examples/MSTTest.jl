push!(LOAD_PATH,abspath("src")) 
using Yao
using MPSCircuit
using MPSSwapTest

# MS Test of Cluster State.
nMeasure = 30000
nBitT = 5
vBit = 1
rBit = 1
ϕ = 0

regTar = rand_state(nBitT)
MPSGen = MPSC("CS", nBitT, vBit, rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
test1 = MSTTest(regTar, circuit, MPSGen.cExtend, vBit, rBit, nMeasure)

# MS Test of differentiable quantum state.
nMeasure = 20000
nBitT = 6
vBit = 2
rBit = 2
ϕ = 0
depth = 4

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
test2 = MSTTest(regTar, circuit, MPSGen.cExtend, vBit, rBit, nMeasure)