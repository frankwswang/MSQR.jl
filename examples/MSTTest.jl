using Yao
using MSQR
using MPSCircuit

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
nMeasure = 15000
nBitT = 5
vBit = 2
rBit = 1
ϕ = 0
depth = 6

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
test2 = MSTTest(regTar, circuit, MPSGen.cExtend, vBit, rBit, nMeasure)