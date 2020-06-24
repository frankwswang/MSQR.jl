push!(LOAD_PATH, abspath("./src"))
push!(LOAD_PATH, abspath("../MPSCircuit.jl/src"))
using Yao, MSQR, MPSCircuit

# MS Test of Cluster State.
nMeasure = 20000
nBitT = 5
vBit = 1
rBit = 1

regTar = rand_state(nBitT)
MPSGen = MPSC("CS", nBitT, vBit, rBit)
circuit = MScircuit(nBitT, vBit, rBit, MPSGen.mpsBlocks)
test1 = MSTtest(regTar, circuit, MPSGen.cExtend, nMeasure=nMeasure)
ExpectOverlap = test1.Eoverlap
ActualOverlap = test1.Aoverlap
println("ExpectOverlap: $(ExpectOverlap)\nActualOverlap: $(ActualOverlap)")
@show isapprox.(ActualOverlap, ExpectOverlap, atol=0.05*ExpectOverlap)
## MS Test of Extended MPS Cluster-state circuit.
ExtendOverlap = SWAPtest(zero_state(nBitT)|>MPSGen.cExtend, regTar, nMeasure=nMeasure).overlap
println("ExtendOverlap: $(ExtendOverlap)\n")
@show isapprox.(ExtendOverlap, ExpectOverlap, atol=0.05*ExpectOverlap)

# MS Test of differentiable quantum state.
nMeasure = 20000
nBitT = 4
vBit = 2
rBit = 1
depth = 5

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, MPSGen.mpsBlocks)
test2 = MSTtest(regTar, circuit, MPSGen.cExtend, nMeasure=nMeasure)
ExpectOverlap = test2.Eoverlap
ActualOverlap = test2.Aoverlap
println("ExpectOverlap: $(ExpectOverlap)\nActualOverlap: $(ActualOverlap)")
@show isapprox.(ActualOverlap, ExpectOverlap, atol=0.05*ExpectOverlap)
## MS Test of Extended MPS differentiable circuit.
ExtendOverlap = SWAPtest(zero_state(nBitT)|>MPSGen.cExtend, regTar, nMeasure=nMeasure).overlap
println("ExtendOverlap: $(ExtendOverlap)\n")
@show isapprox.(ExtendOverlap, ExpectOverlap, atol=0.05*ExpectOverlap)