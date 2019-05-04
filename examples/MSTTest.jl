push!(LOAD_PATH, abspath("src"))
using Test
using Yao
using MSQR
using MPSCircuit

@testset "MPS-Swap Test Reliability Check" begin

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
ExpectOverlaps = test1.Eoverlaps
ActualOverlaps = test1.Aoverlaps
println("\nExpectOverlaps: $(ExpectOverlaps)\nActualOverlaps: $(ActualOverlaps)\n")
@test isapprox.(ActualOverlaps, ExpectOverlaps, atol=0.02*ExpectOverlaps)

# MS Test of differentiable quantum state.
nMeasure = 12000
nBitT = 5
vBit = 2
rBit = 1
ϕ = 0
depth = 6

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
test2 = MSTTest(regTar, circuit, MPSGen.cExtend, vBit, rBit, nMeasure)
ExpectOverlaps = test2.Eoverlaps
ActualOverlaps = test2.Aoverlaps
println("\nExpectOverlaps: $(ExpectOverlaps)\nActualOverlaps: $(ActualOverlaps)\n")
@test isapprox.(ActualOverlaps, ExpectOverlaps, atol=0.02*ExpectOverlaps)

end