using Yao, Yao.Blocks
using QuAlgorithmZoo 
using LinearAlgebra
using Statistics
using Test

#==#
# Author Tests only.  
push!(LOAD_PATH,abspath("../MPSCircuit.jl/src")) 
# =#
using MPSCircuit
push!(LOAD_PATH,abspath("src")) 
using MPSSwapTest

# # MS Test of Cluster State.
# nMeasure = 25000
#    nBitT = 5
#     vBit = 1
#     rBit = 1
#        ϕ = 0

# regTar = rand_state(nBitT)
# MPSGen = MPSC("CS", nBitT, vBit, rBit)
# circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
# test1 = MSTTest(regTar, circuit, MPSGen.cExtend, vBit, rBit, nMeasure)

# MS Test of differentiable quantum state.
nMeasure = 4000
   nBitT = 5
    vBit = 2
    rBit = 1
   depth = 4
       ϕ = 0


regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
test2 = MSTTest(regTar, circuit, MPSGen.cExtend, vBit, rBit, nMeasure)