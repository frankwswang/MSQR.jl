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

# MS Test of Cluster State.
nBitT = 6
 vBit = 1
 rBit = 1
    ϕ = 0

regTar = rand_state(nBitT)
MPSGen = MPSC("CS", nBitT, vBit, rBit)
c2 = MSTest(regTar, MPSGen, 1, 1, ϕ, 25000, true)

# MS Test of differentiable quantum state.
nBitT = 9
 vBit = 3
 rBit = 3
depth = 2
    ϕ = 0

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
c2 = MSTest(regTar, MPSGen, vBit, rBit, ϕ, 20000, true)