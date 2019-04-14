using Yao
using Yao.Blocks
using QuAlgorithmZoo 
using LinearAlgebra
using Random
using Test

"""
Using Matrix Product State Structure to test state overlaps between 2 wave-functions.

Need to add another package if you want to run the code: 
https://github.com/frankwswang/MPSDiffCircuit.jl.git
"""

#==#
# Author Tests only.  
push!(LOAD_PATH,"../MPSDiffCircuit.jl/src") 
# =#

using MPSDiffCircuit

function MScircuit(nBitT::Int64, vBit::Int64, depth::Int64, ϕ::Real, MPSblocks::Array)
    nBitG = 1 + vBit
    nBitA = 1 + nBitG + nBitT 
    Cblocks = []
    push!(Cblocks, put(nBitA, nBitA=>H))
    push!(Cblocks, put(nBitA, nBitA=>shift(ϕ)))
    #println("s1")
    for i = 1:(nBitT-vBit)
        MPSblock = put(nBitA, Tuple( (nBitA-1):-1:(nBitT+1), )=>MPSblocks[i] )
        #println("s2")
        push!(Cblocks, MPSblock)
        SWAPblock = control(nBitA, nBitA, ( (nBitA-1), (nBitT+1-i) )=>SWAP)
        push!(Cblocks, SWAPblock)
        #println("s3")
    end
    #println("s4")
    SWAPvBit = chain(nBitA, [control(nBitA, nBitA, ( (nBitT+i),i )=>SWAP) for i=vBit:-1:1])
    #println("s5")
    push!(Cblocks, SWAPvBit)
    push!(Cblocks, put(nBitA, nBitA=>H))
    #println("s6")
    circuit = chain(nBitA,Cblocks)
    #println("s7")
    circuit
end

function MSTest(nBitT::Int64, vBit::Int64, depth::Int64, ϕ::Real)
    MPSGen = MPSDC(depth, nBitT, vBit)
    MScircuit(nBitT, vBit, depth, ϕ, MPSGen.cBlocks)
end

nBitT = 4
vBit = 2
depth = 1
ϕ = 0

c2 = MSTest(nBitT, vBit, depth, ϕ)


