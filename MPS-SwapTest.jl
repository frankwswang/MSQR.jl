using Yao
using Yao.Blocks
using QuAlgorithmZoo 
using LinearAlgebra
using Random
using Test

#==#
"""Auther Tests only. """ 
push!(LOAD_PATH,"../MPSDiffCircuit.jl/src")
#==#
"""
Need to add another package if you want to run the code: 
https://github.com/frankwswang/MPSDiffCircuit.jl.git
"""
# =#
using MPSDiffCircuit 

function MScircuit(nBitT::Int64, vBit::Int64, depth::Int64, ϕ::Real)
    nBitG = 1 + vBit
    nBitA = 1 + nBitG + nBitT 
    Cblocks = []
    MPSblocks = []
    push!(Cblocks, put(nBitA, nBitA=>H))
    push!(Cblocks, put(nBitA, nBitA=>shift(ϕ)))
    println("s1")
    MPSGen = MPSDC(depth, nBitT, vBit)
    println("s1_2")
    for i = 1:(nBitT-vBit)
        push!( MPSblocks, MPSGen.cBlocks[i] ) 
        println("s2_1")
        MPSblock = put(nBitA, Tuple( (nBitA-1):-1:(nBitT+1), )=>MPSblocks[i] )
        println("s2_2")
        push!(Cblocks, MPSblock)
        SWAPblock = control(nBitA, nBitA, ( (nBitA-1), (nBitT+1-i) )=>SWAP)
        push!(Cblocks, SWAPblock)
        println("s3")
    end
    println("s4")
    SWAPvBit = chain(nBitA, [control(nBitA, nBitA, ( (nBitT+i),i )=>SWAP) for i=vBit:-1:1])
    println("s5")
    push!(Cblocks, SWAPvBit)
    println("s6")
    push!(Cblocks, put(nBitA, nBitA=>H))
    println("s7")
    circuit = chain(nBitA,Cblocks)
    #dispatch!(circuit[2], ϕ)
    println("s8")
    circuit
end

nBitT = 4
vBit = 2
depth = 1
ϕ = 0

c2 = MScircuit(nBitT, vBit, depth, ϕ)


