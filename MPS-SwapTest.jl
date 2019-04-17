using Yao
using Yao.Blocks
using QuAlgorithmZoo 
using LinearAlgebra
using Statistics
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

## Sub-functions.
function MScircuit(nBitT::Int64, vBit::Int64, depth::Int64, ϕ::Real, MPSblocks::Array)
    nBitG = 1 + vBit
    nBitA = 1 + nBitG + nBitT 
    Cblocks = []
    push!(Cblocks, put(nBitA, nBitA=>H))
    push!(Cblocks, put(nBitA, nBitA=>shift(ϕ)))
    #println("s1")
    for i = 1:(nBitT-vBit)
        MPSblock = put(nBitA, Tuple( (nBitT+1):(nBitA-1), )=>MPSblocks[i] )
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
    #println("The circuit:\n $(circuit)")
    circuit
end

function MSTest(regT::DefaultRegister, MPSGen::MPSDC, vBit::Int64, 
                depth::Int64, ϕ::Real, nMeasure::Int64, Test::Bool=false)
    nBitT = nqubits(regT)
    circuit = MScircuit(nBitT, vBit, depth, ϕ, MPSGen.cBlocks)
    #println("MS circuit: \n$(circuit)\n")
    #println("MPS circuit: \n$(MPSGen.cExtend)\n")
    Overlap = []
    for i = 1:nMeasure 
        regA = join(zero_state(2+vBit),copy(regT))
        nBitA = 2 + nBitT
        regA |> circuit[1] |> circuit[2]
        for i = 3:2:( 3 + 2*(MPSGen.nBlock-2) )
            regA |> circuit[i] |> circuit[i+1]
            measure_reset!(regA, (nBitA-1), val = 0) 
            #println("circuit[$(i)]\n")
            #println("circuit[$(i+1)]\n")
        end    
        for i = ( 3 + 2*(MPSGen.nBlock-1) ):length(circuit)
            regA |> circuit[i]
            #println("circuit[$(i)]\n")
        end
        push!(Overlap, expect(put(nBitA, nBitA=>Z), regA) |> tr)
    end
    ActualOverlaps = mean(Overlap)
    res = ActualOverlaps
    if Test == true
        @testset "MPS-Swap Test Reliability Check" begin
            regG = zero_state(nBitT)
            regG |> MPSGen.cExtend
            # ExpectOverlaps = tr(mat(regG |> ρ)*mat(regT |> ρ))
            ExpectOverlaps = ((regG.state'*regT.state)[1]|>abs)^2
            @test  ActualOverlaps ≈ ExpectOverlaps
            res = push!([res], ExpectOverlaps) 
        end
    end
    res
    
end

## Parameters Setup.
nBitT = 4
 vBit = 1
depth = 1
    ϕ = 0

## Main Program.
regTar = rand_state(nBitT)
MPSGen = MPSDC("CS", nBitT, vBit)
c2 = MSTest(regTar, MPSGen, vBit, depth, ϕ, 1, true)
