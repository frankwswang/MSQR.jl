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
push!(LOAD_PATH,"./MPSDiffCircuit.jl/src") 
# =#

using MPSDiffCircuit

## Sub-functions.
function MScircuit(nBitT::Int64, vBit::Int64, rBit::Int64, ϕ::Real, MPSblocks::Array)
    nBitG = rBit + vBit
    nBitA = 1 + nBitG + nBitT 
    if (nBitT - vBit -rBit) % rBit != 0 
        println("Error: nBlock is not integer!")
        return 0
    end
    nBlock = Int((nBitT - vBit) / rBit)
    Cblocks = []
    push!(Cblocks, put(nBitA, nBitA=>H))
    push!(Cblocks, put(nBitA, nBitA=>shift(ϕ)))
    # println("S1")
    for i = 1:nBlock
        MPSblock = put(nBitA, Tuple( (nBitT+1):(nBitA-1), )=>MPSblocks[i] )
        # println("S2")
        push!(Cblocks, MPSblock)
        SWAPblock = chain(nBitA, [control(nBitA, nBitA, ( (nBitA-1-irBit), (nBitT-(i-1)*rBit-irBit) )=>SWAP) for irBit = 0:rBit-1])
        push!(Cblocks, SWAPblock)
        # println("S3")
    end
    # println("S4")
    SWAPvBit = chain(nBitA, [control(nBitA, nBitA, ( (nBitT+i),i )=>SWAP) for i=vBit:-1:1])
    # println("S5")
    push!(Cblocks, SWAPvBit)
    push!(Cblocks, put(nBitA, nBitA=>H))
    # println("S6")
    circuit = chain(nBitA,Cblocks)
    #println("The circuit:\n $(circuit)")
    circuit
end

function MSTest(regT::DefaultRegister, MPSGen::MPSDC, vBit::Int64, rBit::Int64, 
                ϕ::Real, nMeasure::Int64, Test::Bool=false)
    nBitT = nqubits(regT)
    circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
    println("MPS-SwapTest Circuit: \n$(circuit)\n")
    #println("MPS circuit: \n$(MPSGen.cExtend)\n")
    Overlap = []
    println("Measure times:\n")
    for i = 1:nMeasure
        (i*10)%nMeasure==0 && println("nMeasure = $i")
        # println("S7") 
        regA = join(zero_state(1+rBit+vBit),copy(regT))
        nBitA = 1 + rBit + vBit + nBitT
        # println("regA: $(regA)")
        # println("S8\ncircuit[1]:\n$(circuit[1])\ncircuit[2]:\n$(circuit[2])")
        regA |> circuit[1] |> circuit[2]
        # println("S9")
        for i = 3:2:( 3 + 2*(MPSGen.nBlock-2) )
            regA |> circuit[i] |> circuit[i+1]
            measure_reset!(regA, Tuple(nBitA-rBit:nBitA-1,), val = 0) 
            # println("S10")
            #println("circuit[$(i)]\n")
            #println("circuit[$(i+1)]\n")
        end    
        for i = ( 3 + 2*(MPSGen.nBlock-1) ):length(circuit)
            regA |> circuit[i]
            # println("S11")
            #println("circuit[$(i)]\n")
        end
        push!(Overlap, expect(put(nBitA, nBitA=>Z), regA))
    end
    # println("S12")
    ActualOverlaps = mean(Overlap) |> real #Take the real part for simplicity since the imaginary part is 0.
    res = ActualOverlaps
    if Test == true
        @testset "MPS-Swap Test Reliability Check" begin
            regG = zero_state(nBitT)
            regG |> MPSGen.cExtend
            #ExpectOverlaps = tr(mat(regG |> ρ)*mat(regT |> ρ))
            ExpectOverlaps = ((regT.state'*regG.state)[1] |> abs)^2
            #println("regT: $(regT.state)\n regG: $(regG.state)\n")
            @test isapprox.(ActualOverlaps, ExpectOverlaps, atol=0.02*ExpectOverlaps)
            println("\nExpectOverlaps: $(ExpectOverlaps)\nActualOverlaps: $(ActualOverlaps)\n") 
            res = push!([res], ExpectOverlaps)
            # println("S13")
        end
    end
    res
    
end

## Parameters Setup.
nBitT = 9
 vBit = 3
 rBit = 3
depth = 2
    ϕ = 0


## Main Program.
# regTar = rand_state(6)
# MPSGen = MPSDC("CS", 6, 1, 1)
# c2 = MSTest(regTar, MPSGen, 1, 1, ϕ, 30000, true)

regTar = rand_state(nBitT)
MPSGen = MPSDC(("DC",depth),nBitT,vBit,rBit)
c2 = MSTest(regTar, MPSGen, vBit, rBit, ϕ, 10000, true)

