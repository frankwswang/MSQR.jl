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

function train(circuit, nMeasure::Int64, learningRate::Real, nTrain::Int64)
    dGates = collect(circuit, AbstractDiff)
    println("0 $(collect(circuit, AbstractDiff)[1])")
    println("0 $(dGates[1])")
    # println("t1")
    for i=1:nTrain
        mst = MSTest(regTar, circuit, vBit, rBit, nMeasure)
        # println("t2")
        grads = opdiff.(()->mst.regA, dGates,Ref(mst.witnessOp))
        # println("t3")
        dispatch!(+, dGates, grads.*learningRate)
        # println("t4")
        if (10*i)%nTrain==0
            println("Training Step $(i), overlap = $(mst.overlaps)")
            # println("overlap = $(expect(mst.witnessOp, mst.regA))")
            # println("regA measure: $(mst.regA.state)")
            println("Grads: $(grads[1])")
            # println("$(dGates[1])")
            # println("$(collect(circuit, AbstractDiff)[1])")
        end
        # println("t5")
    end
end

nBitT = 2
 vBit = 1
 rBit = 1
depth = 5
    ϕ = 0
nMeasure = 100
learningRate = 0.001
nTrain = 1000

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
train(circuit, nMeasure, learningRate, nTrain)