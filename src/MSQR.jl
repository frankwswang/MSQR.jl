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
    # println("t1")
    for i=1:nTrain
        mst = MSTest(regTar, circuit, vBit, rBit, nMeasure)
        # println("t2")
        grads = opdiff.(()->MSTest(regTar, circuit, vBit, rBit, nMeasure).regA, dGates,Ref(mst.witnessOp))
        # println("t3")
        dispatch!(+, dGates, grads.*learningRate)
        # println("t4")
        if nTrain <= 100
            showCase = i%2==0
        else
            showCase = (20*i)%nTrain==0
        end 
        if  showCase
            println("Training Step $(i), overlap = $(mst.overlaps)")
            # println("Grads: $(grads[1])")
        end
        # println("t5")
    end
end

nBitT = 5
 vBit = 2
 rBit = 1
depth = 4
    ϕ = 0
nMeasure = 2000
learningRate = 0.1
nTrain = 500

regTar = rand_state(nBitT)
MPSGen = MPSC(("DC",depth),nBitT,vBit,rBit)
circuit = MScircuit(nBitT, vBit, rBit, ϕ, MPSGen.cBlocks)
train(circuit, nMeasure, learningRate, nTrain)