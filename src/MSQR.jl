module MSQR
export MSQRpar
export MSQRtrain

# include("MPSSwapTest.jl")

using Yao, Yao.ConstGate
using MPSCircuit
push!(LOAD_PATH,abspath("src")) 
using MPSSwapTest

mutable struct MSQRpar
    circuit::ChainBlock
    reg::DefaultRegister
    vBit::Int64
    rBit::Int64
end

function MSQRtrain(par::MSQRpar, nMeasure::Int64, learningRate::Real, nTrain::Int64; show::Bool=false)
    circuit = par.circuit
    regTar = par.reg
    vBit = par.vBit
    rBit = par.rBit
    dGates = collect_blocks(AbstractDiff, circuit)
    # dGates = collect(circuit, AbstractDiff)
    # println("dGates:\n$(dGates)")
    # println("type of dGates:\n$(typeof(dGates))")
    println("t1")
    for i=1:nTrain
        mst = MSTest(regTar, circuit, vBit, rBit, nMeasure)
        println("t2")
        grads = opdiff.(()->MSTest(regTar, circuit, vBit, rBit, nMeasure).regA, dGates,Ref(mst.witnessOp))
        println("t3")
        println("grads: $(grads)")
        println("dGates: $(dGates)")
        dispatch!.(+, dGates, grads.*learningRate)
        println("t4")
        if show == true
            if nTrain <= 100
                showCase = i%2==0
            else
                showCase = (20*i)%nTrain==0
            end 
            if  showCase
                println("Training Step $(i), overlap = $(mst.overlaps)")
            end
        end
    end
    circuit
end

end